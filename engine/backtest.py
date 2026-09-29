"""
A 股回测器（按 Qlib TopkDropout 的规则复刻，另外支持周期调仓+缓冲区、拥挤度控仓）

规则：T-1 日收盘后的分数 → T 日收盘价成交；涨停/跌停/停牌当天不能交易；
      买入费率 open_cost，卖出费率 close_cost（含印花税），滑点 slip 双边另计。
用法（Python）：
    from engine.backtest import Backtester
    bt = Backtester("2023-01-01", "2026-09-18")
    res = bt.run(pred_df, mode="dropout", topk=50, n_drop=5)
    bt.summary(res)
"""
import os

import numpy as np
import pandas as pd

from engine.common import DATA, read_parts

N_YEAR = 238  # 与 Qlib 年化口径一致


class Backtester:
    def __init__(self, start, end, st_filter=False):
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        pre = self.start - pd.Timedelta(days=20)
        cols = ["datetime", "instrument", "ret", "up_lim", "dn_lim", "susp", "in_pool", "raw_close", "change"]
        p = read_parts(os.path.join(DATA, "panel"), columns=cols, start=pre, end=self.end)
        b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
        self.dates = [d for d in b.index if pre <= d <= self.end]
        self.bench = b.reindex(self.dates).fillna(0.0)
        wide = lambda c: p.pivot(index="datetime", columns="instrument", values=c).reindex(self.dates)  # noqa
        self.R = wide("ret").fillna(0.0).astype("float32")
        self.insts = self.R.columns
        self.block = (wide("up_lim").fillna(False) | wide("dn_lim").fillna(False) |
                      wide("susp").fillna(True)).astype(bool)   # 涨跌停/停牌：不能交易（Qlib forbid_all_trade_at_limit）
        self.inpool = wide("in_pool").fillna(False).astype(bool)
        self.px = wide("raw_close").ffill().astype("float32")   # 未复权价：用于一手 100 股取整
        self.change = wide("change").astype("float32")
        self.block_board = self.block.copy()
        self.has_open = False
        try:   # 开盘成交所需字段（新版面板才有）
            q = read_parts(os.path.join(DATA, "panel"), columns=["datetime", "instrument", "ret_on", "ret_id", "up_open",
                                                                  "dn_open", "raw_open"], start=pre, end=self.end)
            wq = lambda c: q.pivot(index="datetime", columns="instrument", values=c).reindex(index=self.dates, columns=self.insts)  # noqa
            self.R_on, self.R_id = wq("ret_on").fillna(0.0).astype("float32"), wq("ret_id").fillna(0.0).astype("float32")
            self.block_open = (wq("up_open").fillna(False) | wq("dn_open").fillna(False) |
                               wide("susp").reindex(columns=self.insts).fillna(True)).astype(bool)
            self.px_open = wq("raw_open").ffill().astype("float32")
            self.has_open = True
            del q
        except Exception:  # noqa
            pass
        del p

    def use_limit_rule(self, rule="board"):
        """board: 按板块真实涨跌停幅度（默认，更真实）；qlib: |涨跌幅|>=9.5% 一律不能交易（Qlib 默认口径）"""
        if rule == "qlib":
            susp = self.block_board & ~(self.change.abs() > 0).fillna(False)
            self.block = ((self.change.abs() >= 0.095).fillna(False) | susp | self.block_board & self.change.isna())
        else:
            self.block = self.block_board.copy()

    # ------------------------------------------------------------------
    def run(self, pred, mode="dropout", topk=50, n_drop=5, rebalance=5, buffer=1.5,
            open_cost=0.0005, close_cost=0.0010, slip=0.0, risk_degree=0.95, account=1e6, lot=True,
            min_cost=5.0, every=1, exec_at="close"):
        S = pred.pivot(index="datetime", columns="instrument", values="score")
        S = S.reindex(index=self.dates, columns=self.insts)
        S = S.where(self.inpool)                    # 只在股票池内选
        test_dates = [d for d in self.dates if d >= self.start]
        i0 = self.dates.index(test_dates[0])
        Rv, Bv, Sv, Pv = self.R.values, self.block.values, S.values, self.px.values
        at_open = exec_at == "open"
        if at_open:   # T 日收盘出信号 → T+1 开盘成交：先吃隔夜收益，再交易，再吃日内收益
            Ron, Rid, Bv, Pv = self.R_on.values, self.R_id.values, self.block_open.values, self.px_open.values
        lot_size = np.array([200 if str(c).startswith("SH688") else 100 for c in self.insts])
        col = {c: j for j, c in enumerate(self.insts)}
        oc, cc = open_cost + slip, close_cost + slip

        hold = {}  # j -> value
        cash = account
        rec = []
        for t in range(i0, len(self.dates)):
            # 1) 盯市
            for j in hold:
                hold[j] *= 1.0 + (Ron[t, j] if at_open else Rv[t, j])
            v0 = cash + sum(hold.values())
            s = Sv[t - 1]                              # 前一日分数
            valid = ~np.isnan(s)
            traded = 0.0
            fee = 0.0
            do_trade = valid.any() and ((mode == "dropout" and (t - i0) % every == 0) or (mode != "dropout" and (t - i0) % rebalance == 0))
            if do_trade:
                order = np.argsort(-np.where(valid, s, -np.inf))
                order = order[: valid.sum()]
                score_of = lambda j: s[j] if not np.isnan(s[j]) else -np.inf  # noqa
                if mode == "dropout":
                    last = sorted(hold, key=score_of, reverse=True)
                    lastset = set(last)
                    n_add = n_drop + topk - len(last)
                    today = [j for j in order if j not in lastset][: max(n_add, 0)]
                    comb = sorted(last + today, key=score_of, reverse=True)
                    worst = set(comb[-n_drop:]) if n_drop > 0 else set()
                    sell = [j for j in last if j in worst]
                    buy = today[: len(sell) + topk - len(last)]
                else:  # 周期调仓 + 缓冲区：排名仍在 topk*buffer 以内的老仓位不动
                    rank = {j: r for r, j in enumerate(order)}
                    keep = [j for j in hold if rank.get(j, 1e9) < topk * buffer]
                    sell = [j for j in hold if j not in set(keep)]
                    n_need = topk - len(keep)
                    buy = [j for j in order if j not in hold][: max(n_need, 0)]
                # 卖出
                for j in sell:
                    if Bv[t, j]:
                        continue
                    val = hold.pop(j)
                    f = max(val * cc, min_cost) if min_cost else val * cc
                    cash += val - f
                    traded += val
                    fee += f
                # 买入（Qlib 口径：可用现金 * risk_degree 均分）
                buy = [j for j in buy if not Bv[t, j]]
                if buy:
                    budget = max(cash - (1 - risk_degree) * (cash + sum(hold.values())), 0)
                    per = budget / len(buy)
                    per = min(per, v0 * risk_degree / topk * 1.5)
                    for j in buy:
                        amt = per / (1 + oc)
                        if lot:  # A 股一手 100 股（科创板 200 股）向下取整
                            unit = Pv[t, j] * lot_size[j]
                            if not unit > 0:
                                continue
                            amt = np.floor(amt / unit) * unit
                            if amt <= 0:
                                continue
                        f = max(amt * oc, min_cost) if min_cost else amt * oc
                        hold[j] = hold.get(j, 0.0) + amt
                        cash -= amt + f
                        traded += amt
                        fee += f
            if at_open:
                for j in hold:
                    hold[j] *= 1.0 + Rid[t, j]
            v1 = cash + sum(hold.values())
            rec.append((self.dates[t], v1, traded / max(v0, 1), fee / max(v0, 1), len(hold)))
        df = pd.DataFrame(rec, columns=["datetime", "value", "turnover", "cost", "n_hold"]).set_index("datetime")
        prev = pd.Series([account] + df["value"].tolist()[:-1], index=df.index)
        df["return"] = df["value"] / prev - 1
        df["bench"] = self.bench.reindex(df.index).values
        df["excess"] = df["return"] - df["bench"]
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def overlay(res, exposure, switch_cost=0.0002):
        """拥挤度控仓：未暴露部分换成指数（期货/ETF），超额按 exposure 缩放。exposure 必须只用 T-1 及以前的信息。"""
        e = exposure.reindex(res.index).ffill().fillna(1.0).clip(0, 1)
        out = res.copy()
        chg = e.diff().abs().fillna(0)
        out["return"] = e * res["return"] + (1 - e) * res["bench"] - chg * switch_cost
        out["excess"] = out["return"] - out["bench"]
        out["exposure"] = e
        return out

    @staticmethod
    def metrics(res):
        ex = res["excess"]
        cum = ex.cumsum()
        nav = (1 + res["return"]).cumprod()
        yearly = ex.groupby(ex.index.year).apply(lambda x: (1 + x).prod() - 1)
        m = {
            "超额年化": ex.mean() * N_YEAR,
            "信息比率": ex.mean() / (ex.std() + 1e-12) * np.sqrt(N_YEAR),
            "超额回撤": (cum - cum.cummax()).min(),
            "策略年化": nav.iloc[-1] ** (N_YEAR / len(nav)) - 1,
            "策略回撤": (nav / nav.cummax() - 1).min(),
            "日均换手": res["turnover"].mean(),
            "年化成本": res["cost"].mean() * N_YEAR,
        }
        m.update({f"超额{y}": v for y, v in yearly.items()})
        return m

    @staticmethod
    def plot(results: dict, path, title=""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Micro Hei",
                                           "SimHei", "Microsoft YaHei", "PingFang SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for k, r in results.items():
            (1 + r["excess"]).cumprod().plot(ax=ax[0], label=k)
        ax[0].set_title(title or "cumulative excess return vs CSI1000 (after cost)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        first = list(results.values())[0]
        (1 + first["return"]).cumprod().plot(ax=ax[1], label=f"strategy: {list(results)[0]}")
        (1 + first["bench"]).cumprod().plot(ax=ax[1], label="CSI1000")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(path, dpi=105); plt.close(fig)
