"""
进攻版 S_top3 模拟盘：一个 10 万元账户，每个交易日按 senti/live.py 的清单自动成交、记账、出下一日清单
  python paper/sim.py run            # 处理所有未处理的交易日（开盘成交 → 收盘盯市 → 生成下一日清单），可重复运行
  python paper/sim.py summary        # 只重新生成 paper/README.md

每个交易日 d：
  1) 按 d 日开盘价执行上一交易日的清单：先卖（开盘一字跌停/停牌卖不出 → 继续持有），再买（开盘一字涨停/停牌/资金不足一手 → 按顺序用替补）
     费用：买入万5、卖出千1（含印花税），每笔最低 5 元；单边滑点千1
  2) d 日收盘价盯市（除权除息按复权因子折算股数，相当于分红再投资）
  3) 用账户的真实持仓和现金调用 senti/live.py 生成 d+1 的清单
状态：paper/accounts/S_top3/（state.json / nav.csv / trades.csv / positions.csv）；每日清单 paper/signals/<日期>/
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PAPER = os.path.join(ROOT, "paper")
ACC_DIR = os.path.join(PAPER, "accounts")
SIG_DIR = os.path.join(PAPER, "signals")
INIT_CASH = float(os.environ.get("PAPER_CASH", 100_000))

STRATS = {
    "S_top3": dict(kind="conc", family="S",
                   desc="进攻版：排名平均（大涨概率 − 大跌概率，六个 5 日模型），持 3 只，跌出前 60 才卖，不择时，开盘成交"),
}

ETF_CODE = "512100"
# 数据源的复权因子每天有约 0.01% 的计算噪声；变化超过 0.3% 才认为是除权除息（更小的分红忽略）
ADJ_TOL = 0.003


# ============================================================================ 工具
def inst_of(code):
    """'603103.SH' -> 'SH603103'"""
    c = str(code).strip()
    if "." in c:
        n, ex = c.split(".")
        return ex.upper() + n
    from strategies.live import norm_code
    return norm_code(c)


def code_of(inst):
    return inst[2:] + "." + inst[:2]


def lot_of(inst):
    return 200 if inst.startswith("SH688") else 100


def fee(amount, rate, min_cost=5.0, stamp=0.0):
    if amount <= 0:
        return 0.0
    return max(amount * rate, min_cost) + amount * stamp


def trading_days():
    from engine.common import PROVIDER
    cal = pd.read_csv(os.path.join(PROVIDER, "calendars", "day.txt"), header=None)[0]
    return [pd.Timestamp(x) for x in cal]


class Market:
    """某一日的行情（未复权价、复权因子、涨跌停/停牌），带缓存"""

    COLS = ["datetime", "instrument", "raw_close", "raw_open", "factor", "up_lim", "dn_lim", "up_open", "dn_open", "susp"]

    def __init__(self):
        from engine.common import DATA
        self.DATA = DATA
        self.cache = {}
        b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
        self.bench = b

    def day(self, d):
        d = pd.Timestamp(d)
        if d not in self.cache:
            from engine.common import read_parts
            p = read_parts(os.path.join(self.DATA, "panel"), columns=self.COLS, start=d, end=d)
            self.cache[d] = p.set_index("instrument")
        return self.cache[d]

    def quote(self, d, inst):
        """返回 dict(close, open, factor, up_lim, dn_lim, up_open, dn_open, susp)；面板外的股票从 Qlib 全市场取"""
        t = self.day(d)
        if inst in t.index:
            r = t.loc[inst]
            q = {k: r[k] for k in self.COLS[2:]}
        else:
            q = self._qlib_quote(d, inst)
        for k in ("raw_close", "raw_open", "factor"):
            q[k] = float(q[k]) if q[k] is not None and not pd.isna(q[k]) else float("nan")
        for k in ("up_lim", "dn_lim", "up_open", "dn_open", "susp"):
            q[k] = bool(q[k]) if q[k] is not None and not pd.isna(q[k]) else (k == "susp")
        for k in ("raw_close", "raw_open"):
            if q[k] > 0:
                q[k] = round(q[k], 2)       # 复权价反推的未复权价有微小误差，还原到分
        if not q["raw_close"] > 0:
            q["susp"] = True
        return q

    def _qlib_quote(self, d, inst):
        from engine.common import init_qlib
        from qlib.data import D
        init_qlib()
        try:
            f = D.features([inst], ["$close/$factor", "$open/$factor", "$factor", "$volume", "$change"], d, d)
            r = f.iloc[0].values
            vol = r[3]
            return dict(raw_close=r[0], raw_open=r[1], factor=r[2], up_lim=False, dn_lim=False, up_open=False,
                        dn_open=False, susp=not (vol > 0))
        except Exception:  # noqa: BLE001
            return dict(raw_close=np.nan, raw_open=np.nan, factor=np.nan, up_lim=False, dn_lim=False, up_open=False,
                        dn_open=False, susp=True)


def names_table():
    from engine.common import EXTRA
    return pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")["code_name"].to_dict()


# ============================================================================ 账户
class Account:
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(ACC_DIR, name)
        self.fn = os.path.join(self.dir, "state.json")
        if os.path.exists(self.fn):
            self.s = json.load(open(self.fn, encoding="utf-8"))
        else:
            self.s = None
        self.trades = []
        self.navs = []

    @property
    def exists(self):
        return self.s is not None

    def init(self, start):
        self.s = dict(strategy=self.name, init_cash=INIT_CASH, start_date=str(start.date()), last_date=None,
                      cash=INIT_CASH, etf_value=0.0, positions={}, pending=None)

    # ----- 估值
    def mark(self, mk, d):
        """复权折算股数 + 按 d 日收盘估值"""
        val = 0.0
        for j, p in self.s["positions"].items():
            q = mk.quote(d, j)
            f0 = p.get("factor") or 0
            if q["factor"] > 0 and f0 > 0 and abs(q["factor"] / f0 - 1) > ADJ_TOL:     # 除权除息：送转/分红折算成股数
                p["shares"] = round(p["shares"] * q["factor"] / f0, 2)
                p["factor"] = q["factor"]
            elif q["factor"] > 0 and f0 <= 0:
                p["factor"] = q["factor"]
            if q["raw_close"] > 0:
                p["last_px"] = q["raw_close"]
            val += p["shares"] * p.get("last_px", 0.0)
        return val

    def total(self):
        return self.s["cash"] + self.s["etf_value"] + sum(p["shares"] * p.get("last_px", 0.0)
                                                          for p in self.s["positions"].values())

    def stock_value(self):
        return sum(p["shares"] * p.get("last_px", 0.0) for p in self.s["positions"].values())

    # ----- 成交
    def trade(self, d, action, inst, shares, px, f, note="", name=""):
        amt = shares * px
        if action == "买入":
            self.s["cash"] -= amt + f
            p = self.s["positions"].get(inst)
            if p:
                p["cost"] += amt + f
                p["shares"] += shares
            else:
                self.s["positions"][inst] = dict(shares=float(shares), cost=amt + f, buy_date=str(d.date()),
                                                 last_px=px, factor=None)
        else:
            self.s["cash"] += amt - f
            p = self.s["positions"][inst]
            if shares >= p["shares"] - 1e-6:
                del self.s["positions"][inst]
            else:
                p["cost"] *= 1 - shares / p["shares"]
                p["shares"] -= shares
        self.trades.append(dict(日期=str(d.date()), 操作=action, 代码=code_of(inst) if inst != "ETF" else ETF_CODE,
                                名称=name, 股数=int(shares) if float(shares).is_integer() else round(shares, 2), 成交价=round(px, 3), 金额=round(amt, 2),
                                费用=round(f, 2), 备注=note))

    # ----- 保存
    def save(self, names):
        os.makedirs(self.dir, exist_ok=True)
        json.dump(self.s, open(self.fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if self.trades:
            fn = os.path.join(self.dir, "trades.csv")
            df = pd.DataFrame(self.trades)
            df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False, encoding="utf-8-sig")
            self.trades = []
        if self.navs:
            fn = os.path.join(self.dir, "nav.csv")
            df = pd.DataFrame(self.navs)
            if os.path.exists(fn):
                old = pd.read_csv(fn, encoding="utf-8-sig")
                df = pd.concat([old[~old["日期"].isin(df["日期"])], df], ignore_index=True)
            df.to_csv(fn, index=False, encoding="utf-8-sig")
            self.navs = []
        rows = []
        for j, p in sorted(self.s["positions"].items()):
            mv = p["shares"] * p.get("last_px", 0.0)
            rows.append(dict(代码=code_of(j), 名称=names.get(j, ""), 股数=round(p["shares"], 2), 现价=p.get("last_px"),
                             市值=round(mv, 2), 成本=round(p["cost"], 2), 浮动盈亏=round(mv - p["cost"], 2),
                             买入日期=p.get("buy_date", "")))
        if self.s["etf_value"] > 0:
            rows.append(dict(代码=ETF_CODE, 名称="中证1000ETF（指数近似）", 股数="", 现价="", 市值=round(self.s["etf_value"], 2),
                             成本="", 浮动盈亏="", 买入日期=""))
        pd.DataFrame(rows, columns=["代码", "名称", "股数", "现价", "市值", "成本", "浮动盈亏", "买入日期"]).to_csv(
            os.path.join(self.dir, "positions.csv"), index=False, encoding="utf-8-sig")
        if self.s.get("last_date"):      # 每天的收盘持仓另存一份到当天的清单文件夹（日报用）
            dd = os.path.join(SIG_DIR, self.s["last_date"])
            os.makedirs(dd, exist_ok=True)
            pd.DataFrame(rows, columns=["代码", "名称", "股数", "现价", "市值", "成本", "浮动盈亏", "买入日期"]).to_csv(
                os.path.join(dd, f"{self.name}_positions.csv"), index=False, encoding="utf-8-sig")


# ============================================================================ 成交逻辑
def execute_conc(acc, mk, d, pend, names, cfg):
    """K_top3：d 日开盘价成交，滑点 slip"""
    oc, cc, mc, slip = cfg["open_cost"], cfg["close_cost"], cfg["min_cost"], cfg["slip"]
    pos = acc.s["positions"]
    for j in pend["sell"]:
        if j not in pos:
            continue
        q = mk.quote(d, j)
        if q["susp"] or q["dn_open"] or not q["raw_open"] > 0:
            acc.trades.append(dict(日期=str(d.date()), 操作="卖出失败", 代码=code_of(j), 名称=names.get(j, ""), 股数=round(pos[j]["shares"], 2),
                                   成交价="", 金额="", 费用="", 备注="开盘跌停/停牌，继续持有"))
            continue
        px = q["raw_open"] * (1 - slip)
        sh = pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, px, fee(sh * px, cc, mc), pend.get("why", {}).get(j, "清单卖出"), names.get(j, ""))
    slot = pend["slot"]
    backups = list(pend["backup"])
    done = set(pos)

    def try_buy(j, note):
        if j in done:
            return False
        q = mk.quote(d, j)
        if q["susp"] or q["up_open"] or not q["raw_open"] > 0:
            return False
        px, lot = q["raw_open"] * (1 + slip), lot_of(j)
        sh = math.floor(min(slot, acc.s["cash"]) / (1 + oc) / (px * lot)) * lot
        while sh > 0 and sh * px + fee(sh * px, oc, mc) > acc.s["cash"]:
            sh -= lot
        if sh <= 0:
            return False
        acc.trade(d, "买入", j, sh, px, fee(sh * px, oc, mc), note, names.get(j, ""))
        pos[j]["factor"] = q["factor"]
        done.add(j)
        return True

    for b in pend["buy"]:
        if try_buy(b["inst"], "清单买入（开盘）"):
            continue
        while backups:
            r = backups.pop(0)
            if try_buy(r["inst"], f"替补（替 {code_of(b['inst'])}）"):
                break


def holdings_file(acc, with_date=False):
    fn = os.path.join(PAPER, ".tmp", f"{acc.name}_hold.csv")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    rows = [dict(code=code_of(j), shares=p["shares"], buy_date=p.get("buy_date", "")) for j, p in acc.s["positions"].items()]
    df = pd.DataFrame(rows, columns=["code", "shares", "buy_date"])
    if not with_date:
        df = df[["code", "shares"]]
    df.to_csv(fn, index=False)
    return fn if len(rows) else None


def gen_signal(acc, d):
    """返回 (pending dict, 清单 DataFrame, 屏幕输出)"""
    from senti import live as SL
    ns = argparse.Namespace(holdings=holdings_file(acc), cash=float(acc.s["cash"]), date=str(d.date()))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn = SL.cmd_signal(ns)
    text = buf.getvalue()
    out = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
    pend = dict(signal_date=str(d.date()), kind="conc")
    pend["sell"] = [inst_of(c) for c in out.loc[out["操作"] == "卖出", "代码"]]
    pend["why"] = {inst_of(r.代码): str(r.备注) for r in out[out["操作"] == "卖出"].itertuples()}
    pend["buy"] = [dict(inst=inst_of(c)) for c in out.loc[out["操作"] == "买入", "代码"]]
    pend["backup"] = [dict(inst=inst_of(c)) for c in out.loc[out["操作"] == "替补", "代码"]]
    pend["slot"] = acc.total() / SL.CFG["k"]
    dd = os.path.join(SIG_DIR, str(d.date()))
    os.makedirs(dd, exist_ok=True)
    out.to_csv(os.path.join(dd, f"{acc.name}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(dd, f"{acc.name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return pend, out, text


# ============================================================================ 主流程
def strat_cfg(name):
    from senti.live import CFG
    return CFG


def run_one(name, start=None, until=None):
    mk = Market()
    names = names_table()
    days = trading_days()
    data_last = mk.bench.index.max()
    days = [d for d in days if d <= data_last]
    if until:
        days = [d for d in days if d <= pd.Timestamp(until)]
    acc = Account(name)
    if not acc.exists:
        s = pd.Timestamp(start) if start else days[-1]
        s = max(d for d in days if d <= s)
        acc.init(s)
        acc.navs.append(dict(日期=str(s.date()), 现金=round(acc.s["cash"], 2), 股票市值=0.0,
                             总资产=round(INIT_CASH, 2), 当日收益=0.0, 累计收益=0.0, 中证1000当日=0.0,
                             中证1000累计=0.0, 超额累计=0.0, 持仓数=0, 成交笔数=0))
        acc.s["bench_cum"] = 1.0
        print(f"[{name}] 新建账户：初始资金 {INIT_CASH:,.0f}，起始日 {s.date()}", flush=True)
        pend, out, text = gen_signal(acc, s)
        acc.s["pending"] = pend
        acc.s["last_date"] = str(s.date())
        acc.save(names)
        print(text, flush=True)
        todo = [d for d in days if d > s]
    else:
        todo = [d for d in days if d > pd.Timestamp(acc.s["last_date"])]
    if not todo:
        print(f"[{name}] 已是最新（{acc.s['last_date']}），数据最新交易日 {data_last.date()}", flush=True)
        return
    cfg = strat_cfg(name)
    for d in todo:
        prev_total = acc.total()
        n0 = len(acc.trades)
        pend = acc.s.get("pending")
        # 开盘成交前先做除权折算
        acc.mark(mk, d)
        if pend:
            execute_conc(acc, mk, d, pend, names, cfg)
        acc.mark(mk, d)
        tot = acc.total()
        b = float(mk.bench.get(d, 0.0))
        acc.s["bench_cum"] = acc.s.get("bench_cum", 1.0) * (1 + b)
        cum = tot / acc.s["init_cash"] - 1
        acc.navs.append(dict(日期=str(d.date()), 现金=round(acc.s["cash"], 2), 股票市值=round(acc.stock_value(), 2),
                             总资产=round(tot, 2), 当日收益=round(tot / prev_total - 1, 6),
                             累计收益=round(cum, 6), 中证1000当日=round(b, 6), 中证1000累计=round(acc.s["bench_cum"] - 1, 6),
                             超额累计=round(cum - (acc.s["bench_cum"] - 1), 6), 持仓数=len(acc.s["positions"]),
                             成交笔数=len([t for t in acc.trades[n0:] if t["操作"] in ("买入", "卖出")])))
        print(f"[{name}] {d.date()} 成交 {len(acc.trades) - n0} 笔，总资产 {tot:,.0f}（当日 {tot / prev_total - 1:+.2%}，"
              f"累计 {cum:+.2%}），持仓 {len(acc.s['positions'])} 只", flush=True)
        pend, out, text = gen_signal(acc, d)
        acc.s["pending"] = pend
        acc.s["last_date"] = str(d.date())
        acc.save(names)
    print(text, flush=True)


def summary():
    name = "S_top3"
    acc = Account(name)
    if not acc.exists:
        return
    nav = pd.read_csv(os.path.join(acc.dir, "nav.csv"), encoding="utf-8-sig")
    last = nav.iloc[-1]
    v = nav["总资产"]
    mdd = float((v / v.cummax() - 1).min())
    row = (f"| {acc.s['start_date']} | {last['日期']} | {last['总资产']:,.0f} | {last['当日收益']:+.2%} | {last['累计收益']:+.2%} | "
           f"{last['中证1000累计']:+.2%} | {last['超额累计']:+.2%} | {mdd:.2%} | {int(last['持仓数'])} | {acc.s['cash']:,.0f} |")
    pos_md = ["无持仓"]
    pf = os.path.join(acc.dir, "positions.csv")
    if os.path.exists(pf):
        try:
            ps = pd.read_csv(pf, dtype={"代码": str}, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            ps = pd.DataFrame()
        if len(ps):
            ps = ps.fillna("")
            pos_md = ["| " + " | ".join(ps.columns) + " |", "|" + "---|" * len(ps.columns)] + \
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in ps.itertuples(index=False)]
    p = acc.s.get("pending") or {}
    sd = p.get("signal_date", "")
    fn = os.path.join(SIG_DIR, sd, f"{name}.csv")
    ord_md = ["无操作"]
    if os.path.exists(fn):
        o = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
        if (o["操作"] == "买入").any():
            o = pd.concat([o[o["操作"] != "替补"], o[o["操作"] == "替补"].head(5)])
        else:
            o = o[o["操作"] != "替补"]
        if len(o):
            cols = ["操作", "代码", "名称", "排名", "大涨概率", "大跌概率", "五日模型名次", "股数", "参考价", "参考金额", "备注"]
            o = o[cols].fillna("")
            ord_md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in o.itertuples(index=False)]
    md = ["# 进攻版 S_top3 模拟盘（10 万元）", "",
          "打分 = 排名平均（大涨概率 − 大跌概率，K_top3 的六个 5 日模型）；中证1000 内持 3 只，排名跌出前 60 才卖，不择时，"
          "T 日收盘后出清单、T+1 开盘成交。GitHub Actions 每个交易日北京时间 17:00 自动运行，数据晚发布时之后的运行会自动补齐。", "",
          "## 账户", "",
          "| 起始日 | 最新日 | 总资产 | 当日 | 累计 | 中证1000累计 | 超额 | 最大回撤 | 持仓 | 现金 |",
          "|---|---|---|---|---|---|---|---|---|---|", row, "",
          "## 当前持仓", "", *pos_md, "",
          f"## 下一交易日操作清单（信号日 {sd}，下一交易日开盘集合竞价成交）", "", *ord_md, "",
          "明细：`paper/accounts/S_top3/`（nav.csv 每日净值、trades.csv 成交、positions.csv 持仓）；每日完整清单和程序输出在 `paper/signals/<日期>/`。", "",
          "> 仅供学习研究，不构成投资建议。持 3 只的集中策略波动很大（研究中最大回撤 −37% 到 −45%）；模拟盘按开盘价成交，未计冲击成本。"]
    open(os.path.join(PAPER, "README.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    sys.path.insert(0, PAPER)
    import report                    # 每日日报 + 历史总览（paper/report.py）
    report.build(STRATS, "进攻版模拟盘")
    print(row)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", help="新建账户的起始信号日（默认最新交易日）")
    r.add_argument("--until", help="只处理到这一天（测试用）")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "run":
        run_one("S_top3", a.start, a.until)
    summary()


if __name__ == "__main__":
    main()
