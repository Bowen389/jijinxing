"""
进攻版（方案 A，strategies2/README.md 第 7 节的可用配置）：每日调仓清单
  打分  = 排名平均（“不对称”打分 = 大涨概率 − 大跌概率， K_top3 的六个 5 日模型原始打分）
  持仓  = 中证1000 内最多 3 只，等权（每只约 1/3 资金）
  卖出  = 持仓排名跌出前 60，或已不在中证1000
  买入  = 空位按排名补足；开盘一字涨停/停牌买不进时按顺序用替补
  不择时；T 日收盘后出信号，T+1 开盘集合竞价成交

  python senti/live.py signal --holdings my.csv --cash 5000 [--date 2026-09-29]
holdings.csv 两列：code,shares（空仓时可不传 --holdings，此时 --cash 默认 10 万）。清单保存在 senti/signals/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.common import DATA, EXTRA, KEY, read_parts  # noqa: E402
from engine.execution import affordable  # noqa: E402

CFG = dict(k=3, keep_rank=60, open_cost=0.0005, close_cost=0.0010, min_cost=5.0, slip=0.001, account=100_000,
           h5_models=["k_rb_s0", "k_rb_s1", "k_rb_s2", "k_rbh_s0", "k_rbh_s1", "k_rbh_s2"])


def scores(start, end):
    """→ (综合打分 DataFrame[datetime, instrument, score]，不对称打分明细，5 日模型打分)。只保留当日中证1000 成分"""
    from senti import model as SM
    from strategies2 import live as L2
    h5, _ = L2.predict(start, end, CFG["h5_models"], neutral=False)
    sk = SM.predict(start, end)
    blend = L2.rank_blend([sk[KEY + ["score"]], h5[KEY + ["score"]]])
    pool = read_parts(os.path.join(DATA, "panel"), columns=["datetime", "instrument", "in_pool"], start=start, end=end)
    pool = pool[pool["in_pool"]][KEY]
    blend = blend.merge(pool, on=KEY)
    return blend, sk, h5


def cmd_signal(a):
    from strategies.live import last_trading_day, norm_code
    T = pd.Timestamp(a.date) if a.date else last_trading_day()
    k, keep = CFG["k"], CFG["keep_rank"]
    blend, sk, h5 = scores(T - pd.Timedelta(days=10), T)
    if not len(blend) or blend["datetime"].max() < T:
        sys.exit(f"{T.date()} 没有因子数据，请先更新数据（bash paper/run_daily.sh 的 1–5 步）")
    px = read_parts(os.path.join(DATA, "panel"), columns=["datetime", "instrument", "raw_close", "up_lim", "dn_lim", "susp",
                                                          "in_pool", "buy_ok", "status_known", "is_st", "is_delisted"], start=T, end=T).set_index("instrument")
    names = pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")
    if a.holdings:
        h = pd.read_csv(a.holdings, dtype={"code": str})
        h["instrument"] = h["code"].map(norm_code)
        hold = h.groupby("instrument")["shares"].sum()
    else:
        hold = pd.Series(dtype=float)
    hold = hold[hold > 0]
    miss = [j for j in hold.index if j not in px.index]
    if miss:
        from engine.market import Market
        mk = Market()
        for j in miss:
            q = mk.quote(T, j)
            for c in ("raw_close", "up_lim", "dn_lim", "susp", "buy_ok"):
                px.loc[j, c] = q[c]
            px.loc[j, "in_pool"] = False
    cash = a.cash if a.cash is not None else (CFG["account"] if not len(hold) else 0.0)
    val = pd.Series({j: hold[j] * float(np.nan_to_num(px["raw_close"].get(j, np.nan))) for j in hold.index}, dtype=float)
    total = float(val.sum()) + cash

    day = blend[blend["datetime"] == T].set_index("instrument")["score"].sort_values(ascending=False, kind="stable")
    # Known prohibited names leave the ranking; missing status is excluded
    # from new candidates without forcing liquidation of all old holdings.
    prohibited = px["is_st"].fillna(False) | px["is_delisted"].fillna(False) | px.index.to_series().str.startswith("BJ")
    day = day[~day.index.isin(px.index[prohibited])].dropna()
    rank = pd.Series(np.arange(1, len(day) + 1), index=day.index)
    skd = sk[sk["datetime"] == T].set_index("instrument")
    h5r = h5[h5["datetime"] == T].set_index("instrument")["score"].rank(ascending=False)
    sell, keep_list = [], []
    for j in hold.index:
        r = rank.get(j, 10 ** 6)
        why = "ST/退市/北交所：退出" if bool(prohibited.get(j, False)) else "不在中证1000或缺少有效评分" if j not in rank.index else f"排名 {r} 跌出前 {keep}" if r > keep else ""
        (sell if why else keep_list).append((j, why))
    n_free = max(k - len(keep_list), 0)
    cands = [j for j in day.index if j not in hold.index and bool(px["buy_ok"].get(j, False)) and not bool(px["susp"].get(j, True))]
    buy, backup = cands[:n_free], cands[n_free:n_free + 5]
    slot = total / k

    def row(action, j, shares=None, note=""):
        p = float(px["raw_close"].get(j, np.nan))
        if shares is None:
            shares = affordable(j, slot, p * (1 + CFG["slip"]), CFG["open_cost"], CFG["min_cost"])
            if shares == 0:
                note = "资金不足一手，用替补"
        return dict(操作=action, 代码=j[2:] + "." + j[:2], 名称=names["code_name"].get(j, ""), 行业=names["industry"].get(j, ""),
                    排名=int(rank.get(j, -1)), 大涨概率=round(float(skd["up"].get(j, np.nan)), 4),
                    大跌概率=round(float(skd["dn"].get(j, np.nan)), 4), 五日模型名次=int(h5r.get(j, -1)),
                    股数=int(shares), 参考价=round(p, 2), 参考金额=round(shares * p if p > 0 else 0, 0), 备注=note)

    orders = [row("卖出", j, hold[j], why) for j, why in sell] + [row("继续持有", j, hold[j]) for j, _ in keep_list] + \
             [row("买入", j) for j in buy] + [row("替补", j) for j in backup]
    out = pd.DataFrame(orders, columns=["操作", "代码", "名称", "行业", "排名", "大涨概率", "大跌概率", "五日模型名次", "股数",
                                        "参考价", "参考金额", "备注"])
    os.makedirs("senti/signals", exist_ok=True)
    fn = f"senti/signals/S_top3_{T.date()}.csv"
    out.to_csv(fn, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n==== 进攻版 S_top3｜信号日 {T.date()}｜在下一交易日【开盘集合竞价】成交 ====")
    print("打分：排名平均（大涨概率 − 大跌概率，六个 5 日模型）；持 3 只，跌出前 60 才卖，不择时")
    print(f"证券状态未知 {int((~px['status_known'].fillna(False)).sum())} 只（禁止新买入）；合格新买候选 {len(cands)} 只")
    print(f"候选池：当日中证1000 成分 {len(day)} 只；全市场平均 大涨概率 {skd['up'].mean():.3f} / 大跌概率 {skd['dn'].mean():.3f}")
    print(f"账户：股票 {val.sum():,.0f} + 现金 {cash:,.0f} = {total:,.0f}；持仓 {len(hold)} 只 / 最多 {k} 只；每只目标约 {slot:,.0f}\n")
    main = out[out["操作"] != "替补"]
    print(main.to_string(index=False) if len(main) else "今日无需交易")
    if len(buy):
        print("\n替补（买入目标开盘一字涨停/停牌买不进时按顺序替换）：",
              "、".join(f"{r.代码} {r.名称}" for r in out[out["操作"] == "替补"].itertuples()))
        print("清单股数按收盘价估算；执行时按可用资金和实际空位重算。开盘价仅作成交近似；开盘已知后替补不能保证仍以竞价价成交。")
    print(f"\n已保存 {fn}")
    return fn


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signal")
    s.add_argument("--holdings")
    s.add_argument("--cash", type=float)
    s.add_argument("--date")
    a = ap.parse_args()
    cmd_signal(a)


if __name__ == "__main__":
    main()
