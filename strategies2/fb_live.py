"""
机会仓 FB_dip（低位首板 + 次日低开）：每个交易日收盘后生成次日操作清单
  bash setup_env.sh --refresh && source env.sh && python experiments/senti.py build     # 更新行情 + 市场情绪（约 1 分钟）
  python strategies2/fb_live.py signal --holdings my_fb.csv --cash 50000

holdings.csv 三列：code,shares,buy_date（例如 603533,1000,2026-09-30）；空仓时不传 --holdings
规则（研究见 experiments/fb_dip.py 与 strategies2/README.md 第 10、11 节）
  候选：今日首板（今日涨停、昨日未涨停）、今日不是开盘即涨停、10cm、在中证1000 池内、
        首板前收盘 < 过去 60 日收盘中位数
  市场：今日中证1000 收盘在 20 日均线下，且今日全市场跌停家数占比 < 3%；不满足则明天不买
  买入：明天 9:25 集合竞价出来后，开盘价相对今日收盘低开 2%~7% 才买；低开更深的优先；
        机会仓最多同时 2 只，每只 = 机会仓总资金 × 50%
  卖出：买入后的下一个交易日收盘卖出（尾盘集合竞价 14:57 前挂单）；若当天收盘跌停卖不出，顺延到下一个交易日收盘
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
from strategies import live as A  # noqa: E402

norm_code, lot_of, last_trading_day = A.norm_code, A.lot_of, A.last_trading_day
K = 2
GAP_LO, GAP_HI = -0.07, -0.02
COST = 0.00025 + 0.001        # 买入佣金 + 滑点估计（用于算股数）


def market_state(T):
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    b = b[b.index <= T]
    lvl = (1 + b).cumprod()
    ma20 = float(lvl.iloc[-1] / lvl.rolling(20).mean().iloc[-1] - 1)
    M = pd.read_parquet(os.path.join(DATA, "senti_market.parquet")).set_index("datetime")
    if T not in M.index:
        sys.exit(f"senti_market 没有 {T.date()} 的数据，请先运行 python experiments/senti.py build")
    return ma20, float(M.loc[T, "M_DN"])


def candidates(T):
    start = T - pd.Timedelta(days=150)
    p = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["close", "raw_close", "lim", "up_lim", "up_open", "susp", "in_pool"],
                   start=start, end=T).sort_values(KEY)
    g = p.groupby("instrument", sort=False)
    p["pos60"] = g["close"].shift(1) / g["close"].transform(lambda x: x.shift(1).rolling(60, 20).median())
    p["prev_up"] = g["up_lim"].shift(1).fillna(False).astype(bool)
    t = p[p["datetime"] == T]
    c = t[t["up_lim"] & ~t["prev_up"] & ~t["up_open"] & t["in_pool"] & (t["lim"] < 0.15) & (t["pos60"] < 1)]
    return c.set_index("instrument"), t.set_index("instrument")


def cmd_signal(a):
    T = pd.Timestamp(a.date) if a.date else last_trading_day()
    ma20, mdn = market_state(T)
    cand, today = candidates(T)
    names = pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")
    if a.holdings:
        h = pd.read_csv(a.holdings, dtype={"code": str})
        h["instrument"] = h["code"].map(norm_code)
        h = h[h["shares"] > 0]
    else:
        h = pd.DataFrame(columns=["instrument", "shares", "buy_date"])
    val = 0.0
    rows = []
    for r in h.itertuples():
        px = float(today["raw_close"].get(r.instrument, np.nan))
        val += r.shares * (px if px > 0 else 0)
        note = "买入次日收盘卖出" if str(r.buy_date) >= str(T.date()) else "上次卖出日跌停/停牌未卖出，顺延"
        rows.append(dict(操作="明天收盘卖出", 代码=r.instrument[2:] + "." + r.instrument[:2], 名称=names["code_name"].get(r.instrument, ""),
                         股数=int(r.shares), 参考价=round(px, 2), 买入区间="", 备注=note))
    cash = float(a.cash or 0.0)
    total = cash + val
    slot = total / K
    free = K - len(h)   # 持仓明天收盘才卖，明天开盘时仍占位
    ok = ma20 < 0 and mdn < 0.03
    buys = []
    if ok and free > 0:
        for j, r in cand.iterrows():
            if j in set(h["instrument"]):
                continue
            p0 = float(r["raw_close"])
            lo, hi = round(p0 * (1 + GAP_LO) + 1e-9, 2), round(p0 * (1 + GAP_HI) + 1e-9, 2)
            lot = lot_of(j)
            sh_hi = int(np.floor(slot / (1 + COST) / (lo * lot)) * lot)     # 按区间下沿估算的最多股数
            sh_lo = int(np.floor(slot / (1 + COST) / (hi * lot)) * lot)
            buys.append(dict(操作="候选（明早看开盘价）", 代码=j[2:] + "." + j[:2], 名称=names["code_name"].get(j, ""),
                             股数=f"{sh_lo}~{sh_hi}" if sh_hi > 0 else "资金不足一手", 参考价=round(p0, 2),
                             买入区间=f"{lo:.2f} ~ {hi:.2f}", 备注=f"位置 {r['pos60']:.3f}"))
    out = pd.DataFrame(rows + buys)
    os.makedirs("strategies2/signals", exist_ok=True)
    fn = f"strategies2/signals/FB_dip_{T.date()}.csv"
    out.to_csv(fn, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n==== 机会仓 FB_dip｜信号日 {T.date()}｜明天：开盘看价买入，收盘卖出昨天及以前买入的 ====")
    print(f"市场：中证1000 相对 20 日线 {ma20:+.2%}（需 < 0）{'✓' if ma20 < 0 else '✗'}；"
          f"全市场跌停占比 {mdn:.2%}（需 < 3%）{'✓' if mdn < 0.03 else '✗'} → " + ("【可以买】" if ok else "【明天不买】"))
    print(f"机会仓：股票 {val:,.0f} + 现金 {cash:,.0f} = {total:,.0f}；每只目标 {slot:,.0f}；明天开盘时空余名额 {max(free, 0)} / {K}")
    print(f"今日符合个股条件的首板：{len(cand)} 只\n")
    print(out.to_string(index=False) if len(out) else "明天无操作")
    if buys:
        print(f"\n明早操作：9:25 集合竞价结束后，看上面候选的开盘价；落在【买入区间】内的，按低开幅度从深到浅，最多买 {max(free, 0)} 只；"
              f"\n  股数 = 每只目标金额 ÷ 开盘价（向下取整到一手）；9:25-9:30 挂限价单（例如开盘价 +0.5%），9:30 成交。"
              f"\n  开盘价不在区间内（低开不足 2% 或超过 7%、高开）的不买。买入后在持仓文件里记下 buy_date。")
    print(f"\n已保存 {fn}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signal", help="生成次日操作清单")
    s.add_argument("--holdings", help="机会仓持仓 csv：code,shares,buy_date")
    s.add_argument("--cash", type=float, default=0.0, help="机会仓可用现金")
    s.add_argument("--date", help="信号日（默认最新交易日）")
    a = ap.parse_args()
    {"signal": cmd_signal}[a.cmd](a)


if __name__ == "__main__":
    main()
