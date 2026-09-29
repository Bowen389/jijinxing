"""
机会仓 FB_dip 用到的市场条件：全市场跌停家数占比 M_DN → {DATA}/senti_market.parquet
  python paper/market.py

从原仓库 experiments/senti.py build 中抽出，只保留 FB_dip 需要的 M_DN，口径完全相同：
  股票范围 = 面板里的全部股票（曾属中证1000 的约 2800 只中小盘股）
  剔除上市不满 60 个交易日的新股（按面板里的有效交易日计数）和停牌股
  M_DN = 收盘跌停家数 / 有效股票数
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.common import DATA  # noqa: E402


def build():
    pdir = os.path.join(DATA, "panel")
    aggs = []
    for fn in sorted(x for x in os.listdir(pdir) if x.endswith(".parquet")):
        p = pd.read_parquet(os.path.join(pdir, fn), columns=["datetime", "instrument", "dn_lim", "susp"])
        p = p.sort_values(["instrument", "datetime"]).reset_index(drop=True)
        act = ~p["susp"]
        age = act.astype(int).groupby(p["instrument"], sort=False).cumsum()
        act = act & (age >= 60)
        m = pd.DataFrame({"datetime": p["datetime"], "n": act.astype(int), "dn": (p["dn_lim"] & act).astype(int)})
        aggs.append(m.groupby("datetime")[["n", "dn"]].sum())
    A = aggs[0]
    for a in aggs[1:]:
        A = A.add(a, fill_value=0)
    M = pd.DataFrame({"M_DN": (A["dn"] / A["n"]).astype("float32")}).sort_index()
    M.index.name = "datetime"
    M.reset_index().to_parquet(os.path.join(DATA, "senti_market.parquet"), index=False)
    print("senti_market", len(M), "天，最新", M.index.max().date(), f"M_DN={float(M['M_DN'].iloc[-1]):.4f}", flush=True)


if __name__ == "__main__":
    build()
