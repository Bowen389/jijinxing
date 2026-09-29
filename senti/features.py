"""
进攻版的特征：全市场情绪周期指标 + 个股短线特征（+ 训练用标签 y3）
  python senti/features.py      → {DATA}/senti_market.parquet、{DATA}/senti_stock.parquet

与研究代码 experiments/senti.py build 逐行相同（只删掉了进攻版用不到的隔夜标签 yon / yid），保证实盘特征和训练时一致。

情绪指标（T 日收盘可知，来自面板：曾属中证1000 的约 2800 只中小盘股，剔除上市不满 60 个交易日的新股）
  涨停/跌停家数占比、炸板率、最高连板数、2 连板以上家数、昨日涨停股今日平均收益与隔夜收益（接力赚钱效应）、
  上涨家数占比、中位数涨幅、大跌（<-7%）家数、成交额相对 20 日均值；以及各自的 5 日均值和 120 日 z 分数
个股短线特征：连板数、今日涨停/触板/炸板、10 日涨停次数、1/3/5 日涨幅、距 20 日最高、振幅、收盘位置、隔夜跳空、量比
标签 y3：T+1 开盘买入 → T+3 收盘的收益；T+1 开盘一字涨停或停牌（买不进）的样本剔除
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.common import DATA, KEY  # noqa: E402,F401

PCOLS = ["datetime", "instrument", "open", "high", "low", "close", "raw_close", "change", "factor", "amount", "lim",
         "up_lim", "dn_lim", "susp", "in_pool", "ret", "ret_on", "nbo"]
STOCK_F = ["LB", "UP0", "TOUCH0", "ZB0", "NUP10", "R1", "R3", "R5", "DH20", "AMP", "CPOS", "GAP", "VR5", "VR20"]
MKT_F = ["M_UP", "M_DN", "M_ZB", "M_LBMAX", "M_LB2", "M_RELAY", "M_RELAY_ON", "M_BREADTH", "M_MED", "M_BIGDN", "M_AMT"]
# ------------------------------------------------------------------ build
def build():
    pdir = os.path.join(DATA, "panel")
    stock_parts, mk = [], []
    for fn in sorted(x for x in os.listdir(pdir) if x.endswith(".parquet")):
        p = pd.read_parquet(os.path.join(pdir, fn), columns=PCOLS)
        p = p.sort_values(["instrument", "datetime"]).reset_index(drop=True)
        g = p.groupby("instrument", sort=False)
        act = ~p["susp"]
        prev_raw = p["raw_close"] / (1 + p["change"])
        up_p = np.round(prev_raw * (1 + p["lim"]) + 1e-6, 2)
        touch = (p["high"] / p["factor"] >= up_p - 0.0051) & act
        up = p["up_lim"] & act
        zb = touch & ~up
        # 连板数
        blk = (~up).groupby(p["instrument"], sort=False).cumsum()
        lb = up.groupby([p["instrument"], blk], sort=False).cumsum().astype("float32")
        f = pd.DataFrame({"datetime": p["datetime"], "instrument": p["instrument"]})
        f["LB"] = lb
        f["UP0"] = up.astype("float32")
        f["TOUCH0"] = touch.astype("float32")
        f["ZB0"] = zb.astype("float32")
        f["NUP10"] = up.astype(float).groupby(p["instrument"], sort=False).transform(lambda x: x.rolling(10, 1).sum()).astype("float32")
        c = p["close"]
        f["R1"] = (c / g["close"].shift(1) - 1).astype("float32")
        f["R3"] = (c / g["close"].shift(3) - 1).astype("float32")
        f["R5"] = (c / g["close"].shift(5) - 1).astype("float32")
        f["DH20"] = (c / g["high"].transform(lambda x: x.rolling(20, 1).max()) - 1).astype("float32")
        pc = g["close"].shift(1)
        f["AMP"] = ((p["high"] - p["low"]) / pc).astype("float32")
        f["CPOS"] = ((c - p["low"]) / (p["high"] - p["low"]).replace(0, np.nan)).astype("float32")
        f["GAP"] = p["ret_on"].astype("float32")
        am = p["amount"].where(act)
        f["VR5"] = (am / g["amount"].transform(lambda x: x.where(x > 0).rolling(5, 1).mean().shift(1))).astype("float32")
        f["VR20"] = (am / g["amount"].transform(lambda x: x.where(x > 0).rolling(20, 1).mean().shift(1))).astype("float32")
        # 标签：T+1 开盘买入，T+3 收盘
        o1 = g["open"].shift(-1)
        c3 = g["close"].shift(-3)
        y3 = (c3 / o1 - 1).astype("float32")
        y3[p["nbo"]] = np.nan
        f["y3"] = y3
        f["in_pool"] = p["in_pool"]
        stock_parts.append(f[p["in_pool"]].reset_index(drop=True))
        # 市场情绪：按日汇总（加和，后面合并各分块）
        prev_up = up.groupby(p["instrument"], sort=False).shift(1).fillna(False).astype(bool)
        # 情绪统计剔除上市不满 60 个交易日的新股（新股连续一字板会严重扭曲涨停/连板/接力指标）
        age = act.astype(int).groupby(p["instrument"], sort=False).cumsum()
        act, up, touch, zb = act & (age >= 60), up & (age >= 60), touch & (age >= 60), zb & (age >= 60)
        prev_up = prev_up & (age >= 61)
        lb = lb.where(age >= 60, 0)
        m = pd.DataFrame({"datetime": p["datetime"], "n": act.astype(int), "up": up.astype(int),
                          "dn": (p["dn_lim"] & act).astype(int), "touch": touch.astype(int), "zb": zb.astype(int),
                          "lb2": (lb >= 2).astype(int), "lbmax": lb,
                          "relay_s": np.where(prev_up & act, p["ret"], 0.0), "relay_n": (prev_up & act).astype(int),
                          "relayo_s": np.where(prev_up & act, p["ret_on"], 0.0),
                          "pos": ((p["ret"] > 0) & act).astype(int), "bigdn": ((p["ret"] < -0.07) & act).astype(int),
                          "amt": p["amount"].where(act, 0.0)})
        agg = m.groupby("datetime").agg(n=("n", "sum"), up=("up", "sum"), dn=("dn", "sum"), touch=("touch", "sum"),
                                        zb=("zb", "sum"), lb2=("lb2", "sum"), lbmax=("lbmax", "max"),
                                        relay_s=("relay_s", "sum"), relay_n=("relay_n", "sum"),
                                        relayo_s=("relayo_s", "sum"), pos=("pos", "sum"), bigdn=("bigdn", "sum"),
                                        amt=("amt", "sum"))
        med = p.loc[act, ["datetime", "ret"]].groupby("datetime")["ret"].apply(lambda x: x.values.astype("float32"))
        mk.append((agg, med))
        print(fn, len(f), flush=True)
        del p, f, m
    A = mk[0][0]
    for a, _ in mk[1:]:
        A = A.add(a, fill_value=0)   # lbmax 的加和无意义，下面单独取最大值
    A["lbmax"] = pd.concat([a["lbmax"] for a, _ in mk], axis=1).max(axis=1)
    meds = {}
    for _, md in mk:
        for d, v in md.items():
            meds.setdefault(d, []).append(v)
    M = pd.DataFrame(index=A.index)
    M["M_UP"] = A["up"] / A["n"]
    M["M_DN"] = A["dn"] / A["n"]
    M["M_ZB"] = A["zb"] / A["touch"].replace(0, np.nan)
    M["M_LBMAX"] = A["lbmax"]
    M["M_LB2"] = A["lb2"] / A["n"]
    M["M_RELAY"] = A["relay_s"] / A["relay_n"].replace(0, np.nan)
    M["M_RELAY_ON"] = A["relayo_s"] / A["relay_n"].replace(0, np.nan)
    M["M_BREADTH"] = A["pos"] / A["n"]
    M["M_MED"] = pd.Series({d: float(np.nanmedian(np.concatenate(v))) for d, v in meds.items()})
    M["M_BIGDN"] = A["bigdn"] / A["n"]
    M["M_AMT"] = A["amt"] / A["amt"].rolling(20, 5).mean()
    M = M.sort_index()
    # 平滑与相对水平（均只用 T 日及以前）
    for ccol in MKT_F:
        M[ccol + "_5"] = M[ccol].rolling(5, 1).mean()
        M[ccol + "_z"] = (M[ccol] - M[ccol].rolling(120, 20).mean()) / (M[ccol].rolling(120, 20).std() + 1e-9)
    M.index.name = "datetime"
    M = M.reset_index().astype({c: "float32" for c in M.columns if c.startswith("M_")})
    M.to_parquet(os.path.join(DATA, "senti_market.parquet"), index=False)
    S = pd.concat(stock_parts, ignore_index=True)
    S.to_parquet(os.path.join(DATA, "senti_stock.parquet"), index=False)
    print("market", M.shape, "stock", S.shape, "y3>=10% 占比", round((S["y3"] >= 0.10).mean(), 4), flush=True)


if __name__ == "__main__":
    build()
