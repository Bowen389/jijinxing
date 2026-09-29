"""
构建基础行情面板（回测、标签、衍生因子都用它）
  python -m engine.panel
输出 {DATA}/panel/part-*.parquet：
  datetime, instrument, open/high/low/close(复权), raw_close, change, volume, amount, factor,
  in_pool(当日是否在股票池), ret(当日收益), lim(涨跌停幅度), up_lim/dn_lim(收盘是否封板),
  susp(停牌), ret1/ret5(标签: T+1收盘买入持有1/5日)
另存 {DATA}/bench.parquet 基准指数日收益
"""
import os
import shutil

import numpy as np
import pandas as pd

from engine.common import DATA, START, END, POOL, BENCH, init_qlib, pool_spans, limit_pct

FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$change", "$factor"]
NAMES = ["open", "high", "low", "close", "volume", "amount", "change", "factor"]


def build(chunk=250):
    from qlib.data import D
    init_qlib()
    spans = pool_spans(POOL)
    codes = sorted(spans["instrument"].unique())
    out = os.path.join(DATA, "panel")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for k in range(0, len(codes), chunk):
        cc = codes[k:k + chunk]
        df = D.features(cc, FIELDS, START, END).astype("float32")
        df.columns = NAMES
        df = df.reset_index()
        df["instrument"] = df["instrument"].astype(str)
        df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)
        # 股票池成员（逐只股票按区间打标，省内存）
        sp = spans[spans["instrument"].isin(cc)]
        dt = df["datetime"].values
        inp = np.zeros(len(df), dtype=bool)
        bounds = df.groupby("instrument", sort=False).indices
        for inst, rows in sp.groupby("instrument"):
            idx = bounds.get(inst)
            if idx is None:
                continue
            d = dt[idx]
            ok = np.zeros(len(idx), dtype=bool)
            for s0, e0 in zip(rows["start"].values, rows["end"].values):
                ok |= (d >= s0) & (d <= e0)
            inp[idx] = ok
        df["in_pool"] = inp
        g = df.groupby("instrument", sort=False)
        df["susp"] = ~(df["volume"] > 0)
        df["ret"] = (df["close"] / g["close"].shift(1) - 1).astype("float32")
        df.loc[df["susp"], "ret"] = 0.0
        df["raw_close"] = (df["close"] / df["factor"]).astype("float32")
        df["lim"] = limit_pct(df["instrument"], df["datetime"])
        prev_raw = df["raw_close"] / (1 + df["change"])
        up_p = np.round(prev_raw * (1 + df["lim"]) + 1e-6, 2)
        dn_p = np.round(prev_raw * (1 - df["lim"]) + 1e-6, 2)
        df["up_lim"] = ((df["raw_close"] >= up_p - 0.0051) | (df["change"] >= df["lim"] - 0.0015)) & ~df["susp"]
        df["dn_lim"] = ((df["raw_close"] <= dn_p + 0.0051) | (df["change"] <= -df["lim"] + 0.0015)) & ~df["susp"]
        # 次日（成交日）买不进：涨停或停牌 —— 这些样本的标签不可实现，训练时剔除
        blk = (df["up_lim"] | df["susp"]).astype(float)
        df["nb"] = g_blk = blk.groupby(df["instrument"], sort=False).shift(-1).fillna(1.0).astype(bool)
        # 开盘成交口径：开盘一字涨/跌停不能买/卖；标签 = T+1 开盘买入、T+2 开盘卖出
        raw_open = df["open"] / df["factor"]
        df["up_open"] = (raw_open >= up_p - 0.0051) & ~df["susp"]
        df["dn_open"] = (raw_open <= dn_p + 0.0051) & ~df["susp"]
        df["ret_on"] = (df["open"] / g["close"].shift(1) - 1).astype("float32")     # 隔夜
        df["ret_id"] = (df["close"] / df["open"] - 1).astype("float32")             # 日内
        df.loc[df["susp"], ["ret_on", "ret_id"]] = 0.0
        df["raw_open"] = raw_open.astype("float32")
        blk_o = (df["up_open"] | df["susp"]).astype(float)
        df["nbo"] = blk_o.groupby(df["instrument"], sort=False).shift(-1).fillna(1.0).astype(bool)
        o1 = g["open"].shift(-1)
        df["reto1"] = (g["open"].shift(-2) / o1 - 1).astype("float32")
        c1 = g["close"].shift(-1)
        df["ret1"] = (g["close"].shift(-2) / c1 - 1).astype("float32")
        df["ret5"] = (g["close"].shift(-6) / c1 - 1).astype("float32")
        df.to_parquet(os.path.join(out, f"part-{k // chunk:03d}.parquet"), index=False)
        print(f"panel {k + len(cc)}/{len(codes)}  rows={len(df)}", flush=True)

    b = D.features([BENCH[POOL]], ["$close/Ref($close,1)-1"], START, END)
    b.columns = ["bench"]
    b = b.reset_index().drop(columns="instrument")
    b.to_parquet(os.path.join(DATA, "bench.parquet"), index=False)
    print("bench saved", len(b))


if __name__ == "__main__":
    build()
