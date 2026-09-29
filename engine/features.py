"""
分块计算 Qlib 表达式因子，只保留"当日在股票池内"的行，存成 parquet（float32）
  python -m engine.features --set retail
  python -m engine.features --set alpha158
输出 {DATA}/feat_{set}/part-*.parquet
"""
import argparse
import os
import shutil

import pandas as pd

from engine.common import DATA, START, END, init_qlib, pool_codes, read_parts


def get_config(name):
    if name == "retail":
        from retail_factors import retail_feature_config
        return retail_feature_config()
    if name == "behavior":
        from retail_factors import behavior_feature_config
        return behavior_feature_config()
    if name == "alpha158":
        from qlib.contrib.data.loader import Alpha158DL
        conf = {"kbar": {}, "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]}, "rolling": {}}
        f, n = Alpha158DL.get_feature_config(conf)
        return f, ["A_" + x for x in n]
    raise ValueError(name)


def build(name, chunk=100, kernels=1):
    from qlib.data import D
    init_qlib(kernels)
    fields, names = get_config(name)
    codes = pool_codes()
    out = os.path.join(DATA, f"feat_{name}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    import pyarrow.dataset as ds
    for k in range(0, len(codes), chunk):
        cc = codes[k:k + chunk]
        # 每批只读这批股票的成员表（全市场股票池时整表放不进 2GB 内存）
        member = read_parts(os.path.join(DATA, "panel"), columns=["datetime", "instrument", "in_pool"], start="2015-01-01",
                            filters_extra=ds.field("instrument").isin(cc) & ds.field("in_pool"))
        member = member.drop(columns="in_pool")
        df = D.features(cc, fields, START, END).astype("float32")
        df.columns = names
        df = df.reset_index()
        df["instrument"] = df["instrument"].astype(str)
        df = df.merge(member[member["instrument"].isin(cc)], on=["datetime", "instrument"], how="inner")
        df = df.replace([float("inf"), float("-inf")], float("nan"))
        df.to_parquet(os.path.join(out, f"part-{k // chunk:03d}.parquet"), index=False)
        print(f"{name}: {k + len(cc)}/{len(codes)} rows={len(df)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--chunk", type=int, default=100)
    a = ap.parse_args()
    build(a.set, a.chunk)
