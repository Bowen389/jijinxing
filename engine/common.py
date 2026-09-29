"""公共配置：路径、Qlib 初始化、股票池成员、涨跌停幅度"""
import os

import numpy as np
import pandas as pd
import pyarrow as _pa

_pa.set_memory_pool(_pa.system_memory_pool())  # 用系统分配器，释放内存更及时

PROVIDER = os.environ.get("RA_PROVIDER", os.path.expanduser("~/.qlib/qlib_data/cn_data"))
DATA = os.environ.get("RA_DATA", os.path.expanduser("~/.qlib/retail_alpha_data"))   # 中间数据（可重建）
EXTRA = os.environ.get("RA_EXTRA", os.path.expanduser("~/.qlib/retail_extra"))      # 换手率/龙虎榜
POOL = os.environ.get("RA_POOL", "csi1000")
BENCH = {"csi300": "SH000300", "csi500": "SH000905", "csi1000": "SH000852"}
START, END = "2014-06-01", "2026-12-31"
KEY = ["datetime", "instrument"]
os.makedirs(DATA, exist_ok=True)

_inited = False


def init_qlib(kernels=1):
    global _inited
    if not _inited:
        import qlib
        qlib.init(provider_uri=PROVIDER, region="cn", kernels=kernels)
        _inited = True


def pool_spans(pool=POOL):
    df = pd.read_csv(os.path.join(PROVIDER, "instruments", f"{pool}.txt"), sep="\t", header=None,
                     names=["instrument", "start", "end"], parse_dates=["start", "end"])
    return df[df["end"] >= "2015-01-01"].reset_index(drop=True)


def pool_codes(pool=POOL):
    return sorted(pool_spans(pool)["instrument"].unique())


def limit_pct(inst: pd.Series, dt: pd.Series, is_st=None) -> np.ndarray:
    """板块涨跌停幅度：主板10%、创业板(2020-08-24后)/科创板20%、北交所30%、主板ST 5%"""
    inst = inst.astype(str)
    lim = np.full(len(inst), 0.10, dtype="float32")
    star = inst.str.startswith("SH688").values
    gem = (inst.str.startswith("SZ300") | inst.str.startswith("SZ301")).values
    bj = inst.str.startswith("BJ").values
    lim[star] = 0.20
    lim[gem & (dt.values >= np.datetime64("2020-08-24"))] = 0.20
    lim[bj] = 0.30
    if is_st is not None:
        st = np.asarray(is_st).astype(bool)
        lim[st & ~star & ~gem & ~bj] = 0.05
    return lim


def read_parts(path, columns=None, start=None, end=None, filters_extra=None):
    """按日期区间读取分块 parquet 目录（逐文件读取，控制峰值内存）"""
    import pyarrow as pa
    import pyarrow.dataset as ds
    f = None
    if start is not None:
        f = ds.field("datetime") >= pd.Timestamp(start)
    if end is not None:
        g = ds.field("datetime") <= pd.Timestamp(end)
        f = g if f is None else (f & g)
    if filters_extra is not None:
        f = filters_extra if f is None else (f & filters_extra)
    tabs = []
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".parquet"):
            continue
        d = ds.dataset(os.path.join(path, fn), format="parquet")
        t = d.to_table(columns=columns, filter=f, use_threads=False)
        if t.num_rows:
            tabs.append(t)
    if not tabs:
        return pd.DataFrame(columns=columns or [])
    t = pa.concat_tables(tabs)
    del tabs
    df = t.to_pandas(self_destruct=True, split_blocks=True)
    if "instrument" in df:
        df["instrument"] = df["instrument"].astype(str)
    return df
