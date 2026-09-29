"""
新数据因子：股东户数 + 融资融券（东方财富数据中心，data_fetch/fetch_em.py 抓取）
  python -m engine.em_features            # 股东户数 → {DATA}/feat_em
  python -m engine.em_features margin     # 融资融券 → {DATA}/feat_margin（研究中，暂未用于策略）
输出与面板股票池内行顺序一致，可和 feat_retail 直接拼接

防前视：
  股东户数  只用 公告日 < 当日 的记录（公告日当天盘后/晚间才发布，最早下一交易日可用）
  融资融券  T 日数据 T+1 早上才公布，而信号在 T 日收盘后计算 → 一律滞后 1 个交易日

因子（方向未统一，单因子检验给出 IC 符号）：
  股东户数  HN_CHG1 最近一期户数变化(对数) | HN_CHG2 近两期 | HN_CHG_ADJ 户均持股变化(剔除增发送转)
            HN_HOLDVAL 户均持股市值(对数，越小越"散") | HN_AGE 最近一期截止日距今天数 | HN_FRESH 公告后 20 天内=1
  融资      RZ_BUY5 / RZ_BUY20 融资买入额占成交额 | RZ_CHG5 / RZ_CHG20 融资余额变化 | RZ_MV 融资余额/市值
            RQ_RATIO 融券余额/融资余额 | IS_MARGIN 是否两融标的
"""
import os
import shutil

import numpy as np
import pandas as pd

from engine.common import DATA, EXTRA


def _roll(g, s, fn, w, minp=None):
    return s.groupby(g, sort=False).transform(lambda x: getattr(x.rolling(w, min_periods=minp or max(2, w // 2)), fn)())


def holders_table():
    h = pd.read_parquet(os.path.join(EXTRA, "em_holders.parquet"))
    h = h.dropna(subset=["holders", "notice"]).sort_values(["code", "end"]).drop_duplicates(["code", "end"], keep="last")
    g = h.groupby("code")
    h["h1"] = g["holders"].shift(1)
    h["h2"] = g["holders"].shift(2)
    h["a1"] = g["avg_hold"].shift(1)
    h["HN_CHG1"] = np.log(h["holders"] / h["h1"])
    h["HN_CHG2"] = np.log(h["holders"] / h["h2"])
    h["HN_CHG_ADJ"] = np.log(h["avg_hold"] / h["a1"])
    # 可用日 = 公告日 + 1 天；同一天可用多条时取截止日最新的
    h["avail"] = h["notice"] + pd.Timedelta(days=1)
    h = h.sort_values(["code", "avail", "end"]).drop_duplicates(["code", "avail"], keep="last")
    # 公告晚于后一期的旧记录（补发）不应覆盖新记录：可用日上保证截止日单调
    h["end_max"] = h.groupby("code")["end"].cummax()
    h = h[h["end"] >= h["end_max"]]
    h["instrument"] = np.where(h["code"].str[0] == "6", "SH", np.where(h["code"].str[0].isin(["0", "3"]), "SZ", "BJ")) + h["code"]
    return h[["instrument", "avail", "end", "holders", "shares", "HN_CHG1", "HN_CHG2", "HN_CHG_ADJ"]]


def build(kind="em"):
    pdir = os.path.join(DATA, "panel")
    out = os.path.join(DATA, "feat_em" if kind == "em" else "feat_margin")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    H = holders_table()
    mfp = os.path.join(EXTRA, "em_margin.parquet")
    M = None
    if kind == "margin":
        M = pd.read_parquet(mfp)
        M["instrument"] = np.where(M["code"].str[0] == "6", "SH", "SZ") + M["code"]
        M = M.drop_duplicates(["instrument", "date"]).rename(columns={"date": "datetime"})
        m_start = M["datetime"].min()
        print("股东户数", len(H), "条；融资融券", len(M), "条，起始", m_start.date(), flush=True)
    else:
        print("股东户数", len(H), "条", flush=True)
    for fn in sorted(os.listdir(pdir)):
        p = pd.read_parquet(os.path.join(pdir, fn), columns=["datetime", "instrument", "raw_close", "amount", "close",
                                                              "factor", "volume", "in_pool"])
        p = p.reset_index(drop=True)   # 面板已按 (instrument, datetime) 排序，保持原顺序
        f = pd.DataFrame({"datetime": p["datetime"].values, "instrument": p["instrument"].values})
        # ---------- 股东户数（as-of 合并）----------
        hh = H[H["instrument"].isin(p["instrument"].unique())].sort_values("avail")
        q = pd.merge_asof(p[["datetime", "instrument", "raw_close"]].reset_index().sort_values("datetime"),
                          hh, left_on="datetime", right_on="avail", by="instrument", direction="backward")
        q = q.set_index("index").loc[p.index]
        f["HN_CHG1"] = q["HN_CHG1"].values
        f["HN_CHG2"] = q["HN_CHG2"].values
        f["HN_CHG_ADJ"] = q["HN_CHG_ADJ"].values
        f["HN_HOLDVAL"] = np.log(q["shares"] * q["raw_close"] / q["holders"]).values
        f["HN_AGE"] = (q["datetime"] - q["end"]).dt.days.values
        f["HN_FRESH"] = ((q["datetime"] - q["avail"]).dt.days <= 20).astype(float).where(q["avail"].notna()).values
        # ---------- 融资融券（滞后 1 个交易日）----------
        if M is None:
            keep = p["in_pool"].values & (p["datetime"] >= pd.Timestamp("2015-01-01")).values
            f = f[keep]
            cols = [c for c in f.columns if c not in ("datetime", "instrument")]
            f[cols] = f[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
            f.to_parquet(os.path.join(out, fn), index=False)
            print(fn, len(f), "户数覆盖", round(f["HN_CHG1"].notna().mean(), 3), flush=True)
            continue
        r = p[["datetime", "instrument", "amount", "raw_close"]].merge(M, on=["instrument", "datetime"], how="left")
        g = r["instrument"].values
        lag = lambda s: s.groupby(g, sort=False).shift(1)  # noqa: E731
        has = r["rzye"].notna()
        # 两融标的：数据中出现过且最近 5 天有数据
        is_m = _roll(g, has.astype(float), "max", 5, 1)
        rzmre = r["rzmre"].fillna(0)
        amt = r["amount"] * 1000.0   # qlib amount 单位：千元
        buy5 = _roll(g, rzmre, "sum", 5, 3) / (_roll(g, amt, "sum", 5, 3) + 1)
        buy20 = _roll(g, rzmre, "sum", 20, 10) / (_roll(g, amt, "sum", 20, 10) + 1)
        rzye = r["rzye"].groupby(g, sort=False).ffill(limit=3)
        rz5 = rzye / rzye.groupby(g, sort=False).shift(5) - 1
        rz20 = rzye / rzye.groupby(g, sort=False).shift(20) - 1
        rzmv = rzye / r["mv"].groupby(g, sort=False).ffill(limit=3)
        rq = (r["rqyl"] * r["raw_close"]) / (rzye + 1)
        for name, s in [("RZ_BUY5", buy5), ("RZ_BUY20", buy20), ("RZ_CHG5", rz5), ("RZ_CHG20", rz20),
                        ("RZ_MV", rzmv), ("RQ_RATIO", rq)]:
            s = s.where(is_m > 0)
            f[name] = lag(s).values
        f["IS_MARGIN"] = lag(is_m.fillna(0)).values
        before = (f["datetime"] < m_start + pd.Timedelta(days=40)).values
        f.loc[before, ["RZ_BUY5", "RZ_BUY20", "RZ_CHG5", "RZ_CHG20", "RZ_MV", "RQ_RATIO", "IS_MARGIN"]] = np.nan
        f = f.drop(columns=[c for c in f.columns if c.startswith("HN_")])
        keep = p["in_pool"].values & (p["datetime"] >= pd.Timestamp("2015-01-01")).values
        f = f[keep]
        # 和其它特征集一样按面板文件内原顺序输出
        cols = [c for c in f.columns if c not in ("datetime", "instrument")]
        f[cols] = f[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
        f.to_parquet(os.path.join(out, fn), index=False)
        print(fn, len(f), "两融覆盖", round(f["IS_MARGIN"].fillna(0).mean(), 3), flush=True)


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "em")
