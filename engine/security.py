"""Point-in-time ST/trading status. Missing status never permits a new buy."""
import os
import numpy as np
import pandas as pd
from engine.common import EXTRA, limit_pct


def status_for(frame):
    keys = frame[["datetime", "instrument"]].reset_index(drop=True)
    tables = []
    fp = os.path.join(EXTRA, "security_status.parquet")
    if os.path.exists(fp):
        tables.append(pd.read_parquet(fp))
    # Existing research downloads also contain historical isST/tradestatus.
    for inst in keys["instrument"].unique():
        fp = os.path.join(EXTRA, "turn", inst + ".parquet")
        if os.path.exists(fp):
            s = pd.read_parquet(fp, columns=["date", "isST", "tradestatus"]).rename(columns={"date": "datetime"})
            s["instrument"] = inst
            tables.insert(0, s)
    if tables:
        s = pd.concat(tables, ignore_index=True).drop_duplicates(["datetime", "instrument"], keep="last")
        s = keys.merge(s, on=["datetime", "instrument"], how="left")
    else:
        s = keys.copy()
        s["isST"], s["tradestatus"] = np.nan, np.nan
    s["status_known"] = s["isST"].notna() & s["tradestatus"].notna()
    s["is_st"] = s["isST"].eq(1)
    s["is_delisted"] = False
    s["listed_ok"] = False
    fp = os.path.join(EXTRA, "security_basic.parquet")
    if os.path.exists(fp):
        basic = pd.read_parquet(fp).drop_duplicates("instrument", keep="last")
        b = keys.merge(basic, on="instrument", how="left")
        out = pd.to_datetime(b["outDate"], errors="coerce")
        ipo = pd.to_datetime(b["ipoDate"], errors="coerce")
        s["is_delisted"] = out.notna() & (keys["datetime"].reset_index(drop=True) >= out)
        # Current names must never be applied backwards to historical rows.
        observed = pd.to_datetime(b["observed"], errors="coerce")
        s["is_delisted"] |= (b["code_name"].fillna("").str.contains("退", regex=False)
                              & (keys["datetime"].reset_index(drop=True) >= observed))
        s["listed_ok"] = ipo.notna() & (keys["datetime"].reset_index(drop=True) >= ipo + pd.Timedelta(days=30))
    s["buy_ok"] = (s["status_known"] & ~s["is_st"] & ~s["is_delisted"] & s["listed_ok"]
                   & s["tradestatus"].eq(1) & ~s["instrument"].str.startswith("BJ"))
    return s[["status_known", "is_st", "is_delisted", "buy_ok", "tradestatus"]].set_axis(frame.index)


def limit_flags(inst, dt, raw_close, raw_open, change, susp, is_st, known):
    lim = limit_pct(inst, dt, is_st)
    # Unknown main-board status: conservative 5% exit block, buys blocked above.
    main = ~(inst.str.startswith("SH688") | inst.str.startswith("SZ30") | inst.str.startswith("BJ"))
    lim[main.values & ~np.asarray(known, dtype=bool)] = 0.05
    prev = raw_close / (1 + change)
    up = np.round(prev * (1 + lim) + 1e-6, 2)
    dn = np.round(prev * (1 - lim) + 1e-6, 2)
    unknown_price = ~np.isfinite(prev) | (prev <= 0)
    return dict(lim=lim,
                up_lim=((raw_close >= up - 0.0051) | (change >= lim - 0.0015)) & ~susp | unknown_price,
                dn_lim=((raw_close <= dn + 0.0051) | (change <= -lim + 0.0015)) & ~susp | unknown_price,
                up_open=(raw_open >= up - 0.0051) & ~susp | unknown_price,
                dn_open=(raw_open <= dn + 0.0051) & ~susp | unknown_price)
