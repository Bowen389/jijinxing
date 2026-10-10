"""Incremental, dated ST/trading-status cache used by all execution paths.

Daily: --start is derived from the oldest pending account date; full research
backfill: python data_fetch/fetch_status.py --start 2014-06-01 --end YYYY-MM-DD.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.common import EXTRA, PROVIDER, pool_spans


def fetch_group(args):
    import baostock as bs
    codes, start, end = args
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    parts = []
    try:
        for inst in codes:
            rs = bs.query_history_k_data_plus(inst[:2].lower() + "." + inst[2:], "date,isST,tradestatus",
                                              start_date=start, end_date=end, frequency="d", adjustflag="3")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0":
                print(f"status unavailable {inst}: {rs.error_msg}", file=sys.stderr)
                continue
            t = pd.DataFrame(rows, columns=["datetime", "isST", "tradestatus"])
            t["datetime"] = pd.to_datetime(t["datetime"])
            for c in ("isST", "tradestatus"):
                t[c] = pd.to_numeric(t[c], errors="coerce")
            t["instrument"] = inst
            parts.append(t)
    finally:
        bs.logout()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    end = pd.Timestamp(a.end or pd.read_csv(os.path.join(PROVIDER, "calendars/day.txt"), header=None).iloc[-1, 0])
    account_dates = []
    held = set()
    for fp in Path("paper/accounts").glob("*/state.json"):
        state = json.loads(fp.read_text())
        if state.get("last_date"):
            account_dates.append(pd.Timestamp(state["last_date"]))
        held.update(state.get("positions", {}))
    start = pd.Timestamp(a.start) if a.start else min(account_dates + [end]) - pd.Timedelta(days=10)
    spans = pool_spans()
    codes = sorted(set(spans.loc[spans["start"] <= end, "instrument"]) | held)
    codes = [j for j in codes if j.startswith(("SH", "SZ"))]
    import baostock as bs
    bs.login()
    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if rs.error_code != "0" or not rows:
        raise RuntimeError("Stock basic data unavailable; previous cache preserved")
    basic = pd.DataFrame(rows, columns=rs.fields)
    basic["instrument"] = basic["code"].str.replace(".", "", regex=False).str.upper()
    basic["observed"] = pd.Timestamp.now(tz="Asia/Shanghai").normalize().tz_localize(None)
    os.makedirs(EXTRA, exist_ok=True)
    bfp = os.path.join(EXTRA, "security_basic.parquet")
    if os.path.exists(bfp):
        old_basic = pd.read_parquet(bfp).set_index("instrument")
        observed = old_basic.loc[old_basic["code_name"].fillna("").str.contains("退", regex=False), "observed"]
        risk = basic["code_name"].fillna("").str.contains("退", regex=False)
        basic.loc[risk, "observed"] = basic.loc[risk, "instrument"].map(observed).fillna(basic.loc[risk, "observed"])
    basic.to_parquet(bfp + ".tmp", index=False)
    os.replace(bfp + ".tmp", bfp)
    with Pool(a.workers) as pool:
        chunks = pool.map(fetch_group, [(codes[i::a.workers], str(start.date()), str(end.date())) for i in range(a.workers)])
    fp = os.path.join(EXTRA, "security_status.parquet")
    old = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()
    fresh = pd.concat(chunks, ignore_index=True)
    if fresh.empty:
        raise RuntimeError("No dated status returned; previous cache preserved")
    result = pd.concat([old, fresh], ignore_index=True).drop_duplicates(["datetime", "instrument"], keep="last")
    result.to_parquet(fp + ".tmp", index=False)
    os.replace(fp + ".tmp", fp)
    print(f"status {start.date()}..{end.date()}: {len(fresh)} rows; missing symbols cannot buy")


if __name__ == "__main__":
    main()
