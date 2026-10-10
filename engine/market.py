"""Daily panel quotes and independently recorded post-auction execution prices."""
import os
import numpy as np
import pandas as pd
from engine.common import DATA, EXTRA, KEY, read_parts
from engine.security import status_for, limit_flags

QUOTE_COLS = ["raw_close", "raw_open", "factor", "up_lim", "dn_lim", "up_open", "dn_open", "susp", "buy_ok", "status_known", "is_st", "is_delisted", "in_pool", "change", "tradestatus"]


class PanelMarket:
    def __init__(self, panel, bench=None, execution=None):
        missing = set(KEY + QUOTE_COLS) - set(panel.columns)
        if missing:
            raise ValueError(f"Rebuild panel with dated security status; missing: {sorted(missing)}")
        self.panel = panel.set_index(KEY).sort_index()
        self.bench = bench if bench is not None else pd.Series(dtype=float)
        self.execution = execution if execution is not None else pd.DataFrame()
        self.cache = {}

    def day(self, d):
        try:
            return self.panel.xs(pd.Timestamp(d), level="datetime")
        except KeyError:
            return pd.DataFrame(columns=QUOTE_COLS)

    def quote(self, d, inst):
        try:
            q = self.panel.loc[(pd.Timestamp(d), inst)].to_dict()
        except KeyError:
            q = dict(raw_close=np.nan, raw_open=np.nan, factor=np.nan, susp=True, buy_ok=False,
                     up_lim=True, dn_lim=True, up_open=True, dn_open=True)
        for c in ("raw_close", "raw_open", "factor"):
            q[c] = float(q[c]) if pd.notna(q.get(c)) else float("nan")
            if c != "factor" and q[c] > 0:
                q[c] = round(q[c], 2)
        for c in ("up_lim", "dn_lim", "up_open", "dn_open", "susp"):
            q[c] = bool(q[c]) if pd.notna(q.get(c)) else True
        q["buy_ok"] = bool(q.get("buy_ok", False)) if pd.notna(q.get("buy_ok")) else False
        return q

class Market(PanelMarket):
    def __init__(self):
        self.DATA = DATA
        self.cache = {}
        self.bench = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]

    def day(self, d):
        d = pd.Timestamp(d)
        if d not in self.cache:
            p = read_parts(os.path.join(DATA, "panel"), columns=KEY + QUOTE_COLS, start=d, end=d)
            p["susp"] |= p["tradestatus"].eq(0)
            for c, values in limit_flags(p["instrument"], p["datetime"], p["raw_close"], p["raw_open"],
                                         p["change"], p["susp"], p["is_st"], p["status_known"]).items():
                p[c] = values
            self.cache[d] = p.set_index("instrument")
        return self.cache[d]

    def quote(self, d, inst):
        day = self.day(d)
        if inst in day.index:
            q = day.loc[inst].to_dict()
        else:
            from engine.common import init_qlib
            from qlib.data import D
            init_qlib()
            try:
                f = D.features([inst], ["$close/$factor", "$open/$factor", "$factor", "$volume", "$change"], d, d)
                r = f.iloc[0].values
                t = pd.DataFrame([dict(datetime=pd.Timestamp(d), instrument=inst, raw_close=r[0], raw_open=r[1],
                                       factor=r[2], susp=not (r[3] > 0), change=r[4])])
                st = status_for(t)
                flags = limit_flags(t["instrument"], t["datetime"], t["raw_close"], t["raw_open"], t["change"],
                                    t["susp"], st["is_st"], st["status_known"])
                q = t.iloc[0].to_dict()
                q.update({c: v[0] if isinstance(v, np.ndarray) else v.iloc[0] for c, v in flags.items()})
                q["buy_ok"] = bool(st["buy_ok"].iloc[0])
            except (IndexError, KeyError, ValueError, OSError):
                q = dict(raw_close=np.nan, raw_open=np.nan, factor=np.nan, susp=True, buy_ok=False,
                         up_lim=True, dn_lim=True, up_open=True, dn_open=True)
        for c in ("raw_close", "raw_open", "factor"):
            q[c] = float(q[c]) if pd.notna(q.get(c)) else float("nan")
            if c != "factor" and q[c] > 0:
                q[c] = round(q[c], 2)
        for c in ("up_lim", "dn_lim", "up_open", "dn_open", "susp"):
            q[c] = bool(q[c]) if pd.notna(q.get(c)) else True
        q["buy_ok"] = bool(q.get("buy_ok", False)) if pd.notna(q.get("buy_ok")) else False
        return q
