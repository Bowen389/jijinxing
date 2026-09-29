"""下载证监会行业分类（baostock，当前快照）→ {EXTRA}/industry.parquet"""
import os
import sys

import baostock as bs
import pandas as pd

out = os.environ.get("RA_EXTRA", os.path.expanduser("~/.qlib/retail_extra"))
os.makedirs(out, exist_ok=True)
bs.login()
rs = bs.query_stock_industry()
rows = []
while rs.error_code == "0" and rs.next():
    rows.append(rs.get_row_data())
bs.logout()
df = pd.DataFrame(rows, columns=rs.fields)
df["instrument"] = df["code"].str.replace(".", "", regex=False).str.upper()
df = df[["instrument", "code_name", "industry"]]
df.to_parquet(os.path.join(out, "industry.parquet"), index=False)
print(len(df), df["industry"].nunique(), "个行业", file=sys.stderr)
