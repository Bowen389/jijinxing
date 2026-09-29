"""
东方财富数据中心：股东户数 + 融资融券个股明细（中证1000 历史成分股）

  python data_fetch/fetch_em.py holders      # 股东户数（含公告日，防前视）约 10 分钟
  python data_fetch/fetch_em.py margin       # 融资融券每日明细 约 30 分钟
  python data_fetch/fetch_em.py all
  python data_fetch/fetch_em.py update       # 每日增量：只拉最近 45 天公告的股东户数（几秒钟）

结果：$RA_EXTRA/em_holders.parquet、$RA_EXTRA/em_margin.parquet（支持断点续传，已抓的股票会跳过）
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

API = "https://datacenter-web.eastmoney.com/api/data/v1/get"
EXTRA = os.environ.get("RA_EXTRA", os.path.expanduser("~/.qlib/retail_extra"))
PROVIDER = os.environ.get("RA_PROVIDER", os.path.expanduser("~/.qlib/qlib_data/cn_data"))


def get(params, tries=8):
    url = API + "?" + urllib.parse.urlencode(params, safe="(),='")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
            d = json.loads(urllib.request.urlopen(req, timeout=5).read())
            if d.get("result") is None:
                return []
            return d["result"]["data"]
        except Exception as e:  # noqa: BLE001
            time.sleep(0.5 * (k + 1))
            err = e
    print("  失败", params.get("filter"), err, flush=True)
    return None


def get_all(params):
    """接口每页最多返回 500 条（pageSize 再大也会被静默截断），需要逐页取"""
    out, page = [], 1
    while True:
        p = dict(params, pageSize=500, pageNumber=page)
        rows = get(p)
        if rows is None:
            return None
        out += rows
        if len(rows) < 500:
            return out
        page += 1


def codes(pool="csi1000"):
    df = pd.read_csv(f"{PROVIDER}/instruments/{pool}.txt", sep="\t", header=None, names=["c", "s", "e"])
    df = df[df.e >= "2016-01-01"]
    return sorted(set(df.c.str[2:]))


def fetch(kind):
    out = f"{EXTRA}/em_{kind}.parquet"
    part = f"{EXTRA}/em_{kind}_parts"
    os.makedirs(part, exist_ok=True)
    cs = codes()
    done = {f[:6] for f in os.listdir(part)}
    todo = [c for c in cs if c not in done]
    print(f"[{kind}] 共 {len(cs)} 只，待抓 {len(todo)}", flush=True)
    t0 = time.time()

    def one(c):
        if kind == "holders":
            rows = get_all(dict(reportName="RPT_HOLDERNUM_DET",
                                columns="SECURITY_CODE,END_DATE,HOLDER_NUM,HOLD_NOTICE_DATE,AVG_HOLD_NUM,TOTAL_A_SHARES",
                                filter=f'(SECURITY_CODE="{c}")', sortColumns="END_DATE", sortTypes=1))
        else:
            rows = get_all(dict(reportName="RPTA_WEB_RZRQ_GGMX",
                                columns="DATE,SCODE,RZYE,RZMRE,RZCHE,RQYL,RQMCL,SZ",
                                filter=f'(SCODE="{c}")(DATE>=\'2015-06-01\')', sortColumns="DATE", sortTypes=1))
        if rows is None:
            return
        if rows:
            pd.DataFrame(rows).to_parquet(f"{part}/{c}.parquet")
        else:
            open(f"{part}/{c}.empty", "w").close()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(12) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    fs = [pd.read_parquet(f"{part}/{f}") for f in sorted(os.listdir(part)) if f.endswith(".parquet")]
    df = pd.concat(fs, ignore_index=True)
    if kind == "holders":
        df = df.rename(columns={"SECURITY_CODE": "code", "END_DATE": "end", "HOLD_NOTICE_DATE": "notice",
                                "HOLDER_NUM": "holders", "AVG_HOLD_NUM": "avg_hold", "TOTAL_A_SHARES": "shares"})
        df["end"] = pd.to_datetime(df.end)
        df["notice"] = pd.to_datetime(df.notice)
    else:
        df = df.rename(columns={"DATE": "date", "SCODE": "code", "RZYE": "rzye", "RZMRE": "rzmre", "RZCHE": "rzche",
                                "RQYL": "rqyl", "RQMCL": "rqmcl", "SZ": "mv"})
        df["date"] = pd.to_datetime(df.date)
        for c in ["rzye", "rzmre", "rzche", "rqyl", "rqmcl", "mv"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df.to_parquet(out)
    print(f"[{kind}] 保存 {out}  {len(df)} 行", flush=True)


def update_holders(days=45):
    """按公告日增量拉取全市场最近的股东户数公告，合并进 em_holders.parquet"""
    out = f"{EXTRA}/em_holders.parquet"
    old = pd.read_parquet(out)
    since = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    rows = get_all(dict(reportName="RPT_HOLDERNUM_DET",
                        columns="SECURITY_CODE,END_DATE,HOLDER_NUM,HOLD_NOTICE_DATE,AVG_HOLD_NUM,TOTAL_A_SHARES",
                        filter=f"(HOLD_NOTICE_DATE>='{since}')", sortColumns="HOLD_NOTICE_DATE", sortTypes=1))
    if not rows:
        print("没有新公告或请求失败", flush=True)
        return
    new = pd.DataFrame(rows).rename(columns={"SECURITY_CODE": "code", "END_DATE": "end", "HOLD_NOTICE_DATE": "notice",
                                             "HOLDER_NUM": "holders", "AVG_HOLD_NUM": "avg_hold", "TOTAL_A_SHARES": "shares"})
    new["end"] = pd.to_datetime(new.end)
    new["notice"] = pd.to_datetime(new.notice)
    df = pd.concat([old, new[old.columns]], ignore_index=True).drop_duplicates(["code", "end"], keep="last")
    df.to_parquet(out)
    print(f"[holders] 增量 {len(new)} 条（公告日 ≥ {since}），合计 {len(df)} 行，最新公告 {df.notice.max().date()}", flush=True)


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else "all"
    if k == "update":
        update_holders()
        sys.exit(0)
    for kind in (["holders", "margin"] if k == "all" else [k]):
        fetch(kind)
