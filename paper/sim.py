"""
进攻版 S_top3 模拟盘：一个 10 万元账户，每个交易日按 senti/live.py 的清单自动成交、记账、出下一日清单
  python paper/sim.py run            # 处理所有未处理的交易日（开盘成交 → 收盘盯市 → 生成下一日清单），可重复运行
  python paper/sim.py summary        # 只重新生成 paper/README.md

每个交易日 d：
  1) 按 d 日开盘价执行上一交易日的清单：先卖（开盘一字跌停/停牌卖不出 → 继续持有），再买（开盘一字涨停/停牌/资金不足一手 → 按顺序用替补）
     费用：买入万5、卖出千1（含印花税），每笔最低 5 元；单边滑点千1
  2) d 日收盘价盯市（除权除息按复权因子折算股数，相当于分红再投资）
  3) 用账户的真实持仓和现金调用 senti/live.py 生成 d+1 的清单
状态：paper/accounts/S_top3/（state.json / nav.csv / trades.csv / positions.csv）；每日清单 paper/signals/<日期>/
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PAPER = os.path.join(ROOT, "paper")
ACC_DIR = os.path.join(PAPER, "accounts")
SIG_DIR = os.path.join(PAPER, "signals")
INIT_CASH = float(os.environ.get("PAPER_CASH", 100_000))

STRATS = {
    "S_top3": dict(kind="conc", family="S",
                   desc="进攻版：排名平均（大涨概率 − 大跌概率，六个 5 日模型），持 3 只，跌出前 60 才卖，不择时，开盘成交"),
}

from engine.execution import Ledger, ETF_CODE, code_of, execute_conc, EXECUTION_VERSION
from engine.market import Market


# ============================================================================ 工具
def inst_of(code):
    """'603103.SH' -> 'SH603103'"""
    c = str(code).strip()
    if "." in c:
        n, ex = c.split(".")
        return ex.upper() + n
    from strategies.live import norm_code
    return norm_code(c)


def trading_days():
    from engine.common import PROVIDER
    cal = pd.read_csv(os.path.join(PROVIDER, "calendars", "day.txt"), header=None)[0]
    return [pd.Timestamp(x) for x in cal]


def names_table():
    from engine.common import EXTRA
    return pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")["code_name"].to_dict()


# ============================================================================ 账户
class Account(Ledger):
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(ACC_DIR, name)
        self.fn = os.path.join(self.dir, "state.json")
        if os.path.exists(self.fn):
            self.s = json.load(open(self.fn, encoding="utf-8"))
        else:
            self.s = None
        self.trades = []
        self.navs = []

    @property
    def exists(self):
        return self.s is not None

    def init(self, start):
        self.s = dict(strategy=self.name, init_cash=INIT_CASH, start_date=str(start.date()), last_date=None,
                      cash=INIT_CASH, etf_value=0.0, positions={}, pending=None, execution_version=EXECUTION_VERSION, execution_start=str(start.date()))

    @staticmethod
    def write_csv(df, fn):
        tmp = fn + ".tmp"
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, fn)

    # ----- 保存
    def save(self, names):
        os.makedirs(self.dir, exist_ok=True)
        processed = [r["日期"] for r in self.navs]
        if self.trades or processed:
            fn = os.path.join(self.dir, "trades.csv")
            df = pd.DataFrame(self.trades, columns=["日期", "操作", "代码", "名称", "股数", "成交价", "金额", "费用", "备注"])
            replaced = set(processed) | set(df["日期"])
            if os.path.exists(fn):
                old = pd.read_csv(fn, encoding="utf-8-sig")
                retained = old[~old["日期"].isin(replaced)]
                df = retained if df.empty else df if retained.empty else pd.concat([retained, df], ignore_index=True)
            self.write_csv(df, fn)
            self.trades = []
        if self.navs:
            fn = os.path.join(self.dir, "nav.csv")
            df = pd.DataFrame(self.navs)
            if os.path.exists(fn):
                old = pd.read_csv(fn, encoding="utf-8-sig")
                df = pd.concat([old[~old["日期"].isin(df["日期"])], df], ignore_index=True)
            self.write_csv(df, fn)
            self.navs = []
        rows = []
        for j, p in sorted(self.s["positions"].items()):
            mv = p["shares"] * p.get("last_px", 0.0)
            rows.append(dict(代码=code_of(j), 名称=names.get(j, ""), 股数=round(p["shares"], 2), 现价=p.get("last_px"),
                             市值=round(mv, 2), 成本=round(p["cost"], 2), 浮动盈亏=round(mv - p["cost"], 2),
                             买入日期=p.get("buy_date", "")))
        if self.s["etf_value"] > 0:
            rows.append(dict(代码=ETF_CODE, 名称="中证1000ETF（指数近似）", 股数="", 现价="", 市值=round(self.s["etf_value"], 2),
                             成本="", 浮动盈亏="", 买入日期=""))
        positions = pd.DataFrame(rows, columns=["代码", "名称", "股数", "现价", "市值", "成本", "浮动盈亏", "买入日期"])
        self.write_csv(positions, os.path.join(self.dir, "positions.csv"))
        if self.s.get("last_date"):      # 每天的收盘持仓另存一份到当天的清单文件夹（日报用）
            dd = os.path.join(SIG_DIR, self.s["last_date"])
            os.makedirs(dd, exist_ok=True)
            self.write_csv(positions, os.path.join(dd, f"{self.name}_positions.csv"))
        tmp = self.fn + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.s, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.fn)


def holdings_file(acc, with_date=False):
    fn = os.path.join(PAPER, ".tmp", f"{acc.name}_hold.csv")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    rows = [dict(code=code_of(j), shares=p["shares"], buy_date=p.get("buy_date", "")) for j, p in acc.s["positions"].items()]
    df = pd.DataFrame(rows, columns=["code", "shares", "buy_date"])
    if not with_date:
        df = df[["code", "shares"]]
    df.to_csv(fn, index=False)
    return fn if len(rows) else None


def gen_signal(acc, d):
    """返回 (pending dict, 清单 DataFrame, 屏幕输出)"""
    from senti import live as SL
    ns = argparse.Namespace(holdings=holdings_file(acc), cash=float(acc.s["cash"]), date=str(d.date()))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn = SL.cmd_signal(ns)
    text = buf.getvalue()
    out = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
    pend = dict(signal_date=str(d.date()), kind="conc")
    pend["sell"] = [inst_of(c) for c in out.loc[out["操作"] == "卖出", "代码"]]
    pend["why"] = {inst_of(r.代码): str(r.备注) for r in out[out["操作"] == "卖出"].itertuples()}
    pend["buy"] = [dict(inst=inst_of(c)) for c in out.loc[out["操作"] == "买入", "代码"]]
    pend["backup"] = [dict(inst=inst_of(c)) for c in out.loc[out["操作"] == "替补", "代码"]]
    pend["slot"] = acc.total() / SL.CFG["k"]
    dd = os.path.join(SIG_DIR, str(d.date()))
    os.makedirs(dd, exist_ok=True)
    out.to_csv(os.path.join(dd, f"{acc.name}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(dd, f"{acc.name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return pend, out, text


# ============================================================================ 主流程
def strat_cfg(name):
    from senti.live import CFG
    return CFG


def run_one(name, start=None, until=None):
    mk = Market()
    names = names_table()
    days = trading_days()
    data_last = mk.bench.index.max()
    days = [d for d in days if d <= data_last]
    if until:
        days = [d for d in days if d <= pd.Timestamp(until)]
    if not days:
        raise ValueError("所选区间没有可处理的交易日")
    acc = Account(name)
    if not acc.exists:
        s = pd.Timestamp(start) if start else days[-1]
        candidates = [d for d in days if d <= s]
        if not candidates:
            raise ValueError("起始日早于可用行情")
        s = max(candidates)
        acc.init(s)
        acc.navs.append(dict(日期=str(s.date()), 现金=round(acc.s["cash"], 2), 股票市值=0.0,
                             总资产=round(INIT_CASH, 2), 当日收益=0.0, 累计收益=0.0, 中证1000当日=0.0,
                             中证1000累计=0.0, 超额累计=0.0, 持仓数=0, 成交笔数=0))
        acc.s["bench_cum"] = 1.0
        print(f"[{name}] 新建账户：初始资金 {INIT_CASH:,.0f}，起始日 {s.date()}", flush=True)
        pend, out, text = gen_signal(acc, s)
        acc.s["pending"] = pend
        acc.s["last_date"] = str(s.date())
        acc.save(names)
        print(text, flush=True)
        todo = [d for d in days if d > s]
    else:
        todo = [d for d in days if d > pd.Timestamp(acc.s["last_date"])]
    if not todo:
        print(f"[{name}] 已是最新（{acc.s['last_date']}），数据最新交易日 {data_last.date()}", flush=True)
        return
    cfg = strat_cfg(name)
    # Keep old cash/positions/nav intact; start corrected execution prospectively.
    if acc.s.get("execution_version") != EXECUTION_VERSION:
        acc.s["execution_version"] = EXECUTION_VERSION
        acc.s["execution_start"] = str(todo[0].date())
    for d in todo:
        day = mk.day(d)
        pool = day[day["in_pool"].fillna(False)]
        acc.s["data_quality"] = dict(date=str(d.date()), pool_count=len(pool),
                                    unknown_status=int((~pool["status_known"]).sum()),
                                    eligible_buys=int(pool["buy_ok"].sum()))
        prev_total = acc.total()
        n0 = len(acc.trades)
        pend = acc.s.get("pending")
        # 开盘成交前按开盘价格估值并做除权折算
        acc.mark(mk, d, at="open")
        if pend:
            execute_conc(acc, mk, d, pend, names, cfg)
        acc.mark(mk, d)
        tot = acc.total()
        b = float(mk.bench.get(d, 0.0))
        acc.s["bench_cum"] = acc.s.get("bench_cum", 1.0) * (1 + b)
        cum = tot / acc.s["init_cash"] - 1
        acc.navs.append(dict(日期=str(d.date()), 现金=round(acc.s["cash"], 2), 股票市值=round(acc.stock_value(), 2),
                             总资产=round(tot, 2), 当日收益=round(tot / prev_total - 1, 6),
                             累计收益=round(cum, 6), 中证1000当日=round(b, 6), 中证1000累计=round(acc.s["bench_cum"] - 1, 6),
                             超额累计=round(cum - (acc.s["bench_cum"] - 1), 6), 持仓数=len(acc.s["positions"]),
                             成交笔数=len([t for t in acc.trades[n0:] if t["操作"] in ("买入", "卖出")])))
        print(f"[{name}] {d.date()} 成交 {len(acc.trades) - n0} 笔，总资产 {tot:,.0f}（当日 {tot / prev_total - 1:+.2%}，"
              f"累计 {cum:+.2%}），持仓 {len(acc.s['positions'])} 只", flush=True)
        pend, out, text = gen_signal(acc, d)
        acc.s["pending"] = pend
        acc.s["last_date"] = str(d.date())
        acc.save(names)
    print(text, flush=True)


def summary():
    name = "S_top3"
    if not days:
        raise ValueError("所选区间没有可处理的交易日")
    acc = Account(name)
    if not acc.exists:
        return
    nav = pd.read_csv(os.path.join(acc.dir, "nav.csv"), encoding="utf-8-sig")
    last = nav.iloc[-1]
    v = nav["总资产"]
    mdd = float((v / v.cummax() - 1).min())
    row = (f"| {acc.s['start_date']} | {last['日期']} | {last['总资产']:,.0f} | {last['当日收益']:+.2%} | {last['累计收益']:+.2%} | "
           f"{last['中证1000累计']:+.2%} | {last['超额累计']:+.2%} | {mdd:.2%} | {int(last['持仓数'])} | {acc.s['cash']:,.0f} |")
    pos_md = ["无持仓"]
    pf = os.path.join(acc.dir, "positions.csv")
    if os.path.exists(pf):
        try:
            ps = pd.read_csv(pf, dtype={"代码": str}, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            ps = pd.DataFrame()
        if len(ps):
            ps = ps.fillna("")
            pos_md = ["| " + " | ".join(ps.columns) + " |", "|" + "---|" * len(ps.columns)] + \
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in ps.itertuples(index=False)]
    p = acc.s.get("pending") or {}
    sd = p.get("signal_date", "")
    fn = os.path.join(SIG_DIR, sd, f"{name}.csv")
    ord_md = ["无操作"]
    if os.path.exists(fn):
        o = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
        if (o["操作"] == "买入").any():
            o = pd.concat([o[o["操作"] != "替补"], o[o["操作"] == "替补"].head(5)])
        else:
            o = o[o["操作"] != "替补"]
        if len(o):
            cols = ["操作", "代码", "名称", "排名", "大涨概率", "大跌概率", "五日模型名次", "股数", "参考价", "参考金额", "备注"]
            o = o[cols].fillna("")
            ord_md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
                     ["| " + " | ".join(str(x) for x in r) + " |" for r in o.itertuples(index=False)]
    execution_note = (f"执行 v{acc.s.get('execution_version', 1)}，修复口径从 {acc.s.get('execution_start', '尚未迁移')} 起生效；"
                      "此前净值保留原记录。开盘成交是日线价格近似，不保证竞价排队成交。")
    quality = acc.s.get("data_quality", {})
    quality_note = f"最近处理日证券状态：{quality}" if quality else "证券状态尚未按修复版检查；下次处理新交易日时更新。"
    md = ["# 进攻版 S_top3 模拟盘（10 万元）", "",
          "打分 = 排名平均（大涨概率 − 大跌概率，K_top3 的六个 5 日模型）；中证1000 内持 3 只，排名跌出前 60 才卖，不择时，"
          "T 日收盘后出清单、T+1 开盘成交。GitHub Actions 每个交易日北京时间 17:00 自动运行，数据晚发布时之后的运行会自动补齐。", "",
          execution_note, "", quality_note, "", "## 账户", "",
          "| 起始日 | 最新日 | 总资产 | 当日 | 累计 | 中证1000累计 | 超额 | 最大回撤 | 持仓 | 现金 |",
          "|---|---|---|---|---|---|---|---|---|---|", row, "",
          "## 当前持仓", "", *pos_md, "",
          f"## 下一交易日操作清单（信号日 {sd}，下一交易日开盘集合竞价成交）", "", *ord_md, "",
          "明细：`paper/accounts/S_top3/`（nav.csv 每日净值、trades.csv 成交、positions.csv 持仓）；每日完整清单和程序输出在 `paper/signals/<日期>/`。", "",
          "> 仅供学习研究，不构成投资建议。持 3 只的集中策略波动很大（研究中最大回撤 −37% 到 −45%）；模拟盘按开盘价成交，未计冲击成本。"]
    open(os.path.join(PAPER, "README.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    sys.path.insert(0, PAPER)
    import report                    # 每日日报 + 历史总览（paper/report.py）
    report.build(STRATS, "进攻版模拟盘")
    print(row)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", help="新建账户的起始信号日（默认最新交易日）")
    r.add_argument("--until", help="只处理到这一天（测试用）")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "run":
        run_one("S_top3", a.start, a.until)
    summary()


if __name__ == "__main__":
    main()
