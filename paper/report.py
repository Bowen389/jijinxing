"""
每日报告存档（由 sim.py summary 调用，也可单独运行：python paper/report.py）
  paper/signals/<日期>/README.md   当天的日报：账户总览 + 当日成交 + 收盘持仓 + 下一交易日清单（GitHub 打开该日期文件夹时直接显示）
  paper/accounts/history.csv       所有账户每天一行的历史总览（可用 Excel 打开）
  paper/README.md                  末尾追加“每日记录”表，每天一行，可点进当天日报
全部由累积保存的 nav.csv / trades.csv / signals/<日期>/ 重新生成，重复运行或补跑多天结果都一样。
"""
import os

import pandas as pd

PAPER = os.path.dirname(os.path.abspath(__file__))
ACC_DIR = os.path.join(PAPER, "accounts")
SIG_DIR = os.path.join(PAPER, "signals")
DISCLAIMER = "> 仅供学习研究，不构成投资建议。"


def _read(fn, **kw):
    if not os.path.exists(fn):
        return pd.DataFrame()
    try:
        return pd.read_csv(fn, encoding="utf-8-sig", **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _table(df):
    if df is None or not len(df):
        return ["无"]
    df = df.fillna("")
    return ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] + \
           ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]


def _pct(x):
    return f"{x:+.2%}" if pd.notna(x) else ""


def load_history(strats):
    rows = []
    for name in strats:
        nav = _read(os.path.join(ACC_DIR, name, "nav.csv"))
        if not len(nav):
            continue
        v = nav["总资产"].astype(float)
        nav["最大回撤"] = (v / v.cummax() - 1).cummin()
        nav["策略"] = name
        rows.append(nav)
    if not rows:
        return pd.DataFrame()
    h = pd.concat(rows, ignore_index=True)
    cols = ["日期", "策略", "总资产", "当日收益", "累计收益", "中证1000累计", "超额累计", "最大回撤", "持仓数", "现金", "成交笔数"]
    return h[[c for c in cols if c in h.columns]].sort_values(["日期", "策略"], key=lambda s: s.map(
        {n: i for i, n in enumerate(strats)}) if s.name == "策略" else s).reset_index(drop=True)


def day_report(d, h, strats, days, title):
    i = days.index(d)
    prev_l = f"[← {days[i - 1]}](../{days[i - 1]}/README.md)" if i > 0 else ""
    next_l = f"[{days[i + 1]} →](../{days[i + 1]}/README.md)" if i + 1 < len(days) else ""
    md = [f"# {title}日报 {d}", "", " ｜ ".join(x for x in [prev_l, "[返回总览](../../README.md)", next_l] if x), "",
          "## 账户", ""]
    t = h[h["日期"] == d].copy()
    show = pd.DataFrame({"策略": t["策略"], "总资产": t["总资产"].map(lambda x: f"{x:,.0f}"),
                         "当日": t["当日收益"].map(_pct), "累计": t["累计收益"].map(_pct),
                         "中证1000累计": t["中证1000累计"].map(_pct),
                         "超额累计": t["超额累计"].map(_pct) if "超额累计" in t else "",
                         "最大回撤": t["最大回撤"].map(lambda x: f"{x:.2%}"), "持仓": t["持仓数"],
                         "现金": t["现金"].map(lambda x: f"{x:,.0f}"), "当日成交": t["成交笔数"]})
    md += _table(show) + [""]
    for name in strats:
        if name not in set(t["策略"]):
            continue
        md += [f"## {name}", ""]
        tr = _read(os.path.join(ACC_DIR, name, "trades.csv"), dtype={"代码": str})
        tr = tr[tr["日期"] == d].drop(columns=["日期"]) if len(tr) else tr
        md += [f"**当日成交**（{len(tr)} 笔）", ""] + _table(tr) + [""]
        pos = _read(os.path.join(SIG_DIR, d, f"{name}_positions.csv"), dtype={"代码": str})
        if os.path.exists(os.path.join(SIG_DIR, d, f"{name}_positions.csv")):
            md += [f"<details><summary><b>收盘持仓</b>（{len(pos)} 只）</summary>", ""] + _table(pos) + ["", "</details>", ""]
        o = _read(os.path.join(SIG_DIR, d, f"{name}.csv"), dtype={"代码": str})
        if len(o) and "操作" in o:
            if (o["操作"] == "买入").any():
                o = pd.concat([o[o["操作"] != "替补"], o[o["操作"] == "替补"].head(5)])
            else:
                o = o[o["操作"] != "替补"]
            act = o[o["操作"] != "继续持有"]
            md += [f"**下一交易日清单**（信号日 {d}）", ""] + (_table(act) if len(act) else ["无操作（全部继续持有）"]) + [""]
        elif os.path.exists(os.path.join(SIG_DIR, d, f"{name}.csv")):
            md += [f"**下一交易日清单**（信号日 {d}）", "", "无操作", ""]
    md += [f"程序原始输出：本文件夹下的 `<策略>.txt`；完整清单：`<策略>.csv`。", "", DISCLAIMER]
    return md


def build(strats, title="模拟盘"):
    strats = list(strats)
    h = load_history(strats)
    if not len(h):
        return
    h.to_csv(os.path.join(ACC_DIR, "history.csv"), index=False, encoding="utf-8-sig", float_format="%.6g")
    days = sorted(h["日期"].unique())
    for d in days:
        os.makedirs(os.path.join(SIG_DIR, d), exist_ok=True)
        with open(os.path.join(SIG_DIR, d, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(day_report(d, h, strats, days, title)) + "\n")
    # 总览页末尾追加“每日记录”
    fn = os.path.join(PAPER, "README.md")
    if not os.path.exists(fn):
        return
    txt = open(fn, encoding="utf-8").read()
    txt = txt.split("\n## 每日记录\n")[0].rstrip("\n")
    names = [n for n in strats if n in set(h["策略"])]
    wide = h.pivot_table(index="日期", columns="策略", values="累计收益", aggfunc="last")[names]
    bench = h.groupby("日期")["中证1000累计"].last()
    lines = ["| 日期 | " + " | ".join(names) + " | 中证1000 | 日报 |", "|" + "---|" * (len(names) + 3)]
    for d in sorted(days, reverse=True):
        lines.append(f"| {d} | " + " | ".join(_pct(wide.loc[d, n]) for n in names) +
                     f" | {_pct(bench.get(d))} | [查看](signals/{d}/README.md) |")
    disc = ""
    if DISCLAIMER in txt:
        txt, disc = txt.split(DISCLAIMER, 1)
        disc = DISCLAIMER + disc
        txt = txt.rstrip("\n")
    out = txt + "\n\n## 每日记录\n\n各账户累计收益（最新在上）。每天的完整日报（账户、当日成交、收盘持仓、下一日清单）点“查看”；" \
                "所有数字也在 `paper/accounts/history.csv`。\n\n" + "\n".join(lines) + "\n"
    if disc:
        out += "\n" + disc.rstrip("\n") + "\n"
    open(fn, "w", encoding="utf-8").write(out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, PAPER)
    from sim import STRATS
    build(STRATS)
