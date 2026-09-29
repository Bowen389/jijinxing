"""
组合模拟盘：一个 10 万元账户，按 strategies2/README.md 第 11 节的做法分成两个仓位
  主仓 C_top50（三模型集成，持 50 只，每天换 2 只，R3 风控，尾盘成交）
  机会仓 FB_dip（低位首板次日低开 2–7% 开盘买，次日收盘卖，最多 2 只，每只半个机会仓）
两个仓位分开记账、各自用自己的现金和持仓调用策略自带的 signal 工具；每天收盘后用现金把两仓调回目标比例（不强制卖股票）

  python paper/sim.py run            # 处理所有未处理的交易日（成交 → 盯市 → 再平衡 → 生成下一日清单），可重复运行
  python paper/sim.py summary        # 只重新生成 paper/README.md

每个交易日 d：
  1) 成交：C_top50 按 d 日收盘价执行上一日清单（涨停/停牌用替补，跌停/停牌卖不出继续持有；R3 半仓时卖出约一半换中证1000ETF）；
          FB_dip 开盘价落在买入区间才买，d 日之前买入的在 d 日收盘卖出（跌停/停牌顺延）
  2) 盯市：d 日收盘价估值（除权除息按复权因子折算股数）
  3) 再平衡：机会仓目标 = 总资产 × PAPER_FB_RATIO（默认 30%），只用现金在两仓之间划转
  4) 出清单：两个仓位各自生成 d+1 的清单
状态：paper/accounts/{C_top50,FB_dip}/（各仓 state.json / nav.csv / trades.csv / positions.csv），
      paper/accounts/combined_nav.csv（组合净值）；每日清单 paper/signals/<日期>/
中证1000ETF 没有单独的价格数据，用中证1000 指数日收益近似
"""
import argparse
import contextlib
import io
import json
import math
import os
import subprocess
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

# kind: dropout = 尾盘成交 + R3；fb = 机会仓
FB_RATIO = float(os.environ.get("PAPER_FB_RATIO", 0.30))     # 机会仓占总资金的比例（README 第 11 节：30%–50%）
STRATS = {
    "C_top50": dict(kind="dropout", family="C", desc="主仓：三模型集成（含股东户数），持 50 只，每天换 2 只，R3 风控，尾盘成交"),
    "FB_dip": dict(kind="fb", family="FB", desc="机会仓：低位首板次日低开 2–7% 开盘买，次日收盘卖，最多 2 只"),
}
ETF_CODE = "512100"
# 数据源的复权因子每天有约 0.01% 的计算噪声；变化超过 0.3% 才认为是除权除息（更小的分红忽略）
ADJ_TOL = 0.003
FB_BUY_COMM, FB_SELL_COMM, FB_STAMP, FB_SLIP = 0.00025, 0.00025, 0.0005, 0.001


# ============================================================================ 工具
def inst_of(code):
    """'603103.SH' -> 'SH603103'"""
    c = str(code).strip()
    if "." in c:
        n, ex = c.split(".")
        return ex.upper() + n
    from strategies.live import norm_code
    return norm_code(c)


def code_of(inst):
    return inst[2:] + "." + inst[:2]


def lot_of(inst):
    return 200 if inst.startswith("SH688") else 100


def fee(amount, rate, min_cost=5.0, stamp=0.0):
    if amount <= 0:
        return 0.0
    return max(amount * rate, min_cost) + amount * stamp


def trading_days():
    from engine.common import PROVIDER
    cal = pd.read_csv(os.path.join(PROVIDER, "calendars", "day.txt"), header=None)[0]
    return [pd.Timestamp(x) for x in cal]


class Market:
    """某一日的行情（未复权价、复权因子、涨跌停/停牌），带缓存"""

    COLS = ["datetime", "instrument", "raw_close", "raw_open", "factor", "up_lim", "dn_lim", "up_open", "dn_open", "susp"]

    def __init__(self):
        from engine.common import DATA
        self.DATA = DATA
        self.cache = {}
        b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
        self.bench = b

    def day(self, d):
        d = pd.Timestamp(d)
        if d not in self.cache:
            from engine.common import read_parts
            p = read_parts(os.path.join(self.DATA, "panel"), columns=self.COLS, start=d, end=d)
            self.cache[d] = p.set_index("instrument")
        return self.cache[d]

    def quote(self, d, inst):
        """返回 dict(close, open, factor, up_lim, dn_lim, up_open, dn_open, susp)；面板外的股票从 Qlib 全市场取"""
        t = self.day(d)
        if inst in t.index:
            r = t.loc[inst]
            q = {k: r[k] for k in self.COLS[2:]}
        else:
            q = self._qlib_quote(d, inst)
        for k in ("raw_close", "raw_open", "factor"):
            q[k] = float(q[k]) if q[k] is not None and not pd.isna(q[k]) else float("nan")
        for k in ("up_lim", "dn_lim", "up_open", "dn_open", "susp"):
            q[k] = bool(q[k]) if q[k] is not None and not pd.isna(q[k]) else (k == "susp")
        for k in ("raw_close", "raw_open"):
            if q[k] > 0:
                q[k] = round(q[k], 2)       # 复权价反推的未复权价有微小误差，还原到分
        if not q["raw_close"] > 0:
            q["susp"] = True
        return q

    def _qlib_quote(self, d, inst):
        from engine.common import init_qlib
        from qlib.data import D
        init_qlib()
        try:
            f = D.features([inst], ["$close/$factor", "$open/$factor", "$factor", "$volume", "$change"], d, d)
            r = f.iloc[0].values
            vol = r[3]
            return dict(raw_close=r[0], raw_open=r[1], factor=r[2], up_lim=False, dn_lim=False, up_open=False,
                        dn_open=False, susp=not (vol > 0))
        except Exception:  # noqa: BLE001
            return dict(raw_close=np.nan, raw_open=np.nan, factor=np.nan, up_lim=False, dn_lim=False, up_open=False,
                        dn_open=False, susp=True)


def names_table():
    from engine.common import EXTRA
    return pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")["code_name"].to_dict()


# ============================================================================ 账户
class Account:
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

    def init(self, start, cash):
        self.s = dict(strategy=self.name, init_cash=cash, start_date=str(start.date()), last_date=None,
                      cash=cash, etf_value=0.0, positions={}, pending=None, transfer_in=0.0)

    # ----- 估值
    def mark(self, mk, d):
        """复权折算股数 + 按 d 日收盘估值"""
        val = 0.0
        for j, p in self.s["positions"].items():
            q = mk.quote(d, j)
            f0 = p.get("factor") or 0
            if q["factor"] > 0 and f0 > 0 and abs(q["factor"] / f0 - 1) > ADJ_TOL:     # 除权除息：送转/分红折算成股数
                p["shares"] = round(p["shares"] * q["factor"] / f0, 2)
                p["factor"] = q["factor"]
            elif q["factor"] > 0 and f0 <= 0:
                p["factor"] = q["factor"]
            if q["raw_close"] > 0:
                p["last_px"] = q["raw_close"]
            val += p["shares"] * p.get("last_px", 0.0)
        return val

    def total(self):
        return self.s["cash"] + self.s["etf_value"] + sum(p["shares"] * p.get("last_px", 0.0)
                                                          for p in self.s["positions"].values())

    def stock_value(self):
        return sum(p["shares"] * p.get("last_px", 0.0) for p in self.s["positions"].values())

    # ----- 成交
    def trade(self, d, action, inst, shares, px, f, note="", name=""):
        amt = shares * px
        if action == "买入":
            self.s["cash"] -= amt + f
            p = self.s["positions"].get(inst)
            if p:
                p["cost"] += amt + f
                p["shares"] += shares
            else:
                self.s["positions"][inst] = dict(shares=float(shares), cost=amt + f, buy_date=str(d.date()),
                                                 last_px=px, factor=None)
        else:
            self.s["cash"] += amt - f
            p = self.s["positions"][inst]
            if shares >= p["shares"] - 1e-6:
                del self.s["positions"][inst]
            else:
                p["cost"] *= 1 - shares / p["shares"]
                p["shares"] -= shares
        self.trades.append(dict(日期=str(d.date()), 操作=action, 代码=code_of(inst) if inst != "ETF" else ETF_CODE,
                                名称=name, 股数=int(shares) if float(shares).is_integer() else round(shares, 2), 成交价=round(px, 3), 金额=round(amt, 2),
                                费用=round(f, 2), 备注=note))

    # ----- 保存
    def save(self, names):
        os.makedirs(self.dir, exist_ok=True)
        json.dump(self.s, open(self.fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if self.trades:
            fn = os.path.join(self.dir, "trades.csv")
            df = pd.DataFrame(self.trades)
            df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False, encoding="utf-8-sig")
            self.trades = []
        if self.navs:
            fn = os.path.join(self.dir, "nav.csv")
            df = pd.DataFrame(self.navs)
            if os.path.exists(fn):
                old = pd.read_csv(fn, encoding="utf-8-sig")
                df = pd.concat([old[~old["日期"].isin(df["日期"])], df], ignore_index=True)
            df.to_csv(fn, index=False, encoding="utf-8-sig")
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
        pd.DataFrame(rows, columns=["代码", "名称", "股数", "现价", "市值", "成本", "浮动盈亏", "买入日期"]).to_csv(
            os.path.join(self.dir, "positions.csv"), index=False, encoding="utf-8-sig")


# ============================================================================ 成交逻辑
def execute_dropout(acc, mk, d, pend, names, cfg):
    """A/C 系列：d 日收盘价成交"""
    oc, cc, mc = cfg["open_cost"], cfg["close_cost"], cfg["min_cost"]
    pos = acc.s["positions"]
    # 1) 卖出清单
    for j in pend["sell"]:
        if j not in pos:
            continue
        q = mk.quote(d, j)
        if q["susp"] or q["dn_lim"]:
            acc.trades.append(dict(日期=str(d.date()), 操作="卖出失败", 代码=code_of(j), 名称=names.get(j, ""), 股数=round(pos[j]["shares"], 2),
                                   成交价="", 金额="", 费用="", 备注="跌停/停牌，继续持有"))
            continue
        sh = pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, q["raw_close"], fee(sh * q["raw_close"], cc, mc), "清单卖出", names.get(j, ""))
    # 2) R3
    expo_next = pend.get("expo_next", 1.0)
    total = acc.total()
    if expo_next < 1:
        target = total * (1 - expo_next)
        if acc.s["etf_value"] < 0.9 * target:
            need = target - acc.s["etf_value"]
            sv = acc.stock_value()
            frac = min(1.0, need / sv) if sv > 0 else 0
            # 每只卖出约一半（按手取整）；只有 1 手的小仓位取整后为 0，再逐手补卖，直到凑够 ETF 目标
            plan = {}
            for j in pos:
                q = mk.quote(d, j)
                if q["susp"] or q["dn_lim"]:
                    continue
                plan[j] = [math.floor(pos[j]["shares"] * frac / lot_of(j)) * lot_of(j), q["raw_close"]]
            val = lambda: sum(sh * px for sh, px in plan.values())  # noqa: E731
            while val() < need * 0.95:
                left = [(pos[j]["shares"] - sh) * px for j, (sh, px) in plan.items()]
                if not plan or max(left) < 1:
                    break
                j = list(plan)[int(np.argmax(left))]
                plan[j][0] = min(plan[j][0] + lot_of(j), pos[j]["shares"])
            got = 0.0
            for j, (sh, px) in plan.items():
                if sh <= 0:
                    continue
                if pos[j]["shares"] - sh < lot_of(j):      # 剩下不足 1 手（除权后的零股）就整笔卖掉
                    sh = pos[j]["shares"]
                amt = sh * px
                f = fee(amt, cc, mc)
                acc.trade(d, "卖出", j, sh, px, f, "R3 半仓：卖出约一半换 ETF", names.get(j, ""))
                got += amt - f
            buy_amt = min(need, got, acc.s["cash"])
            if buy_amt > 100:
                f = fee(buy_amt, oc, mc)
                acc.s["cash"] -= buy_amt
                acc.s["etf_value"] += buy_amt - f
                acc.trades.append(dict(日期=str(d.date()), 操作="买入", 代码=ETF_CODE, 名称="中证1000ETF（指数近似）", 股数="",
                                       成交价="", 金额=round(buy_amt - f, 2), 费用=round(f, 2), 备注="R3 半仓"))
    elif acc.s["etf_value"] > 0:
        v = acc.s["etf_value"]
        f = fee(v, oc, mc)
        acc.s["cash"] += v - f
        acc.s["etf_value"] = 0.0
        acc.trades.append(dict(日期=str(d.date()), 操作="卖出", 代码=ETF_CODE, 名称="中证1000ETF（指数近似）", 股数="",
                               成交价="", 金额=round(v, 2), 费用=round(f, 2), 备注="解除 R3，卖出全部 ETF"))
    # 3) 买入（买不进/资金不足一手 → 按顺序用替补）
    backups = [b for b in pend["backup"]]
    done = set(pos)

    def try_buy(j, sh, note):
        if j in done or sh <= 0:
            return False
        q = mk.quote(d, j)
        if q["susp"] or q["up_lim"] or not q["raw_close"] > 0:
            return False
        px, lot = q["raw_close"], lot_of(j)
        while sh > 0 and sh * px + fee(sh * px, oc, mc) > acc.s["cash"]:
            sh -= lot
        if sh <= 0:
            return False
        acc.trade(d, "买入", j, sh, px, fee(sh * px, oc, mc), note, names.get(j, ""))
        pos[j]["factor"] = q["factor"]
        done.add(j)
        return True

    for b in pend["buy"]:
        if try_buy(b["inst"], b["shares"], "清单买入"):
            continue
        while backups:
            r = backups.pop(0)
            if try_buy(r["inst"], r["shares"], f"替补（替 {code_of(b['inst'])}）"):
                break


def execute_fb(acc, mk, d, pend, names):
    """FB_dip：开盘按区间买入；收盘卖出 d 之前买入的持仓"""
    pos = acc.s["positions"]
    held_before = [j for j, p in pos.items() if p["buy_date"] < str(d.date())]
    # 开盘买入
    if pend and pend.get("cands"):
        free = pend["K"] - len(pos)
        opts = []
        for c in pend["cands"]:
            j = c["inst"]
            if j in pos:
                continue
            q = mk.quote(d, j)
            if q["susp"] or q["up_open"] or not q["raw_open"] > 0:
                continue
            o = round(q["raw_open"], 2)
            if c["lo"] - 1e-9 <= o <= c["hi"] + 1e-9:
                opts.append((o / c["ref"] - 1, j, o, q))
        for gap, j, o, q in sorted(opts)[:max(free, 0)]:
            px, lot = o * (1 + FB_SLIP), lot_of(j)
            sh = math.floor(min(pend["slot"], acc.s["cash"]) / (1 + FB_BUY_COMM) / (px * lot)) * lot
            while sh > 0 and sh * px + fee(sh * px, FB_BUY_COMM) > acc.s["cash"]:
                sh -= lot
            if sh <= 0:
                continue
            acc.trade(d, "买入", j, sh, px, fee(sh * px, FB_BUY_COMM), f"开盘低开 {gap:+.2%}，落在买入区间", names.get(j, ""))
            pos[j]["factor"] = q["factor"]
    # 收盘卖出
    for j in held_before:
        q = mk.quote(d, j)
        p = pos[j]
        if q["susp"] or q["dn_lim"]:
            acc.trades.append(dict(日期=str(d.date()), 操作="卖出失败", 代码=code_of(j), 名称=names.get(j, ""), 股数=round(p["shares"], 2),
                                   成交价="", 金额="", 费用="", 备注="收盘跌停/停牌，顺延到下一交易日"))
            continue
        px = q["raw_close"] * (1 - FB_SLIP)
        sh = p["shares"]
        acc.trade(d, "卖出", j, sh, px, fee(sh * px, FB_SELL_COMM, stamp=FB_STAMP), "买入次日收盘卖出", names.get(j, ""))


# ============================================================================ 生成清单（调用各策略自己的 live 工具）
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
    name, kind = acc.name, STRATS[acc.name]["kind"]
    hfile = holdings_file(acc, with_date=(kind == "fb"))
    ns = argparse.Namespace(strategy=name, holdings=hfile, cash=float(acc.s["cash"]), etf_value=float(acc.s["etf_value"]),
                            date=str(d.date()), r3_lookback=365, day_index=0, lookback=365)
    buf = io.StringIO()
    r3 = {}
    if kind == "dropout":
        from strategies import live as A
        orig = A.r3_state

        def rec(excess, cfg):
            out = orig(excess, cfg)
            r3.update(expo_next=float(out[0]), expo_today=float(out[1]), dd=float(out[2]))
            return out

        A.r3_state = rec
        try:
            with contextlib.redirect_stdout(buf):
                if STRATS[name]["family"] == "A":
                    A.cmd_signal(ns)
                else:
                    from strategies2 import live as S2
                    S2.cmd_signal(ns)
        finally:
            A.r3_state = orig
        fn = f"strategies/signals/{name}_{d.date()}.csv"
    else:
        from strategies2 import fb_live as FB
        with contextlib.redirect_stdout(buf):
            FB.cmd_signal(argparse.Namespace(holdings=hfile, cash=float(acc.s["cash"]), date=str(d.date())))
        fn = f"strategies2/signals/FB_dip_{d.date()}.csv"
    text = buf.getvalue()
    try:
        out = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        out = pd.DataFrame(columns=["操作", "代码"])
    pend = dict(signal_date=str(d.date()), kind=kind)
    if kind == "dropout":
        pend.update(r3)
        pend["sell"] = [inst_of(c) for c in out.loc[out["操作"] == "卖出", "代码"]]
        pend["buy"] = [dict(inst=inst_of(r.代码), shares=int(r.股数)) for r in out[out["操作"] == "买入"].itertuples()]
        pend["backup"] = [dict(inst=inst_of(r.代码), shares=int(r.股数)) for r in out[out["操作"] == "替补"].itertuples()]
    else:
        from strategies2 import fb_live as FB
        ok = "【可以买】" in text
        c = out[out["操作"].astype(str).str.startswith("候选")] if len(out) else out
        cands = []
        for r in c.itertuples():
            lo, hi = [float(x) for x in str(r.买入区间).split("~")]
            cands.append(dict(inst=inst_of(r.代码), lo=lo, hi=hi, ref=float(r.参考价)))
        pend.update(cands=cands if ok else [], K=FB.K, slot=acc.total() / FB.K)
    # 保存清单副本
    dd = os.path.join(SIG_DIR, str(d.date()))
    os.makedirs(dd, exist_ok=True)
    out.to_csv(os.path.join(dd, f"{name}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(dd, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return pend, out, text


# ============================================================================ 主流程
def strat_cfg(name):
    if STRATS[name]["family"] == "A":
        from strategies.config import STRATEGIES
        return STRATEGIES[name]
    if name == "FB_dip":
        return {}
    from strategies2.config import STRATEGIES
    return STRATEGIES[name]


def process_day(acc, mk, d, names):
    """成交 + 盯市（不含出清单）"""
    kind = STRATS[acc.name]["kind"]
    pend = acc.s.get("pending")
    acc.mark(mk, d)
    acc.s["etf_value"] *= 1 + float(mk.bench.get(d, 0.0))     # 昨日持有的 ETF 吃当日收益，再按收盘成交
    if kind == "dropout" and pend:
        execute_dropout(acc, mk, d, pend, names, strat_cfg(acc.name))
    elif kind == "fb":
        execute_fb(acc, mk, d, pend, names)
    acc.mark(mk, d)


def rebalance(C, F, d):
    """收盘后只用现金把机会仓调回目标比例；返回划给机会仓的金额（负数 = 划给主仓）"""
    total = C.total() + F.total()
    gap = total * FB_RATIO - F.total()
    amt = min(gap, max(C.s["cash"], 0.0)) if gap > 0 else -min(-gap, max(F.s["cash"], 0.0))
    if abs(amt) < 1.0:
        return 0.0
    C.s["cash"] -= amt
    F.s["cash"] += amt
    F.s["transfer_in"] = F.s.get("transfer_in", 0.0) + amt
    C.s["transfer_in"] = C.s.get("transfer_in", 0.0) - amt
    return amt


def sleeve_nav(acc, mk, d, prev_total, flow, n0):
    tot = acc.total()
    b = float(mk.bench.get(d, 0.0))
    acc.s["bench_cum"] = acc.s.get("bench_cum", 1.0) * (1 + b)
    acc.s["unit"] = acc.s.get("unit", 1.0) * ((tot - flow) / prev_total if prev_total > 0 else 1.0)   # 剔除资金划转的单位净值
    acc.navs.append(dict(日期=str(d.date()), 现金=round(acc.s["cash"], 2), 股票市值=round(acc.stock_value(), 2),
                         ETF市值=round(acc.s["etf_value"], 2), 总资产=round(tot, 2), 资金划入=round(flow, 2),
                         当日收益=round((tot - flow) / prev_total - 1 if prev_total > 0 else 0.0, 6),
                         单位净值=round(acc.s["unit"], 6), 中证1000累计=round(acc.s["bench_cum"] - 1, 6),
                         持仓数=len(acc.s["positions"]),
                         成交笔数=len([t for t in acc.trades[n0:] if t["操作"] in ("买入", "卖出")])))


COMBINED = os.path.join(ACC_DIR, "combined_nav.csv")


def combined_row(C, F, mk, d, prev_total, bench_cum):
    tot = C.total() + F.total()
    row = dict(日期=str(d.date()), 总资产=round(tot, 2), 主仓C_top50=round(C.total(), 2), 机会仓FB_dip=round(F.total(), 2),
               机会仓占比=round(F.total() / tot, 4), 当日收益=round(tot / prev_total - 1, 6), 累计收益=round(tot / INIT_CASH - 1, 6),
               中证1000累计=round(bench_cum - 1, 6), 超额累计=round(tot / INIT_CASH - bench_cum, 6))
    df = pd.DataFrame([row])
    if os.path.exists(COMBINED):
        old = pd.read_csv(COMBINED, encoding="utf-8-sig")
        df = pd.concat([old[old["日期"] != row["日期"]], df], ignore_index=True)
    os.makedirs(ACC_DIR, exist_ok=True)
    df.to_csv(COMBINED, index=False, encoding="utf-8-sig")
    return row


def run(start=None, until=None):
    mk = Market()
    names = names_table()
    data_last = mk.bench.index.max()
    days = [d for d in trading_days() if d <= data_last]
    if until:
        days = [d for d in days if d <= pd.Timestamp(until)]
    C, F = Account("C_top50"), Account("FB_dip")
    if not (C.exists and F.exists):
        s = pd.Timestamp(start) if start else days[-1]
        s = max(d for d in days if d <= s)
        C.init(s, INIT_CASH * (1 - FB_RATIO))
        F.init(s, INIT_CASH * FB_RATIO)
        for a in (C, F):
            a.s["bench_cum"], a.s["unit"] = 1.0, 1.0
            a.navs.append(dict(日期=str(s.date()), 现金=round(a.s["cash"], 2), 股票市值=0.0, ETF市值=0.0, 总资产=round(a.s["cash"], 2),
                               资金划入=0.0, 当日收益=0.0, 单位净值=1.0, 中证1000累计=0.0, 持仓数=0, 成交笔数=0))
        print(f"新建组合账户：{INIT_CASH:,.0f} = 主仓 C_top50 {C.s['cash']:,.0f} + 机会仓 FB_dip {F.s['cash']:,.0f}，起始信号日 {s.date()}",
              flush=True)
        for a in (C, F):
            pend, _, text = gen_signal(a, s)
            a.s["pending"], a.s["last_date"] = pend, str(s.date())
            a.save(names)
            print(text, flush=True)
        if os.path.exists(COMBINED):
            os.remove(COMBINED)
        combined_row(C, F, mk, s, INIT_CASH, 1.0)
        todo = [d for d in days if d > s]
    else:
        assert C.s["last_date"] == F.s["last_date"], "两个仓位的处理日期不一致"
        todo = [d for d in days if d > pd.Timestamp(C.s["last_date"])]
    if not todo:
        print(f"组合账户已是最新（{C.s['last_date']}），数据最新交易日 {data_last.date()}", flush=True)
        return
    for d in todo:
        prev = {a.name: a.total() for a in (C, F)}
        n0 = {a.name: len(a.trades) for a in (C, F)}
        for a in (C, F):
            process_day(a, mk, d, names)
        amt = rebalance(C, F, d)
        sleeve_nav(C, mk, d, prev["C_top50"], -amt, n0["C_top50"])
        sleeve_nav(F, mk, d, prev["FB_dip"], amt, n0["FB_dip"])
        row = combined_row(C, F, mk, d, prev["C_top50"] + prev["FB_dip"], C.s["bench_cum"])
        print(f"{d.date()}  总资产 {row['总资产']:,.0f}（当日 {row['当日收益']:+.2%}，累计 {row['累计收益']:+.2%}）｜"
              f"主仓 {row['主仓C_top50']:,.0f} 成交 {len(C.trades) - n0['C_top50']} 笔｜机会仓 {row['机会仓FB_dip']:,.0f} 成交 "
              f"{len(F.trades) - n0['FB_dip']} 笔｜划转 {amt:+,.0f}", flush=True)
        texts = []
        for a in (C, F):
            pend, _, text = gen_signal(a, d)
            a.s["pending"], a.s["last_date"] = pend, str(d.date())
            a.save(names)
            texts.append(text)
    print("\n".join(texts), flush=True)


def summary():
    C, F = Account("C_top50"), Account("FB_dip")
    if not (C.exists and F.exists) or not os.path.exists(COMBINED):
        return
    nav = pd.read_csv(COMBINED, encoding="utf-8-sig")
    last = nav.iloc[-1]
    v = nav["总资产"]
    mdd = float((v / v.cummax() - 1).min())
    rows = [f"| **组合** | {C.s['start_date']} | {last['日期']} | {last['总资产']:,.0f} | {last['当日收益']:+.2%} | {last['累计收益']:+.2%} | "
            f"{last['中证1000累计']:+.2%} | {last['超额累计']:+.2%} | {mdd:.2%} | {len(C.s['positions']) + len(F.s['positions'])} | "
            f"{C.s['cash'] + F.s['cash']:,.0f} |"]
    for a in (C, F):
        n = pd.read_csv(os.path.join(a.dir, "nav.csv"), encoding="utf-8-sig")
        u = n["单位净值"]
        m = float((u / u.cummax() - 1).min())
        l = n.iloc[-1]
        rows.append(f"| {a.name}（{STRATS[a.name]['desc'].split('：')[0]}） | {a.s['start_date']} | {l['日期']} | {l['总资产']:,.0f} | "
                    f"{l['当日收益']:+.2%} | {l['单位净值'] - 1:+.2%} | {l['中证1000累计']:+.2%} | {l['单位净值'] - 1 - l['中证1000累计']:+.2%} | "
                    f"{m:.2%} | {len(a.s['positions'])} | {a.s['cash']:,.0f} |")
    orders = []
    for a in (C, F):
        p = a.s.get("pending") or {}
        sd = p.get("signal_date", "")
        fn = os.path.join(SIG_DIR, sd, f"{a.name}.csv")
        if not os.path.exists(fn):
            continue
        try:
            o = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            o = pd.DataFrame(columns=["操作"])
        if len(o):
            bk = o[o["操作"] == "替补"].head(5)
            o = pd.concat([o[~o["操作"].isin(["替补", "继续持有"])], bk]) if (o["操作"] == "买入").any() else \
                o[~o["操作"].isin(["替补", "继续持有"])]
        extra = ""
        if p.get("kind") == "dropout" and p.get("expo_next", 1) < 1:
            extra = f"（R3 半仓：股票仓位 {p['expo_next']:.0%}）"
        if p.get("kind") == "fb" and not p.get("cands"):
            extra = "（市场条件不满足或没有候选，明天不买）" if len(o) == 0 else ""
        lines = [f"\n### {a.name}｜信号日 {sd}{extra}\n"]
        if len(o):
            cols = [c for c in ["操作", "代码", "名称", "股数", "参考价", "买入区间", "参考金额", "备注"] if c in o.columns]
            o = o[cols].fillna("")
            lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in o.itertuples(index=False)]
        else:
            lines.append("无操作")
        orders.append("\n".join(lines))
    md = ["# 组合模拟盘：C_top50 主仓 + FB_dip 机会仓（10 万元）", "",
          f"一个账户分两仓：主仓 C_top50 {1 - FB_RATIO:.0%}、机会仓 FB_dip {FB_RATIO:.0%}（strategies2/README.md 第 11 节的做法），"
          "每天收盘后用现金调回目标比例。",
          "GitHub Actions 每个交易日北京时间 17:00 自动运行：成交上一交易日的清单 → 盯市 → 再平衡 → 生成下一交易日清单。"
          "数据当天没发布时，之后的运行会按真实成交日价格自动补齐。", "",
          "## 账户", "",
          "| 账户 | 起始日 | 最新日 | 总资产 | 当日 | 累计 | 中证1000累计 | 超额 | 最大回撤 | 持仓 | 现金 |",
          "|---|---|---|---|---|---|---|---|---|---|---|", *rows, "",
          "分仓的累计收益按单位净值计算（剔除两仓之间的资金划转）。明细：`paper/accounts/combined_nav.csv`（组合每日净值）、"
          "`paper/accounts/<仓位>/`（nav / trades / positions）；每日完整清单在 `paper/signals/<日期>/`。", "",
          "## 下一交易日操作清单", *orders, "",
          "> 仅供学习研究，不构成投资建议。模拟盘按收盘/开盘价成交，未计冲击成本，实盘会有差距。"]
    open(os.path.join(PAPER, "README.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", help="新建账户的起始信号日（默认最新交易日）")
    r.add_argument("--until", help="只处理到这一天（测试用）")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.start, a.until)
    summary()


if __name__ == "__main__":
    main()
