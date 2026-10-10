"""Shared cash/share execution for historical replay and daily paper accounts.

Orders are decided on the previous signal day. Only execution prices, transaction
eligibility and cash/slots are checked on the execution day. ETF values remain an
explicit index-return proxy; corporate actions remain a total-return share proxy.
"""
import math

ETF_CODE = "512100"
ADJ_TOL = 0.003
EXECUTION_VERSION = 2


def code_of(inst):
    return inst[2:] + "." + inst[:2]


def lot_of(inst):
    return 200 if inst.startswith("SH688") else 100


def fee(amount, rate, min_cost=5.0, stamp=0.0):
    return max(amount * rate, min_cost) + amount * stamp if amount > 0 else 0.0


def stamp_rate(d):
    return 0.001 if str(d.date()) < "2023-08-28" else 0.0005


def sell_fee(amount, cfg, d):
    # Config close_cost includes the current 0.05% stamp duty. Minimum applies
    # to the commission component, never to commission + stamp duty together.
    return fee(amount, max(cfg["close_cost"] - 0.0005, 0), cfg["min_cost"], stamp_rate(d))


def affordable(inst, budget, px, rate, min_cost):
    if not (math.isfinite(px) and px > 0 and budget > 0):
        return 0
    step = 1 if inst.startswith("SH688") else 100
    sh = math.floor(budget / (1 + rate) / px / step) * step
    while sh > 0 and sh * px + fee(sh * px, rate, min_cost) > budget + 1e-8:
        sh -= step
    return sh if sh >= lot_of(inst) else 0


class Ledger:
    def __init__(self, cash=0.0, state=None):
        self.s = state if state is not None else dict(cash=float(cash), etf_value=0.0, positions={})
        self.trades = []

    def total(self):
        return self.s["cash"] + self.s["etf_value"] + self.stock_value()

    def stock_value(self):
        return sum(p["shares"] * p.get("last_px", 0.0) for p in self.s["positions"].values())

    def mark(self, mk, d, at="close"):
        for j, p in self.s["positions"].items():
            q = mk.quote(d, j)
            f0 = p.get("factor") or 0
            f = q.get("factor", float("nan"))
            if f > 0 and f0 > 0 and abs(f / f0 - 1) > ADJ_TOL:
                p["shares"] = round(p["shares"] * f / f0, 6)
                p["factor"] = f
            elif f > 0 and f0 <= 0:
                p["factor"] = f
            px = q.get("raw_" + at, float("nan"))
            if px > 0:
                p["last_px"] = px
        return self.stock_value()

    def trade(self, d, action, inst, shares, px, f, note="", name=""):
        if not (shares > 0 and px > 0 and math.isfinite(px) and f >= 0):
            raise ValueError("Invalid trade")
        amt = shares * px
        if action == "买入":
            if amt + f > self.s["cash"] + 1e-7:
                raise ValueError("Buy exceeds available cash")
            self.s["cash"] = max(0.0, self.s["cash"] - amt - f)
            p = self.s["positions"].get(inst)
            if p:
                p["cost"] += amt + f
                p["shares"] += shares
            else:
                self.s["positions"][inst] = dict(shares=float(shares), cost=amt + f, buy_date=str(d.date()),
                                                 last_px=px, factor=None)
        else:
            p = self.s["positions"][inst]
            if p.get("buy_date", "") >= str(d.date()):
                raise ValueError("T+1: same-day stock sale is forbidden")
            if shares > p["shares"] + 1e-7:
                raise ValueError("Sell exceeds holding")
            self.s["cash"] += amt - f
            if shares >= p["shares"] - 1e-7:
                del self.s["positions"][inst]
            else:
                p["cost"] *= 1 - shares / p["shares"]
                p["shares"] -= shares
        self.trades.append(dict(日期=str(d.date()), 操作=action, 代码=code_of(inst), 名称=name, 股数=shares,
                                成交价=round(px, 4), 金额=round(amt, 2), 费用=round(f, 2), 备注=note))


def rejected(acc, d, action, j, reason, names):
    acc.trades.append(dict(日期=str(d.date()), 操作=action + "失败", 代码=code_of(j), 名称=names.get(j, ""),
                           股数=0, 成交价="", 金额="", 费用="", 备注=reason))


def allowed(q, side, at):
    px = q.get("raw_" + at, float("nan"))
    limit = ("up_" if side == "buy" else "dn_") + ("open" if at == "open" else "lim")
    return (math.isfinite(px) and px > 0 and not q.get("susp", True) and not q.get(limit, True)
            and (side != "buy" or q.get("buy_ok", False)))


def concentrated_orders(score, hold, k, keep, on=True, eligible=None):
    s = score.dropna().sort_values(ascending=False, kind="stable")
    rank = {j: i + 1 for i, j in enumerate(s.index)}
    sell = [j for j in hold if not on or rank.get(j, float("inf")) > keep]
    n = max(k - len([j for j in hold if j not in sell]), 0) if on else 0
    cand = [j for j in s.index if j not in hold and (eligible is None or eligible.get(j, False))]
    return dict(sell=sell, buy=[dict(inst=j) for j in cand[:n]],
                backup=[dict(inst=j) for j in cand[n:n + 5]])


def execute_conc(acc, mk, d, pend, names, cfg):
    at = cfg.get("exec_at", "open")
    oc, mc, slip = cfg["open_cost"], cfg["min_cost"], cfg.get("slip", 0.001)
    pos = acc.s["positions"]
    for j in pend["sell"]:
        if j not in pos:
            continue
        q = mk.quote(d, j)
        if not allowed(q, "sell", at):
            rejected(acc, d, "卖出", j, "跌停/停牌/缺价，继续持有", names)
            continue
        px, sh = q["raw_" + at] * (1 - slip), pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, px, sell_fee(sh * px, cfg, d), "清单卖出", names.get(j, ""))
    free = max(cfg["k"] - len(pos), 0)  # failed sales still occupy a slot
    backup = list(pend["backup"])
    slot = acc.total() / cfg["k"]  # marked to execution time, not today's close
    for primary in pend["buy"][:free]:
        queue = [primary]
        while queue:
            j = queue.pop(0)["inst"]
            q = mk.quote(d, j)
            px = q.get("raw_" + at, float("nan")) * (1 + slip)
            sh = affordable(j, min(slot, acc.s["cash"]), px, oc, mc) if allowed(q, "buy", at) else 0
            if j not in pos and sh > 0:
                acc.trade(d, "买入", j, sh, px, fee(sh * px, oc, mc), "清单/替补买入", names.get(j, ""))
                pos[j]["factor"] = q["factor"]
                break
            rejected(acc, d, "买入", j, "不符合交易状态/封板/资金不足", names)
            if backup:
                queue.append(backup.pop(0))


