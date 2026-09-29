"""
实盘工具：训练生产模型 + 每日生成调仓清单

  # 1) 训练生产模型（建议每年重训一次；两个模型约 5 分钟）
  python strategies/live.py train                 # 训练 config.MODELS 里的全部模型
  python strategies/live.py train --model rb      # 只训练其中一个

  # 2) 每个交易日收盘后（先 bash setup_env.sh --refresh 更新数据）
  python strategies/live.py signal --strategy A_top50 --holdings my_holdings.csv --cash 23000
  python strategies/live.py signal --strategy A_top20 --holdings my_top20.csv --cash 5000 --etf_value 0
  python strategies/live.py signal --strategy AE_top20 --holdings my_ae.csv --cash 8000      # 两模型集成

holdings.csv 两列：code,shares   （code 可写 600000 / 600000.SH / SH600000；空仓时可不传 --holdings）
输出 strategies/signals/{策略}_{日期}.csv，并在屏幕打印：
  - 卖出 / 买入清单（股数已按一手取整）、参考价、行业
  - 替补名单：买入目标在成交日涨停或停牌买不进时，按顺序用替补
  - R3 风控状态：模拟盘超额回撤、明日应有的股票仓位比例、需要切换时的具体操作
成交：信号日 T 收盘后生成，T+1 尾盘集合竞价（14:57-15:00）成交；跌停卖不出的就继续持有，第二天再卖
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.common import DATA, EXTRA, KEY, read_parts  # noqa: E402
from strategies.config import MODELS, STRATEGIES  # noqa: E402


def model_path(name):
    f = os.path.join(ROOT, MODELS[name]["file"])
    return f, f.replace(".txt", ".json")


# ----------------------------------------------------------------------------------------------
def last_trading_day():
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet"))
    return b["datetime"].max()


def industry_neutral(pred):
    """打分 → 当日百分位排名 → 减去行业均值（与研究中的 experiments/neutralize.py --mode ind 相同）"""
    ind = pd.read_parquet(os.path.join(EXTRA, "industry.parquet"))[["instrument", "industry"]]
    df = pred.merge(ind, on="instrument", how="left")
    df["industry"] = df["industry"].replace("", np.nan).fillna("UNK")
    df["s"] = df.groupby("datetime")["score"].rank(pct=True) - 0.5
    df["s"] = df["s"] - df.groupby(["datetime", "industry"])["s"].transform("mean")
    return df[KEY + ["s"]].rename(columns={"s": "score"})


def rank_blend(preds):
    """多个模型：每个模型的（行业中性）打分先转成当日百分位排名，再等权平均；某模型缺分时用其余模型"""
    if len(preds) == 1:
        return preds[0]
    ps, ws = [], []
    for d in preds:
        r = d.set_index(KEY)["score"].groupby(level=0).rank(pct=True)
        ps.append(r)
        ws.append(r.notna().astype(float))
    num = pd.concat(ps, axis=1).sum(axis=1)
    den = pd.concat(ws, axis=1).sum(axis=1)
    return (num / den.replace(0, np.nan)).rename("score").dropna().reset_index()


def norm_code(c):
    c = str(c).strip().upper().replace(".", "")
    if c[:2] in ("SH", "SZ", "BJ"):
        return c[:8]
    if c[-2:] in ("SH", "SZ", "BJ"):
        return c[-2:] + c[:-2]
    c = c.zfill(6)
    return ("SH" if c[0] in "69" else "BJ" if c[0] in "48" else "SZ") + c


def lot_of(inst):
    return 200 if inst.startswith("SH688") else 100


# ----------------------------------------------------------------------------------------------
def cmd_train(a):
    from engine import model as M
    T = last_trading_day()
    v_end = pd.Timestamp(a.valid_end) if a.valid_end else T - pd.Timedelta(days=15)      # 标签要用未来 2 天
    v_start = pd.Timestamp(a.valid_start) if a.valid_start else v_end - pd.DateOffset(years=2)
    t_end = pd.Timestamp(a.train_end) if a.train_end else v_start - pd.Timedelta(days=15)
    t_start = pd.Timestamp(a.train_start)
    names = list(MODELS) if a.model == "all" else [a.model]
    for name in names:
        sets = MODELS[name]["sets"]
        path, meta_path = model_path(name)
        print(f"[{name}] 特征 {'+'.join(sets)}｜训练 {t_start.date()} ~ {t_end.date()}，验证（早停）{v_start.date()} ~ {v_end.date()}",
              flush=True)
        m, cols, imp, it = M.fit(lambda: M.load(sets, t_start, t_end, 1), lambda: M.load(sets, v_start, v_end, 1),
                                 "lgb", "robust", a.threads, a.seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m.save_model(path, num_iteration=it)
        meta = dict(sets=sets, features=cols, best_iter=int(it), train=[str(t_start.date()), str(t_end.date())],
                    valid=[str(v_start.date()), str(v_end.date())], data_last_day=str(T.date()), seed=a.seed,
                    top_features=list(imp.index[:8]))
        json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
        print(f"[{name}] 已保存 {path}  best_iter={it}  重要因子：{', '.join(imp.index[:8])}", flush=True)


def predict(start, end, names=("retail",)):
    """返回（多模型集成后的）行业中性打分，以及各模型的 meta"""
    import lightgbm as lgb
    from engine import model as M
    preds, metas = [], {}
    for name in names:
        path, meta_path = model_path(name)
        if not os.path.exists(path):
            sys.exit(f"找不到模型 {path}，请先运行：python strategies/live.py train --model {name}")
        meta = json.load(open(meta_path))
        m = lgb.Booster(model_file=path)
        m.best_iteration = meta["best_iter"]
        sets = meta.get("sets", MODELS[name]["sets"])
        raw = M.predict_range(m, meta["features"], sets, start, end, 1, "lgb")
        preds.append(industry_neutral(raw))
        metas[name] = meta
    return rank_blend(preds), metas


def r3_state(excess, cfg):
    """按 T 日及以前的模拟盘超额，给出 T+1 的股票仓位比例（与回测中的 R3 完全一致）"""
    cum = (1 + excess).cumprod()
    dd = cum / cum.cummax() - 1
    st, hist = 1.0, []
    for v in dd.values:            # v 是当日收盘后的回撤 → 决定下一日状态
        hist.append(st)
        if st == 1.0 and v < cfg["r3_in"]:
            st = cfg["r3_exposure"]
        elif st < 1.0 and v > cfg["r3_out"]:
            st = 1.0
    return st, hist[-1], dd.iloc[-1]


def dropout_orders(score, hold, topk, n_drop):
    """与 engine/backtest.py 的 dropout 规则相同：卖出排名最差的 n_drop 只，补入未持有的高分股"""
    s = score.dropna().sort_values(ascending=False)
    sc = lambda j: s.get(j, -np.inf)  # noqa: E731
    # 没有打分的持仓（不在中证1000 / 已调出 / 手动买入的其他股票）直接全部卖出，不占 n_drop 名额
    extra = [j for j in hold if j not in s.index]
    hold = [j for j in hold if j in s.index]
    last = sorted(hold, key=sc, reverse=True)
    n_add = n_drop + topk - len(last)
    cand = [j for j in s.index if j not in set(last)]
    today = cand[:max(n_add, 0)]
    comb = sorted(last + today, key=sc, reverse=True)
    worst = set(comb[-n_drop:]) if n_drop > 0 else set()
    sell = [j for j in last if j in worst]
    buy = today[:len(sell) + topk - len(last)]
    sell = extra + sell
    backup = [j for j in cand if j not in set(buy)][:max(5, 2 * len(buy))]
    return sell, buy, backup, s


def cmd_signal(a):
    from engine.backtest import Backtester
    cfg = STRATEGIES[a.strategy]
    T = pd.Timestamp(a.date) if a.date else last_trading_day()
    start = T - pd.Timedelta(days=a.r3_lookback)
    pred, metas = predict(start - pd.Timedelta(days=40), T, cfg["models"])
    if pred["datetime"].max() < T:
        sys.exit(f"{T.date()} 没有因子数据，请先运行 bash setup_env.sh --refresh")

    # ---- R3：用同一模型的模拟盘（从 T-lookback 开始）计算策略超额回撤
    bt = Backtester(start, T)
    paper = bt.run(pred, mode="dropout", topk=cfg["topk"], n_drop=cfg["n_drop"], every=cfg["every"],
                   open_cost=cfg["open_cost"], close_cost=cfg["close_cost"], account=cfg["account"],
                   min_cost=cfg["min_cost"], risk_degree=cfg["risk_degree"])
    expo_next, expo_today, dd = r3_state(paper["excess"], cfg)

    # ---- 当前持仓与价格
    px = read_parts(os.path.join(DATA, "panel"), columns=["datetime", "instrument", "raw_close", "up_lim", "dn_lim", "susp"],
                    start=T, end=T).set_index("instrument")
    names = pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")
    if a.holdings:
        h = pd.read_csv(a.holdings, dtype={"code": str})
        h["instrument"] = h["code"].map(norm_code)
        hold = h.groupby("instrument")["shares"].sum()
    else:
        hold = pd.Series(dtype=float)
    hold = hold[hold > 0]
    miss = [j for j in hold.index if j not in px.index]
    if miss:   # 非中证1000成分：直接从 Qlib 全市场数据取收盘价
        from engine.common import init_qlib
        from qlib.data import D
        init_qlib()
        q = D.features(miss, ["$close/$factor"], T, T)
        q = q.reset_index().set_index("instrument").iloc[:, -1]
        for j in miss:
            px.loc[j, "raw_close"] = float(q.get(j, np.nan))
            px.loc[j, "dn_lim"] = False
        bad = [j for j in miss if not px.loc[j, "raw_close"] > 0]
        if bad:
            print("⚠ 以下持仓查不到价格（代码有误或已退市），按 0 估值：", bad)
    val = pd.Series({j: hold[j] * np.nan_to_num(px["raw_close"].get(j, 0.0)) for j in hold.index}, dtype=float)
    stock_value = float(val.sum())
    total = stock_value + a.cash + a.etf_value

    score = pred[pred["datetime"] == T].set_index("instrument")["score"]
    sell, buy, backup, s = dropout_orders(score, list(hold.index), cfg["topk"], cfg["n_drop"])
    if cfg["every"] > 1 and a.day_index % cfg["every"] != 0:
        sell, buy = [], []

    # ---- 资金分配（与回测一致：可用现金 × risk_degree 均分，单票不超过 1.5 倍等权）
    sleeve = total * expo_next if expo_next < 1 else total - a.etf_value
    cash_after = a.cash + sum(val.get(j, 0.0) * (1 - cfg["close_cost"]) for j in sell)
    budget = max(cash_after - (1 - cfg["risk_degree"]) * sleeve, 0)
    per = min(budget / max(len(buy), 1), sleeve * cfg["risk_degree"] / cfg["topk"] * 1.5)
    rank = pd.Series(np.arange(1, len(s) + 1), index=s.index)

    def row(action, j, shares=None):
        p = float(px["raw_close"].get(j, np.nan))
        if shares is None:
            lot = lot_of(j)
            shares = int(np.floor(per / (1 + cfg["open_cost"]) / (p * lot)) * lot) if p > 0 else 0
        return dict(操作=action, 代码=j[2:] + "." + j[:2], 名称=names["code_name"].get(j, ""),
                    行业=names["industry"].get(j, ""), 排名=int(rank.get(j, -1)), 股数=int(shares),
                    参考价=round(p, 2), 参考金额=round(shares * p if p > 0 else 0, 0),
                    备注=("今日跌停，可能卖不出" if action == "卖出" and bool(px["dn_lim"].get(j, False)) else
                        "非策略股票，整笔卖出" if action == "卖出" and j not in s.index else
                        "资金不足一手，用替补" if action == "买入" and shares == 0 else ""))

    orders = [row("卖出", j, hold[j]) for j in sell] + [row("买入", j) for j in buy] + \
             [row("替补", j) for j in backup]
    out = pd.DataFrame(orders)
    os.makedirs("strategies/signals", exist_ok=True)
    fn = f"strategies/signals/{a.strategy}_{T.date()}.csv"
    out.to_csv(fn, index=False, encoding="utf-8-sig")

    # ---- 打印
    pd.set_option("display.width", 200); pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n==== {a.strategy}｜信号日 {T.date()}｜在下一交易日尾盘集合竞价成交 ====")
    for name, meta in metas.items():
        print(f"模型 {name}：{'+'.join(meta.get('sets', MODELS[name]['sets']))}，训练 {meta['train'][0]}~{meta['train'][1]}，"
              f"best_iter={meta['best_iter']}" + ("（多模型：排名等权平均）" if len(metas) > 1 else ""))
    print(f"账户：股票 {stock_value:,.0f} + 现金 {a.cash:,.0f} + ETF {a.etf_value:,.0f} = {total:,.0f}；"
          f"持仓 {len(hold)} 只 / 目标 {cfg['topk']} 只")
    print(f"R3：模拟盘超额回撤 {dd:+.1%}（阈值 {cfg['r3_in']:.0%} 进 / {cfg['r3_out']:.0%} 出），"
          f"明日股票仓位 {expo_next:.0%}")
    etf_target = total * (1 - expo_next)
    if expo_next < 1:
        tag = "进入" if expo_today >= 1 else "维持"
        print(f"  ▶ {tag} R3 半仓：股票约 {total * expo_next:,.0f} 元，中证1000ETF（{cfg['hedge_etf']}）约 {etf_target:,.0f} 元"
              f"（当前 {a.etf_value:,.0f}）。" + ("ETF 不足时，把每只持仓卖出约一半（按手取整）去买 ETF；" if a.etf_value < 0.9 * etf_target else "")
              + "下方买入清单已按半仓计算")
    elif a.etf_value > 0:
        print(f"  ▶ 解除 R3：卖出全部中证1000ETF（{a.etf_value:,.0f} 元），资金并入股票账户，之后的信号会逐步补足持仓")
    print(f"模拟盘近 {a.r3_lookback} 天：超额 {paper['excess'].sum():+.1%}，日均换手 {paper['turnover'].mean():.1%}\n")
    main = out[out["操作"] != "替补"]
    print(main.to_string(index=False) if len(main) else "今日无需交易")
    if len(buy):
        print("\n替补（买入目标涨停/停牌买不进时按顺序替换）：",
              "、".join(f"{r.代码} {r.名称}" for r in out[out["操作"] == "替补"].head(10).itertuples()))
    print(f"\n已保存 {fn}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="训练生产模型")
    t.add_argument("--model", choices=["all"] + list(MODELS), default="all")
    t.add_argument("--train_start", default="2015-01-01")
    t.add_argument("--train_end")
    t.add_argument("--valid_start")
    t.add_argument("--valid_end")
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("signal", help="生成调仓清单")
    s.add_argument("--strategy", choices=list(STRATEGIES), default="A_top50")
    s.add_argument("--holdings", help="当前持仓 csv：code,shares")
    s.add_argument("--cash", type=float, default=None, help="可用现金（默认等于策略配置中的初始资金，用于首次建仓）")
    s.add_argument("--etf_value", type=float, default=0.0, help="当前持有的中证1000ETF市值（R3 用）")
    s.add_argument("--date", help="信号日（默认最新交易日）")
    s.add_argument("--r3_lookback", type=int, default=365, help="R3 模拟盘回看天数（自然日）")
    s.add_argument("--day_index", type=int, default=0, help="every>1 时的交易日计数")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    else:
        if a.cash is None:
            a.cash = 0.0 if a.holdings else float(STRATEGIES[a.strategy]["account"])
        cmd_signal(a)


if __name__ == "__main__":
    main()
