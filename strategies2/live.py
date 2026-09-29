"""
第二批策略（C_top50 分散版 / K_top3 集中版）：训练生产模型 + 每日生成调仓清单

  # 0) 第一次使用：抓股东户数并生成因子（约 15 分钟）
  bash strategies2/setup_data.sh

  # 1) 训练生产模型（建议每年重训一次）
  python strategies2/live.py train --strategy C_top50      # 训练 rbh（retail / rb 与 A 系列共用，已存在则跳过）约 3 分钟
  python strategies2/live.py train --strategy K_top3       # 六个 5 日模型，约 10 分钟

  # 2) 每个交易日收盘后（先 bash setup_env.sh --refresh && bash strategies2/setup_data.sh --refresh）
  python strategies2/live.py signal --strategy C_top50 --holdings my_c.csv --cash 20000     # T+1 尾盘成交
  python strategies2/live.py signal --strategy K_top3  --holdings my_k.csv --cash 5000      # T+1 开盘成交

holdings.csv 两列：code,shares（空仓时可不传 --holdings）。清单保存在 strategies2/signals/（已加入 .gitignore）
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
from engine.common import DATA, EXTRA, read_parts  # noqa: E402
from strategies import live as A  # noqa: E402   复用 A 系列已验证的工具函数（不修改 A 系列）
from strategies2.config import MODELS, STRATEGIES  # noqa: E402

last_trading_day, industry_neutral, rank_blend, norm_code, lot_of = (
    A.last_trading_day, A.industry_neutral, A.rank_blend, A.norm_code, A.lot_of)


def model_path(name):
    f = os.path.join(ROOT, MODELS[name]["file"])
    return f, f.replace(".txt", ".json")


# ----------------------------------------------------------------------------------------------
def cmd_train(a):
    from engine import model as M
    T = last_trading_day()
    v_end = pd.Timestamp(a.valid_end) if a.valid_end else T - pd.Timedelta(days=15)      # 标签要用未来数据，留出间隔
    v_start = v_end - pd.DateOffset(years=2)
    t_end = v_start - pd.Timedelta(days=15)
    t_start = pd.Timestamp(a.train_start)
    names = STRATEGIES[a.strategy]["models"] if a.strategy else [a.model]
    for name in names:
        spec = MODELS[name]
        path, meta_path = model_path(name)
        if os.path.exists(path) and not a.force and name in ("retail", "rb"):
            print(f"[{name}] 已存在（与 A 系列共用），跳过；要重训请加 --force 或用 strategies/live.py train", flush=True)
            continue
        sets, h = spec["sets"], spec["horizon"]
        print(f"[{name}] 特征 {'+'.join(sets)}｜持有 {h} 天｜种子 {spec['seed']}｜训练 {t_start.date()}~{t_end.date()}，"
              f"早停 {v_start.date()}~{v_end.date()}", flush=True)
        m, cols, imp, it = M.fit(lambda: M.load(sets, t_start, t_end, h), lambda: M.load(sets, v_start, v_end, h),
                                 "lgb", "robust", a.threads, spec["seed"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m.save_model(path, num_iteration=it)
        meta = dict(sets=sets, horizon=h, seed=spec["seed"], features=cols, best_iter=int(it),
                    train=[str(t_start.date()), str(t_end.date())], valid=[str(v_start.date()), str(v_end.date())],
                    data_last_day=str(T.date()), top_features=list(imp.index[:8]))
        json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
        print(f"[{name}] 已保存 {path}  best_iter={it}  重要因子：{', '.join(imp.index[:8])}", flush=True)


def predict(start, end, names, neutral=True):
    """各模型打分（neutral=True 先行业中性），再按当日排名等权平均"""
    import lightgbm as lgb
    from engine import model as M
    preds, metas = [], {}
    for name in names:
        path, meta_path = model_path(name)
        if not os.path.exists(path):
            sys.exit(f"找不到模型 {path}，请先运行：python strategies2/live.py train --model {name}")
        meta = json.load(open(meta_path))
        m = lgb.Booster(model_file=path)
        m.best_iteration = meta["best_iter"]
        sets = meta.get("sets", MODELS[name]["sets"])
        raw = M.predict_range(m, meta["features"], sets, start, end, meta.get("horizon", 1), "lgb")
        preds.append(industry_neutral(raw) if neutral else raw)
        metas[name] = meta
    return rank_blend(preds), metas


# ----------------------------------------------------------------------------------------------
def signal_dropout(a, cfg):
    """C_top50：与 A 系列完全相同的下单逻辑（复用 strategies/live.py 的 cmd_signal），只是换成三模型集成"""
    A.STRATEGIES = dict(A.STRATEGIES, **{a.strategy: cfg})
    A.MODELS = dict(A.MODELS, **{k: MODELS[k] for k in cfg["models"]})
    A.predict = lambda s, e, names: predict(s, e, names, neutral=True)
    a.r3_lookback = a.lookback
    a.day_index = 0
    A.cmd_signal(a)


def vote_state(T, mas, need):
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime").iloc[:, 0]
    lvl = (1 + b[b.index <= T]).cumprod()
    above = {m: bool(lvl.iloc[-1] > lvl.iloc[-m:].mean()) for m in mas}
    return sum(above.values()) >= need, above


def signal_concentrated(a, cfg):
    """K_top3：持仓排名跌出前 keep_rank 或择时转空 → 卖；空位按排名补足；T+1 开盘集合竞价成交"""
    T = pd.Timestamp(a.date) if a.date else last_trading_day()
    k, keep = cfg["k"], cfg["keep_rank"]
    start = T - pd.Timedelta(days=a.lookback) if a.lookback > 0 else T - pd.Timedelta(days=10)
    pred, metas = predict(start, T, cfg["models"], neutral=False)
    if pred["datetime"].max() < T:
        sys.exit(f"{T.date()} 没有因子数据，请先运行 bash setup_env.sh --refresh && bash strategies2/setup_data.sh --refresh")
    on, above = vote_state(T, cfg["timing_mas"], cfg["timing_need"])

    px = read_parts(os.path.join(DATA, "panel"), columns=["datetime", "instrument", "raw_close", "up_lim", "dn_lim", "susp",
                                                          "in_pool"], start=T, end=T).set_index("instrument")
    names = pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")
    if a.holdings:
        h = pd.read_csv(a.holdings, dtype={"code": str})
        h["instrument"] = h["code"].map(norm_code)
        hold = h.groupby("instrument")["shares"].sum()
    else:
        hold = pd.Series(dtype=float)
    hold = hold[hold > 0]
    miss = [j for j in hold.index if j not in px.index]
    if miss:   # 非中证1000成分：从 Qlib 全市场数据取收盘价（与 A 系列相同）
        from engine.common import init_qlib
        from qlib.data import D
        init_qlib()
        q = D.features(miss, ["$close/$factor"], T, T).reset_index().set_index("instrument").iloc[:, -1]
        for j in miss:
            px.loc[j, "raw_close"] = float(q.get(j, np.nan))
            px.loc[j, ["up_lim", "dn_lim", "susp"]] = False
        bad = [j for j in miss if not px.loc[j, "raw_close"] > 0]
        if bad:
            print("⚠ 以下持仓查不到价格（代码有误或已退市），按 0 估值：", bad)
    cash = a.cash if a.cash is not None else (cfg["account"] if not len(hold) else 0.0)
    val = pd.Series({j: hold[j] * float(np.nan_to_num(px["raw_close"].get(j, np.nan))) for j in hold.index}, dtype=float)
    total = float(val.sum()) + cash

    score = pred[pred["datetime"] == T].set_index("instrument")["score"].sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(score) + 1), index=score.index)
    sell, keep_list = [], []
    for j in hold.index:
        r = rank.get(j, 10 ** 6)
        why = ("择时转空，全部卖出" if not on else "不在中证1000" if j not in rank.index else
               f"排名 {r} 跌出前 {keep}" if r > keep else "")
        (sell if why else keep_list).append((j, why))
    n_free = k - len(keep_list) if on else 0
    cands = [j for j in score.index if j not in hold.index and not bool(px["susp"].get(j, False))]
    buy, backup = cands[:max(n_free, 0)], cands[max(n_free, 0):max(n_free, 0) + 5]
    slot = total / k

    def row(action, j, shares=None, note=""):
        p = float(px["raw_close"].get(j, np.nan))
        if shares is None:
            lot = lot_of(j)
            shares = int(np.floor(slot / (1 + cfg["open_cost"] + cfg["slip"]) / (p * lot)) * lot) if p > 0 else 0
            if shares == 0:
                note = "资金不足一手，用替补"
        return dict(操作=action, 代码=j[2:] + "." + j[:2], 名称=names["code_name"].get(j, ""), 行业=names["industry"].get(j, ""),
                    排名=int(rank.get(j, -1)), 股数=int(shares), 参考价=round(p, 2),
                    参考金额=round(shares * p if p > 0 else 0, 0), 备注=note)

    orders = [row("卖出", j, hold[j], why) for j, why in sell] + [row("继续持有", j, hold[j]) for j, _ in keep_list] + \
             [row("买入", j) for j in buy] + [row("替补", j) for j in backup]
    out = pd.DataFrame(orders)
    os.makedirs("strategies2/signals", exist_ok=True)
    fn = f"strategies2/signals/{a.strategy}_{T.date()}.csv"
    out.to_csv(fn, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n==== {a.strategy}｜信号日 {T.date()}｜在下一交易日【开盘集合竞价】成交 ====")
    print(f"模型：{len(metas)} 个 5 日模型（{', '.join(metas)}），原始打分按排名等权平均；"
          f"训练截至 {list(metas.values())[0]['train'][1]}")
    marks = "、".join(str(m) + "日" + ("✓" if v else "✗") for m, v in above.items())
    print(f"择时：中证1000 站上 {sum(above.values())}/{len(above)} 条均线（{marks}），需要 ≥{cfg['timing_need']} → "
          + ("【持股】" if on else "【空仓】"))
    print(f"账户：股票 {val.sum():,.0f} + 现金 {cash:,.0f} = {total:,.0f}；持仓 {len(hold)} 只 / 最多 {k} 只；每只目标约 {slot:,.0f}")
    if a.lookback > 0:
        from experiments.conc_lib import load_panel, simulate, stats
        ps = pred["datetime"].min()
        panel = load_panel(ps, T)
        b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime").iloc[:, 0]
        lvl = (1 + b[b.index <= T]).cumprod()
        tm = sum((lvl > lvl.rolling(m).mean()).astype(int) for m in cfg["timing_mas"]) >= cfg["timing_need"]
        res = simulate(panel, pred, k=k, M=keep, exec_at=cfg["exec_at"], timing=tm, slip=cfg["slip"],
                       buy_cost=cfg["open_cost"], sell_cost=cfg["close_cost"])
        st = stats(res)
        bb = (1 + b[(b.index > ps) & (b.index <= T)]).prod() - 1
        print(f"模拟盘近 {a.lookback} 天：收益 {(1 + res['ret']).prod() - 1:+.1%}（中证1000 {bb:+.1%}），最大回撤 {st['最大回撤']:.1%}，"
              f"持股天数 {st['持仓天数占比']:.0%}；模拟盘当前持仓：{res['hold'].iloc[-1] or '空仓'}")
    print()
    main = out[out["操作"] != "替补"]
    print(main.to_string(index=False) if len(main) else "今日无需交易（空仓）")
    if len(buy):
        print("\n替补（买入目标开盘一字涨停/停牌买不进时按顺序替换）：",
              "、".join(f"{r.代码} {r.名称}" for r in out[out["操作"] == "替补"].itertuples()))
        print("下单建议：开盘集合竞价（9:15-9:25）挂买入；股数按今日收盘价估算，开盘价偏离较大时按 目标金额÷开盘价 调整")
    print(f"\n已保存 {fn}")


def cmd_signal(a):
    cfg = STRATEGIES[a.strategy]
    if a.lookback is None:
        a.lookback = 365
    if cfg["kind"] == "dropout":
        if a.cash is None:
            a.cash = cfg["account"] if not a.holdings else 0.0
        signal_dropout(a, cfg)
    else:
        signal_concentrated(a, cfg)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="训练生产模型")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--strategy", choices=list(STRATEGIES))
    g.add_argument("--model", choices=list(MODELS))
    t.add_argument("--train_start", default="2015-01-01")
    t.add_argument("--valid_end")
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--force", action="store_true", help="retail / rb 已存在也重训")
    s = sub.add_parser("signal", help="生成调仓清单")
    s.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    s.add_argument("--holdings", help="当前持仓 csv：code,shares")
    s.add_argument("--cash", type=float, default=None, help="可用现金（空仓首次建仓时默认等于配置中的初始资金）")
    s.add_argument("--etf_value", type=float, default=0.0, help="C_top50：当前持有的中证1000ETF 市值（R3 用）")
    s.add_argument("--date", help="信号日（默认最新交易日）")
    s.add_argument("--lookback", type=int, default=None, help="模拟盘回看天数（自然日，默认 365；K_top3 设 0 可跳过、更快）")
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_signal(a)


if __name__ == "__main__":
    main()
