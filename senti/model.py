"""
进攻版的两个“大波动”模型：大涨概率（3 日涨 ≥10%）、大跌概率（3 日跌 ≥10%）
  python senti/model.py train          # 训练两个生产模型 → models/senti_up.txt、models/senti_dn.txt（2 核约 20–40 分钟）
  python senti/model.py check          # 打印模型信息

特征 = 散户因子(16) + 长期犯错因子(12) + 个股短线特征(14) + 市场情绪(33)；参数、抽样与研究代码 experiments/senti.py train 相同。
训练区间与 K_top3 的六个 5 日模型一致：训练 2015-01-01 ~ 2024-08-29，早停验证 2024-09-13 ~ 2026-09-13。
"""
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.common import DATA, KEY  # noqa: E402
from senti.features import STOCK_F  # noqa: E402

TRAIN = ("2015-01-01", "2024-08-29")
VALID = ("2024-09-13", "2026-09-13")
PARAMS = dict(colsample_bytree=0.8, learning_rate=0.03, subsample=0.8, subsample_freq=1, lambda_l1=10.0, lambda_l2=500.0,
              max_depth=6, num_leaves=31, min_data_in_leaf=1000, verbosity=-1, num_threads=2, seed=0,
              objective="binary", metric="auc")
TASKS = {"up": (lambda y: (y >= 0.10).astype("float32"), "3 日（T+1 开盘买 → T+3 收盘）涨 ≥10% 的概率"),
         "dn": (lambda y: (y <= -0.10).astype("float32"), "3 日（T+1 开盘买 → T+3 收盘）跌 ≥10% 的概率")}


def model_path(tag):
    f = os.path.join(ROOT, "models", f"senti_{tag}.txt")
    return f, f.replace(".txt", ".json")


def load_xy(start, end, sample=1, need_y=True, ycol="y3"):
    """与 experiments/senti.py load_xy 相同：散户+行为因子矩阵 + 个股短线特征 + 市场情绪，按索引对齐写入预分配矩阵"""
    from engine import model as EM
    keys, X, names, _ = EM.load(["retail", "behavior"], start, end, 1, sample=sample, with_label=False)
    if keys is None or X is None or len(keys) == 0:
        raise ValueError(f"{start}..{end} has no model feature rows")
    dt = keys["datetime"].values
    ins = pd.Categorical(keys["instrument"].values)
    del keys
    # 省内存：只读用到的交易日，股票代码按字典读入（结果与逐行读取完全相同）
    import pyarrow.parquet as pq
    days = [pd.Timestamp(x) for x in np.unique(dt)]
    S = pq.read_table(os.path.join(DATA, "senti_stock.parquet"), columns=KEY + STOCK_F + [ycol],
                      filters=[("datetime", "in", days)], read_dictionary=["instrument"]).to_pandas()
    S = S[S["instrument"].isin(ins.categories)].reset_index(drop=True)
    S = S.drop_duplicates(KEY).reset_index(drop=True)
    S["instrument"] = S["instrument"].cat.remove_unused_categories()
    sidx = pd.MultiIndex.from_arrays([S["datetime"].values, pd.Categorical(S["instrument"].values, categories=ins.categories)])
    pos = sidx.get_indexer(pd.MultiIndex.from_arrays([dt, ins]))
    del sidx
    if S.empty:
        raise ValueError("No aligned stock sentiment features; rebuild senti/features.py")
    y = np.where(pos >= 0, S[ycol].values[np.maximum(pos, 0)], np.nan).astype("float32")
    keep = ~np.isnan(y) if need_y else np.ones(len(y), dtype=bool)
    M = pd.read_parquet(os.path.join(DATA, "senti_market.parquet")).set_index("datetime")
    if M.empty:
        raise ValueError("No market sentiment features")
    mcols = list(M.columns)
    mpos = M.index.get_indexer(dt)
    if not need_y and ((pos < 0).any() or (mpos < 0).any()):
        raise ValueError("Incomplete sentiment feature coverage; prediction aborted")
    n = int(keep.sum())
    out = np.empty((n, X.shape[1] + len(STOCK_F) + len(mcols)), dtype="float32")
    out[:, :X.shape[1]] = X[keep]
    del X
    gc.collect()
    sv = S[STOCK_F].values.astype("float32")
    del S
    p = pos[keep]
    blk = sv[np.maximum(p, 0)]
    blk[p < 0] = np.nan
    out[:, len(names):len(names) + len(STOCK_F)] = blk
    del blk, sv
    mv = M.values.astype("float32")
    mp = mpos[keep]
    blk = mv[np.maximum(mp, 0)]
    blk[mp < 0] = np.nan
    out[:, len(names) + len(STOCK_F):] = blk
    del blk
    out[~np.isfinite(out)] = np.nan
    kk = pd.DataFrame({"datetime": dt[keep], "instrument": np.asarray(ins)[keep]})
    gc.collect()
    return kk, out, y[keep], names + STOCK_F + mcols


def train():
    import lightgbm as lgb
    sample = int(os.environ.get("SENTI_SAMPLE", "2"))   # 训练集每隔 2 个交易日取一天（研究同设置），省内存
    last = pd.read_parquet(os.path.join(DATA, "bench.parquet"))["datetime"].max()
    for tag, (tf, desc) in TASKS.items():
        _, Xtr, ytr, feats = load_xy(*TRAIN, sample=sample)
        ntr = len(ytr)
        dtr = lgb.Dataset(Xtr, tf(ytr), feature_name=feats, free_raw_data=True)
        dtr.construct()
        del Xtr, ytr
        gc.collect()
        _, Xva, yva, _ = load_xy(*VALID)
        nva = len(yva)
        dva = lgb.Dataset(Xva, tf(yva), reference=dtr)
        dva.construct()
        del Xva, yva
        gc.collect()
        bst = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        del dtr, dva
        gc.collect()
        path, meta_path = model_path(tag)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bst.save_model(path, num_iteration=bst.best_iteration)
        imp = pd.Series(bst.feature_importance("gain", iteration=bst.best_iteration), index=feats).sort_values(ascending=False)
        meta = dict(target=desc, features=feats, best_iter=int(bst.best_iteration),
                    valid_auc=round(float(bst.best_score["valid_0"]["auc"]), 4), train=list(TRAIN), valid=list(VALID),
                    train_sample_every=sample, n_train=ntr, n_valid=nva, data_last_day=str(last.date()),
                    top_features=list(imp.index[:12]))
        json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[senti_{tag}] best_iter={meta['best_iter']} 验证 AUC={meta['valid_auc']}  重要特征：{', '.join(imp.index[:8])}",
              flush=True)
        del bst
        gc.collect()


def predict(start, end):
    """→ DataFrame[datetime, instrument, up, dn, score]，score = 大涨概率 − 大跌概率（“不对称”打分）"""
    import lightgbm as lgb
    kk, X, _, feats = load_xy(start, end, need_y=False)
    out = kk.copy()
    for tag in TASKS:
        path, meta_path = model_path(tag)
        if not os.path.exists(path):
            sys.exit(f"找不到模型 {path}，请先运行：python senti/model.py train")
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta["features"] != feats:
            sys.exit(f"senti_{tag} 的特征列与当前数据不一致，请重新训练")
        out[tag] = lgb.Booster(model_file=path).predict(X, num_iteration=meta["best_iter"]).astype("float32")
    out["score"] = out["up"] - out["dn"]
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "train":
        train()
    else:
        for t in TASKS:
            m = json.load(open(model_path(t)[1], encoding="utf-8"))
            print(t, {k: v for k, v in m.items() if k != "features"})
