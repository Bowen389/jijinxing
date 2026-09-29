"""
训练 + 预测（省内存：按日期读取、训练集可隔日抽样；支持滚动重训）
  python -m engine.model --sets retail --horizon 1 --tag retail_h1
  python -m engine.model --sets alpha158,retail --horizon 1 --sample 3 --tag both_h1
  python -m engine.model --sets alpha158,retail,extra --rolling 63 --tag both_extra_roll
输出 output/preds/{tag}.parquet（datetime, instrument, score）与 output/importance/{tag}.csv
"""
import argparse
import gc
import os

import numpy as np
import pandas as pd

from engine.common import DATA, KEY, read_parts

LGB_PRESETS = {
    "official": dict(objective="regression", colsample_bytree=0.8879, learning_rate=0.2, subsample=0.8789,
                     subsample_freq=1, lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210,
                     verbosity=-1),
    "robust": dict(objective="regression", colsample_bytree=0.8, learning_rate=0.03, subsample=0.8,
                   subsample_freq=1, lambda_l1=10.0, lambda_l2=500.0, max_depth=6, num_leaves=31,
                   min_data_in_leaf=2000, verbosity=-1),
}


TRADABLE_LABEL = os.environ.get("RA_TRADABLE_LABEL", "1") == "1"


def _trim():
    """把已释放的内存还给操作系统（glibc），小内存机器上很关键"""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa
        pass


def trading_days(start, end):
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet"))
    d = b["datetime"]
    return d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))].tolist()


def _read_table(path, start, end, flt, columns=None):
    import pyarrow as pa
    import pyarrow.dataset as ds
    f = (ds.field("datetime") >= pd.Timestamp(start)) & (ds.field("datetime") <= pd.Timestamp(end))
    if flt is not None:
        f = f & flt
    tabs = []
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".parquet"):
            t = ds.dataset(os.path.join(path, fn), format="parquet").to_table(columns=columns, filter=f, use_threads=False)
            if t.num_rows:
                tabs.append(t)
    return pa.concat_tables(tabs) if tabs else None


def load(sets, start, end, horizon, sample=1, with_label=True):
    """读取多个特征集 → (keys DataFrame, X float32 矩阵, 特征名, y)。
    所有特征集与面板按 (instrument, datetime) 同序存储，直接写入预分配矩阵，避免 DataFrame 拷贝。
    sample>1 时每隔 sample 个交易日取一天。"""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    days = trading_days(start, end)[::sample]
    flt = ds.field("datetime").isin(pd.to_datetime(days).values) if sample > 1 else None
    names_all = []
    for s in sets:
        d = os.path.join(DATA, f"feat_{s}")
        first = sorted(x for x in os.listdir(d) if x.endswith(".parquet"))[0]
        names_all.append([c for c in pq.read_schema(os.path.join(d, first)).names if c not in KEY])
    ncol = sum(len(n) for n in names_all)
    import pyarrow.dataset as ds2
    f = (ds2.field("datetime") >= pd.Timestamp(start)) & (ds2.field("datetime") <= pd.Timestamp(end))
    if flt is not None:
        f = f & flt
    keys, X, j = None, None, 0
    for s, names in zip(sets, names_all):
        d = os.path.join(DATA, f"feat_{s}")
        files = sorted(x for x in os.listdir(d) if x.endswith(".parquet"))
        # 逐文件读、逐文件写入矩阵：峰值内存 ≈ 矩阵 + 单个文件
        blocks_k, r0 = [], 0
        if keys is None:
            n = sum(ds2.dataset(os.path.join(d, fn)).count_rows(filter=f) for fn in files)
            X = np.full((n, ncol), np.nan, dtype="float32")
        for fn in files:
            t = ds2.dataset(os.path.join(d, fn), format="parquet").to_table(filter=f, use_threads=False)
            if t.num_rows == 0:
                continue
            dt = t.column("datetime").to_numpy()
            ins = t.column("instrument").to_numpy(zero_copy_only=False).astype(str)
            blocks_k.append((dt, ins))
            r1 = r0 + t.num_rows
            for jj, c in enumerate(names):
                X[r0:r1, j + jj] = t.column(c).to_numpy().astype("float32", copy=False)
            r0 = r1
            del t
        dt_all = np.concatenate([b[0] for b in blocks_k]) if blocks_k else np.array([], dtype="datetime64[ns]")
        ins_all = np.concatenate([b[1] for b in blocks_k]) if blocks_k else np.array([], dtype=str)
        if keys is None:
            if len(dt_all) == 0:
                return None, None, [], None
            keys = pd.DataFrame({"datetime": dt_all, "instrument": ins_all})
        elif not (len(dt_all) == len(keys) and (dt_all == keys["datetime"].values).all()
                  and (ins_all == keys["instrument"].values).all()):
            raise RuntimeError(f"特征集 {s} 与 {sets[0]} 的行顺序不一致，请用相同的 member 表重建")
        j += len(names)
        _trim()
    X[~np.isfinite(X)] = np.nan
    names_flat = [n for ns in names_all for n in ns]
    y = None
    if with_label:
        # RA_LABEL / RA_NB 可切换标签口径，例如开盘成交：RA_LABEL=reto1 RA_NB=nbo
        lcol, ncol = os.environ.get("RA_LABEL", f"ret{horizon}"), os.environ.get("RA_NB", "nb")
        lab = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["in_pool", ncol, lcol], start=start,
                         end=end, filters_extra=flt).rename(columns={lcol: f"ret{horizon}", ncol: "nb"})
        lab = lab[lab["in_pool"] & (lab["datetime"] >= pd.Timestamp("2015-01-01"))].reset_index(drop=True)
        if TRADABLE_LABEL:   # 次日买不进（涨停/停牌）的样本不参与训练，避免学到"追涨停"这种执行不了的规律
            lab.loc[lab["nb"], f"ret{horizon}"] = np.nan
        if len(lab) == len(keys) and (lab["datetime"].values == keys["datetime"].values).all() and \
                (lab["instrument"].values == keys["instrument"].values).all():
            y = lab[f"ret{horizon}"].values.astype("float32")
        else:
            y = keys.merge(lab, on=KEY, how="left")[f"ret{horizon}"].values.astype("float32")
        del lab
    _trim()
    return keys, X, names_flat, y


def cs_zscore(dt, y):
    s = pd.Series(y)
    g = s.groupby(dt)
    return ((s - g.transform("mean")) / (g.transform("std") + 1e-12)).clip(-5, 5).values.astype("float32")


def cs_decile(dt, y):
    s = pd.Series(y)
    return s.groupby(dt).transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)).values.astype(int)


def _clean(data):
    k, X, cols, y = data
    m = ~np.isnan(y)
    if not m.all():
        k, X, y = k[m].reset_index(drop=True), X[m], y[m]
    return k, X, cols, y


def fit(train_fn, valid_fn, model="lgb", preset="robust", threads=2, seed=0, fixed_rounds=0):
    """train_fn/valid_fn: 无参函数，返回 (keys, X, names, y)。先建训练集并释放原始矩阵，再加载验证集，降低峰值内存。"""
    if model == "lgb":
        import lightgbm as lgb
        params = dict(LGB_PRESETS[preset], num_threads=threads, seed=seed)
        ktr, Xtr, cols, ytr = _clean(train_fn())
        dtr = lgb.Dataset(Xtr, cs_zscore(ktr["datetime"].values, ytr), free_raw_data=True, params={"verbosity": -1})
        dtr.construct()
        del Xtr, ktr, ytr
        _trim()
        kva, Xva, _, yva = _clean(valid_fn())
        dva = lgb.Dataset(Xva, cs_zscore(kva["datetime"].values, yva), reference=dtr)
        dva.construct()
        del Xva
        _trim()
        if fixed_rounds:
            m = lgb.train(params, dtr, num_boost_round=fixed_rounds)
            m.best_iteration = fixed_rounds
        else:
            m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        return m, cols, imp, m.best_iteration
    ktr, Xtr, cols, ytr = _clean(train_fn())
    kva, Xva, _, yva = _clean(valid_fn())
    if model in ("lgb", "lgb_rank"):
        import lightgbm as lgb
        params = dict(LGB_PRESETS[preset], num_threads=threads, seed=seed)
        if model == "lgb_rank":
            # 排序目标：每天一组，标签 = 当日收益分 10 档
            otr = np.argsort(ktr["datetime"].values, kind="stable")
            ova = np.argsort(kva["datetime"].values, kind="stable")
            ktr, Xtr, ytr = ktr.iloc[otr].reset_index(drop=True), Xtr[otr], ytr[otr]
            kva, Xva, yva = kva.iloc[ova].reset_index(drop=True), Xva[ova], yva[ova]
            params.update(objective="lambdarank", metric="ndcg", eval_at=[50], lambdarank_truncation_level=100,
                          label_gain=list(range(10)))
            dtr = lgb.Dataset(Xtr, cs_decile(ktr["datetime"].values, ytr), group=ktr.groupby("datetime", sort=False).size().values)
            dva = lgb.Dataset(Xva, cs_decile(kva["datetime"].values, yva), group=kva.groupby("datetime", sort=False).size().values, reference=dtr)
        else:
            dtr = lgb.Dataset(Xtr, cs_zscore(ktr["datetime"].values, ytr), free_raw_data=True)
            dva = lgb.Dataset(Xva, cs_zscore(kva["datetime"].values, yva), reference=dtr)
        dtr.construct(); dva.construct()
        del Xtr, Xva
        gc.collect()
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        return m, cols, imp, m.best_iteration
    if model == "xgb":
        import xgboost as xgb
        dtr = xgb.QuantileDMatrix(Xtr, cs_zscore(ktr["datetime"].values, ytr), max_bin=64)
        dva = xgb.DMatrix(Xva, cs_zscore(kva["datetime"].values, yva))
        del Xtr
        gc.collect()
        params = dict(max_depth=6, eta=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=2000,
                      reg_lambda=500, tree_method="hist", max_bin=64, nthread=threads, seed=seed)
        m = xgb.train(params, dtr, 2000, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=False)
        imp = pd.Series(m.get_score(importance_type="gain")).rename(index=lambda k: cols[int(k[1:])])
        return m, cols, imp.sort_values(ascending=False), m.best_iteration
    raise ValueError(model)


def predict(m, X, model):
    if model == "xgb":
        import xgboost as xgb
        return m.predict(xgb.DMatrix(X), iteration_range=(0, m.best_iteration + 1))
    return m.predict(X, num_iteration=m.best_iteration)


def predict_range(m, cols, sets, start, end, horizon, model):
    """按季度分块预测，控制内存"""
    out = []
    for s, e in quarter_chunks(start, end):
        keys, X, names, _ = load(sets, s, e, horizon, with_label=False)
        if keys is None or len(keys) == 0:
            continue
        assert names == cols, "特征列不一致"
        keys["score"] = predict(m, X, model).astype("float32")
        out.append(keys)
        del X
        gc.collect()
    return pd.concat(out, ignore_index=True)


def quarter_chunks(start, end):
    qs = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="QS")
    edges = [pd.Timestamp(start)] + [q for q in qs if q > pd.Timestamp(start)] + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    return [(edges[i], edges[i + 1] - pd.Timedelta(days=1)) for i in range(len(edges) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="retail")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--model", choices=["lgb", "lgb_rank", "xgb"], default="lgb")
    ap.add_argument("--preset", choices=list(LGB_PRESETS), default="robust")
    ap.add_argument("--train", nargs=2, default=["2015-01-01", "2020-12-31"])
    ap.add_argument("--valid", nargs=2, default=["2021-01-01", "2022-12-31"])
    ap.add_argument("--test", nargs=2, default=["2023-01-01", "2026-09-18"])
    ap.add_argument("--sample", type=int, default=1, help="训练集每隔N天取1天（省内存）")
    ap.add_argument("--rolling", type=int, default=0, help=">0: 每N个交易日滚动重训（滑动窗口）")
    ap.add_argument("--window_years", type=int, default=6)
    ap.add_argument("--fixed_rounds", type=int, default=0, help=">0: 不早停，固定树数（滚动模式推荐 100），验证期并入训练")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    sets = a.sets.split(",")
    os.makedirs("output/preds", exist_ok=True)
    os.makedirs("output/importance", exist_ok=True)

    if a.rolling <= 0:
        m, cols, imp, it = fit(lambda: load(sets, *a.train, a.horizon, sample=a.sample),
                               lambda: load(sets, *a.valid, a.horizon, sample=max(1, a.sample - 1)),
                               a.model, a.preset, a.threads, a.seed)
        _trim()
        print(f"[{a.tag}] best_iter={it} features={len(cols)}", flush=True)
        pred = predict_range(m, cols, sets, *a.test, a.horizon, a.model)
    else:
        # 滚动：每 rolling 个交易日，用过去 window_years 年（最后 1 年做验证）重训
        days = trading_days(a.test[0], a.test[1])
        preds, imps = [], []
        for k in range(0, len(days), a.rolling):
            t0 = days[k]
            t1 = days[min(k + a.rolling, len(days)) - 1]
            gap = pd.Timedelta(days=15)  # 标签需要未来数据，留出间隔防泄露
            v_end = t0 - gap
            v_start = v_end - pd.DateOffset(years=1)
            tr_start = max(pd.Timestamp("2015-01-01"), v_end - pd.DateOffset(years=a.window_years))
            tr_end = v_end if a.fixed_rounds else v_start - gap
            vs = v_end - pd.DateOffset(months=2) if a.fixed_rounds else v_start
            m, cols, imp, it = fit(lambda: load(sets, tr_start, tr_end, a.horizon, sample=a.sample),
                                   lambda: load(sets, vs, v_end, a.horizon, sample=max(1, a.sample - 1)),
                                   a.model, a.preset, a.threads, a.seed, a.fixed_rounds)
            _trim()
            p = predict_range(m, cols, sets, t0, t1, a.horizon, a.model)
            preds.append(p)
            imps.append(imp / imp.sum())
            print(f"[{a.tag}] {t0.date()}~{t1.date()} train {tr_start.date()}~{tr_end.date()} iter={it}", flush=True)
        pred = pd.concat(preds, ignore_index=True)
        imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    pred.to_parquet(f"output/preds/{a.tag}.parquet", index=False)
    imp.to_csv(f"output/importance/{a.tag}.csv", header=["gain"])
    print(f"[{a.tag}] saved {len(pred)} preds; top features:", ", ".join(imp.index[:8]), flush=True)


if __name__ == "__main__":
    main()
