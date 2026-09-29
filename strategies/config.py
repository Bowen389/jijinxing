"""
三个实盘策略的全部参数（改这里即可，live.py / backtest.py 都从这里读）

共同部分：
  股票池    中证1000 成分股（按历史成分，防幸存者偏差）
  因子      retail_factors.py 中的散户行为因子（每个策略用哪些模型见 models）
  模型      LightGBM（robust 超参），标签 = T+1 收盘买入、持有 1 日的收益（剔除次日买不进的样本）
  中性化    每日把打分转成排名后减去所属证监会行业均值 → 行业中性
  多模型    各模型分别行业中性后，按当日排名等权平均
  成交      T 日收盘后出信号，T+1 日尾盘集合竞价成交
  风控 R3   策略超额（相对中证1000）从高点回撤超过 8% → 一半仓位换成中证1000 ETF；回撤收窄到 4% 以内恢复
"""

# 可用的模型：名称 -> 特征集 + 模型文件（python strategies/live.py train 会训练全部）
MODELS = {
    "retail": dict(sets=["retail"], file="models/retail_lgb.txt",
                   desc="16 个散户短期行为因子（20 日以内：追涨、彩票偏好、异常放量、尾盘拉升等）"),
    "rb": dict(sets=["retail", "behavior"], file="models/rb_lgb.txt",
               desc="16 个短期 + 12 个长期犯错因子（60~250 日：反复被关注、处置效应、锚定、长期过度反应、恐慌）"),
}

# 回测复现用的预测文件（研究期, 样本外），均已行业中性
RESEARCH_PREDS = {
    "retail": ("retail_h1_neu_ind", "retail_h1_oos_neu_ind"),
    "rb": ("rb_h1_neu_ind", "rb_h1_oos_neu_ind"),
}

COMMON = dict(
    pool="csi1000",
    bench="SH000852",
    models=["retail"],
    open_cost=0.0005,     # 买入：佣金+过户等（万5，含最低 5 元）
    close_cost=0.0010,    # 卖出：佣金 + 印花税 0.05%（万10）
    min_cost=5.0,
    risk_degree=0.95,     # 最多用 95% 资金买股票
    r3_in=-0.08,          # 超额回撤低于 -8% → 半仓
    r3_out=-0.04,         # 回到 -4% 以内 → 恢复满仓
    r3_exposure=0.5,
    hedge_etf="512100",   # 中证1000ETF（南方）；也可用 159845 / 560010 等
)

STRATEGIES = {
    # 主策略：分散、稳健。2023-26 超额 9.3% / IR 1.24；样本外 2020-22 超额 16.4% / IR 2.01
    "A_top50": dict(COMMON, topk=50, n_drop=2, every=1, account=1_000_000),
    # 进取版：更集中、收益弹性更大，但波动与回撤更大。2023-26 超额 12.1% / IR 1.22；样本外 15.6% / IR 1.49
    "A_top20": dict(COMMON, topk=20, n_drop=1, every=1, account=1_000_000),
    # 集成进取版：两个模型（短期因子 / 短期+长期犯错因子）打分平均，持 20 只。
    # 2023-26 超额 11.8% / IR 1.17；样本外 26.9% / IR 2.43。两段差距很大，结果不稳定，见 strategies/README.md
    "AE_top20": dict(COMMON, models=["retail", "rb"], topk=20, n_drop=1, every=1, account=1_000_000),
}
