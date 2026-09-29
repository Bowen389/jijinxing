"""
第二批策略的全部参数（与 strategies/ 下的 A 系列完全独立，互不影响）

  C_top50  分散版：散户行为 + 股东户数（散户涌入）  —— 三个模型集成，行业中性，持 50 只，每天换 2 只，R3 风控
  K_top3   集中版：每天最多重仓 3 只，允许空仓      —— 六个 5 日模型集成，原始打分，多均线投票择时，开盘成交

股票池都是中证1000（按历史成分）。
"""

# ---------------------------------------------------------------- 模型
# horizon=1：预测 T+1 收盘买入持有 1 天（C_top50 用）；horizon=5：持有 5 天（K_top3 用）
# neutral=True：行业中性后再集成（分散持仓用）；False：直接用原始打分（集中持仓用，保留小盘/行业倾向）
MODELS = {
    # C_top50
    "retail": dict(sets=["retail"], horizon=1, seed=0, file="models/retail_lgb.txt"),            # 与 A 系列共用
    "rb": dict(sets=["retail", "behavior"], horizon=1, seed=0, file="models/rb_lgb.txt"),       # 与 AE_top20 共用
    "rbh": dict(sets=["retail", "behavior", "em"], horizon=1, seed=0, file="models/rbh_lgb.txt"),
    # K_top3：两组特征 × 三个随机种子
    "k_rb_s0": dict(sets=["retail", "behavior"], horizon=5, seed=0, file="models/k_rb_h5_s0.txt"),
    "k_rb_s1": dict(sets=["retail", "behavior"], horizon=5, seed=1, file="models/k_rb_h5_s1.txt"),
    "k_rb_s2": dict(sets=["retail", "behavior"], horizon=5, seed=2, file="models/k_rb_h5_s2.txt"),
    "k_rbh_s0": dict(sets=["retail", "behavior", "em"], horizon=5, seed=0, file="models/k_rbh_h5_s0.txt"),
    "k_rbh_s1": dict(sets=["retail", "behavior", "em"], horizon=5, seed=1, file="models/k_rbh_h5_s1.txt"),
    "k_rbh_s2": dict(sets=["retail", "behavior", "em"], horizon=5, seed=2, file="models/k_rbh_h5_s2.txt"),
}

# 回测复现用的预测文件（研究期, 样本外）
RESEARCH_PREDS = {
    "C_top50": (["retail_h1_neu_ind", "rb_h1_neu_ind", "rbh_h1_neu_ind"],
                ["retail_h1_oos_neu_ind", "rb_h1_oos_neu_ind", "rbh_h1_oos_neu_ind"]),
    "K_top3": (["h5ens6"], ["h5ens6_oos"]),   # 六个 5 日模型原始打分的排名平均（已合成好）
}

COSTS = dict(open_cost=0.0005, close_cost=0.0010, min_cost=5.0)

STRATEGIES = {
    # ------------------------------------------------------------ 分散版
    # 2023-26 超额 12.5% / IR 1.62；样本外 2020-22 超额 16.9% / IR 2.09（A_top50 为 9.3% / 16.4%）
    "C_top50": dict(COSTS, kind="dropout", models=["retail", "rb", "rbh"], neutral=True,
                    topk=50, n_drop=2, every=1, account=1_000_000, risk_degree=0.95,
                    r3_in=-0.08, r3_out=-0.04, r3_exposure=0.5, hedge_etf="512100", bench="SH000852"),
    # ------------------------------------------------------------ 集中版
    # 2023-26 年化 20.2% / 最大回撤 -23%；样本外 2020-22 年化 13.0% / 最大回撤 -35%（绝对收益，扣费）
    # 换模型种子/特征后的结果中位数约 13%/年 —— 这才是合理预期，见 strategies2/README.md
    "K_top3": dict(COSTS, kind="concentrated", models=["k_rb_s0", "k_rb_s1", "k_rb_s2", "k_rbh_s0", "k_rbh_s1", "k_rbh_s2"],
                   neutral=False,
                   k=3,               # 最多同时持有 3 只，等权（每只约 1/3 资金）
                   keep_rank=60,      # 持仓股票排名跌出前 60 才卖（缓冲，降低换手）
                   timing_mas=[20, 40, 60, 80, 120],   # 中证1000 指数站上其中至少 timing_need 条均线才持股
                   timing_need=3,
                   exec_at="open",    # T 日收盘后出信号，T+1 开盘集合竞价成交
                   slip=0.001, account=1_000_000),
}
