"""
散户行为因子库（Qlib 表达式版）
================================
思路：A 股超额收益的一大来源是散户的系统性行为偏差。
这里把行为金融学里研究得比较透的几类“散户愚蠢”翻译成 Qlib 表达式，
既可以单独用（RetailAlpha），也可以叠加到 Qlib 自带的 Alpha158 上（Alpha158Retail）。

约定：每个因子都已调整好方向 —— 数值越大，理论上未来收益越好（便于看 IC 正负）。
可用字段（社区数据 chenditc/investment_data）：
    $open $high $low $close $vwap $volume（均为复权）  $amount（成交额）
    $change（当日涨跌幅） $factor（复权因子，$close/$factor ≈ 未复权价）
"""
from qlib.contrib.data.handler import Alpha158, _DEFAULT_LEARN_PROCESSORS
from qlib.contrib.data.handler import DataHandlerLP, check_transform_proc

# (表达式, 名称, 说明)
RETAIL_FACTORS = [
    # ---------- 1. 追涨杀跌 / 过度反应 -> 短期反转 ----------
    ("-1*($close/Ref($close,5)-1)",                 "REV5",     "5日反转：散户追涨后回吐"),
    ("-1*($close/Ref($close,20)-1)",                "REV20",    "20日反转"),
    ("-1*Mean($close/$open-1,20)",                  "INTRA20",  "日内收益反转：散户盘中追涨"),
    ("Mean($open/Ref($close,1)-1,20)",            "OVN20",    "隔夜收益：机构/信息主导的部分"),
    # ---------- 2. 彩票偏好 ----------
    ("-1*Max($change,20)",                          "MAX20",    "MAX效应：近期暴涨过的票被高估"),
    ("-1*Skew($change,20)",                         "SKEW20",   "正偏度（彩票型）被高估"),
    ("-1*Sum(If(Gt($change,0.095),1,0),20)",        "BIGUP20",  "近20日大涨(>9.5%)次数，涨停追逐"),
    ("-1*Std($change,20)",                          "VOL20",    "低波动异象"),
    # ---------- 3. 过度交易 / 注意力驱动 ----------
    ("-1*Mean($volume,5)/(Mean($volume,60)+1e-12)", "ABVOL5",   "异常放量：被关注后透支"),
    ("-1*Std($volume,20)/(Mean($volume,20)+1e-12)", "VOLVOL20", "成交量波动：情绪不稳"),
    ("-1*Corr($close/Ref($close,1),Log($volume+1),20)", "PVCORR20", "量价同向：放量追涨"),
    ("-1*Mean(($high-Greater($open,$close))/$close,20)", "USHADOW20", "上影线：冲高回落/诱多"),
    # ---------- 4. 小盘 / 低价 / 流动性 ----------
    ("Log(Mean(Abs($change)/($amount+1),20)+1e-15)", "AMIHUD20", "Amihud 非流动性溢价"),
    ("-1*Log(Mean($amount,20)+1)",                  "SMALL20",  "成交额规模(小盘代理)"),
    ("-1*Log($close/$factor)",                      "LOWPRICE", "低价股偏好（原始价格）"),
    # ---------- 5. 尾盘行为 ----------
    ("-1*($close/$vwap-1)",                         "CLOSEVWAP","收盘相对均价：尾盘拉升"),
]


# 散户"长期反复犯的错"（窗口 60~250 日）。方向统一为：数值越大，理论上未来收益越好；不确定方向的按原值给出，由 IC 检验
BEHAVIOR_FACTORS = [
    # ---------- 6. 处置效应：赚了急着卖、亏了死扛（Grinblatt-Han 资本利得悬置 CGO，参考价 = 成交量加权成本） ----------
    ("$close/(Sum($close*$volume,60)/(Sum($volume,60)+1e-12))-1",   "CGO60",    "近60日持仓浮盈：浮盈大→散户急于卖出→价格被压低"),
    ("$close/(Sum($close*$volume,250)/(Sum($volume,250)+1e-12))-1", "CGO250",   "近一年持仓浮盈"),
    # ---------- 7. 锚定：盯着一年高点/低点 ----------
    ("$close/Max($high,250)-1",                                     "HIGH250",  "离一年高点的距离：接近高点时散户不敢买（锚定）"),
    ("-1*($close/Min($low,250)-1)",                                 "LOW250",   "离一年低点的距离：远离低点=涨多了"),
    # ---------- 8. 长期过度反应 / 外推 ----------
    ("-1*($close/Ref($close,120)-1)",                               "REV120",   "半年反转：散户对趋势外推过度"),
    ("-1*Sum(If(Gt($change,0),1,0),10)/10",                         "UPDAYS10", "近10日上涨天数占比：连涨→外推买入"),
    # ---------- 9. 注意力与过度交易（长窗口） ----------
    ("-1*Sum(If(Gt($volume,2*Mean($volume,60)),1,0),60)",           "ATTN60",   "近60日爆量天数：反复被关注"),
    ("-1*Mean($volume,20)/(Mean($volume,250)+1e-12)",               "ABVOL250", "成交量相对一年均值：过度自信期"),
    ("-1*Sum(If(Gt(Abs($change),0.07),1,0),60)",                    "EXTREME60","近60日大波动天数：彩票属性"),
    ("-1*Sum(If(Gt($change,0),$volume,0),20)/(Sum($volume,20)+1e-12)", "UPVOL20", "上涨日成交占比：散户只在上涨时买"),
    # ---------- 10. 恐慌 ----------
    ("Sum(If(Lt($change,-0.095),1,0),20)",                          "DNLIM20",  "近20日跌停次数：散户恐慌抛售后的反弹"),
    ("-1*Min($change,20)",                                          "MIN20",    "近20日最大单日跌幅（取负）：恐慌程度"),
]


def behavior_feature_config():
    return [f for f, _, _ in BEHAVIOR_FACTORS], [n for _, n, _ in BEHAVIOR_FACTORS]


def retail_feature_config():
    fields = [f for f, _, _ in RETAIL_FACTORS]
    names = [n for _, n, _ in RETAIL_FACTORS]
    return fields, names


class RetailAlpha(DataHandlerLP):
    """只用散户行为因子（16 个），轻量、可解释。"""

    def __init__(self, instruments="csi1000", start_time=None, end_time=None, freq="day",
                 infer_processors=[], learn_processors=_DEFAULT_LEARN_PROCESSORS,
                 fit_start_time=None, fit_end_time=None, process_type=DataHandlerLP.PTYPE_A,
                 filter_pipe=None, inst_processors=None, label=None, **kwargs):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        label = label or (["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {"feature": retail_feature_config(), "label": label},
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(instruments=instruments, start_time=start_time, end_time=end_time,
                         data_loader=data_loader, infer_processors=infer_processors,
                         learn_processors=learn_processors, process_type=process_type, **kwargs)


class Alpha158Retail(Alpha158):
    """Alpha158（158个通用量价因子） + 16 个散户行为因子。"""

    def get_feature_config(self):
        f158, n158 = super().get_feature_config()
        fr, nr = retail_feature_config()
        return f158 + fr, n158 + nr
