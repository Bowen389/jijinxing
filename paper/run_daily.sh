#!/usr/bin/env bash
# 组合模拟盘每日流程（GitHub Actions 调用，本地也可以直接跑）：
#   更新数据 → 重建面板/因子 → 股东户数增量 → 市场跌停占比 → 两仓成交 + 再平衡 + 出清单 → 生成 paper/README.md
# 数据目录用环境变量覆盖；行业和股东户数放在仓库里的 paper/data/extra（每天增量更新并提交，避免每次全量抓取 15 分钟）
set -euo pipefail
cd "$(dirname "$0")/.."
export RA_PROVIDER=${RA_PROVIDER:-$HOME/.qlib/qlib_data/cn_data}
export RA_DATA=${RA_DATA:-$HOME/.qlib/retail_alpha_data}
export RA_EXTRA=${RA_EXTRA:-$(pwd)/paper/data/extra}
export PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH}
PY=${PYTHON:-python}
mkdir -p "$RA_PROVIDER" "$RA_DATA" "$RA_EXTRA"

echo "::group::1) 下载 Qlib 日线数据"
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  curl -sSL --retry 5 --retry-delay 10 -o /tmp/qlib_bin.tar.gz \
    https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
  rm -rf "$RA_PROVIDER"/* && tar -xzf /tmp/qlib_bin.tar.gz -C "$RA_PROVIDER" --strip-components=1 && rm -f /tmp/qlib_bin.tar.gz
fi
LAST_DAY=$(tail -1 "$RA_PROVIDER/calendars/day.txt")
echo "Qlib 数据最新交易日：$LAST_DAY"
echo "::endgroup::"

# 所有账户都已处理到最新交易日 → 不需要再跑（重复触发时省时间）
if [ "${FORCE:-0}" != "1" ] && [ -d paper/accounts ]; then
  NEED=$($PY - "$LAST_DAY" <<'EOF'
import json, os, sys
last = sys.argv[1]
names = ["C_top50", "FB_dip"]
need = 0
for n in names:
    f = f"paper/accounts/{n}/state.json"
    if not os.path.exists(f) or (json.load(open(f, encoding="utf-8")).get("last_date") or "") < last:
        need = 1
print(need)
EOF
)
  if [ "$NEED" = "0" ]; then echo "所有模拟账户已处理到 $LAST_DAY，本次无需运行"; echo "NOTHING_TO_DO=1" >> "${GITHUB_ENV:-/dev/null}"; exit 0; fi
fi

echo "::group::2) 行业分类（baostock；失败则沿用仓库里的旧文件）"
if ! timeout 300 $PY data_fetch/fetch_industry.py; then
  echo "⚠ 行业分类更新失败，沿用 $RA_EXTRA/industry.parquet"
  test -f "$RA_EXTRA/industry.parquet"
fi
echo "::endgroup::"

echo "::group::3) 面板 + 散户因子 + 长期犯错因子（C_top50 的三个模型要用）"
$PY -m engine.panel
$PY -m engine.features --set retail
$PY -m engine.features --set behavior
echo "::endgroup::"

echo "::group::4) 股东户数（东方财富；增量失败则沿用旧数据）"
if [ -f "$RA_EXTRA/em_holders.parquet" ]; then
  timeout 300 $PY data_fetch/fetch_em.py update || echo "⚠ 股东户数增量更新失败，沿用旧数据"
else
  $PY data_fetch/fetch_em.py holders
  rm -rf "$RA_EXTRA/em_holders_parts"
fi
$PY -m engine.em_features
echo "::endgroup::"

echo "::group::5) 全市场跌停占比（FB_dip 的市场条件）"
$PY paper/market.py
echo "::endgroup::"

echo "::group::6) 组合模拟盘：成交 → 盯市 → 再平衡 → 生成下一交易日清单"
$PY paper/sim.py run
echo "::endgroup::"
rm -rf paper/.tmp
echo "PAPER_DONE $LAST_DAY"
