#!/bin/bash
# exp086 chain on rsna-cron (CPU). Amendment 18-oct: all2 = pilot2 (Opus L6 high x50 + 20 adjudications at max)
# -> budget check ($100; stops with OVER_BUDGET_100 unless work/GO_OVER_BUDGET) -> L6 Opus batch -> adjudication batch -> build.
D=/workspace/users/rodrigo/orq/labels_claude
EXPD=/workspace/users/rodrigo/wt_labels_claude/experiments/exp086_etiquetas_l6_claude
mkdir -p $D/work
until grep -q '^ANTHROPIC_WORKSPACE_ID=.\+' /root/.anthropic_env 2>/dev/null; do
  echo "$(date -u +%FT%TZ) [labels_claude exp086] waiting ANTHROPIC_WORKSPACE_ID" >> $D/status.log; sleep 600; done
cd $EXPD
/root/venv_l6/bin/python generar_L6.py all2 --out $D/work --dest $D --secrets /root/.anthropic_env >> $D/work/run.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) [labels_claude exp086] chain end rc=$rc" >> $D/status.log
