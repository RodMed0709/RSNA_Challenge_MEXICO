#!/bin/bash
# exp086 amendment 2: wait for L6 (Opus batch) -> debate (GPT-5.6-sol vs Opus 5.5) -> judge -> build consenso_v6 + READY.
D=/workspace/users/rodrigo/orq/labels_claude
W=$D/work
EXPD=/workspace/users/rodrigo/wt_labels_claude/experiments/exp086_etiquetas_l6_claude
cd $EXPD
while pgrep -f "generar_L6.py l6" > /dev/null; do sleep 60; done
n=$(wc -l < $W/l6.jsonl)
echo "$(date -u +%FT%TZ) [labels_claude exp086] L6 Opus finished: $n lines in l6.jsonl" >> $D/status.log
if [ "$n" -lt 4360 ]; then
  /root/venv_l6/bin/python generar_L6.py l6 --out $W --secrets /root/.anthropic_env >> $W/run.log 2>&1
fi
/root/venv_l6/bin/python debate_L6.py run --out $W >> $W/run.log 2>&1
echo "$(date -u +%FT%TZ) [labels_claude exp086] debate rc=$?" >> $D/status.log
/root/venv_l6/bin/python debate_L6.py build --out $W --dest $D >> $W/run.log 2>&1
rc=$?
grep "debate: build" $W/status.log | tail -1 >> $D/status.log
echo "$(date -u +%FT%TZ) [labels_claude exp086] chain end rc=$rc (READY: $(ls $D/READY.json 2>/dev/null))" >> $D/status.log
