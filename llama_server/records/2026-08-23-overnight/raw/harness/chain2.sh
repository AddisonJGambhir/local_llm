#!/usr/bin/env bash
bash /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/wave_tpp.sh
sleep 20
bash /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/wave_temp.sh
sleep 20
bash /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/wave_m5.sh
echo "[$(date +%H:%M:%S)] CHAIN2 COMPLETE" >> /tmp/claude-1000/-home-addison-gambhir-Desktop-local-llm/5d1e4b40-5a13-49fe-976a-6202c289bb7b/scratchpad/overnight/run.log
