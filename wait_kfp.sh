#!/bin/bash
RUN_ID=$1
until result=$(conda run -n kfp python get_kfp.py --run_id $RUN_ID 2>/dev/null); echo "$result" | grep -qE '"status": "(Succeeded|Failed|Error|Skipped)"'; do sleep 60; done
echo "$result"
