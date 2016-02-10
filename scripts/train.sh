#!/bin/bash

# set -e

for i in `seq 1 2`;
do
    JOB="new_baseline$i"
    SAVE="state/$JOB"
    PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -explore=10000 -memory_threshold=1000`
    ACTIVE="~/public_html/exp_vis/active/"$JOB
    VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
    SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
    FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
    EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; 
fi;"
    nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
done
