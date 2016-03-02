#!/bin/bash

# set -e

# 2-25-16 Again test update ratio. Hopefully no disk crashes this time...
values=".05 .1 .5"
for v in $values;
do
    JOB="UpdateRatio$v"
    SAVE="state/$JOB"
    PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -update_ratio=$v`
    ACTIVE="~/public_html/exp_vis/active/"$JOB
    VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
    SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
    FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
    EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
    nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
done

# 2-24-16 Testing different values of update ratio
# values=".1 .5 1 2"
# for v in $values;
# do
#     JOB="UpdateRatio$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -update_ratio=$v`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 2-24-16 Testing soft_update_freq
# values="1 10 100 1000"
# for v in $values;
# do
#     TAU=$(echo $v*.001 | bc)
#     JOB="SoftUpdateFreq$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -soft_update_freq=$v -tau=$TAU`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 2-23-16 Single threaded learning with new HFO
# JOB="st"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -memory_threshold=10000`
# ACTIVE="~/public_html/exp_vis/active/"$JOB
# VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
# SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
# FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
# EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
# nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &

# 2-23-16 Batch Normalization Jobs
# threads="1 3 6 9"
# MAX_ITER=2000000
# for t in $threads;
# do
#     JOB="bn_threads$t"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -memory_snapshot=state/new_baseline1_iter_130000.replaymemory -max_iter=$MAX_ITER -mt -player_threads=$t`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# ACTOR_LRS=".00001 .00003 .000003"
# CRITIC_LRS=".001 .003 .0003"
# MAX_ITER=200000
# for actor_lr in $ACTOR_LRS;
# do
#     for critic_lr in $CRITIC_LRS;
#     do
#         JOB="alr$actor_lr"_"clr$critic_lr"
#         SAVE="state/$JOB"
#         PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -actor_lr=$actor_lr -critic_lr=$critic_lr -memory_snapshot=state/new_baseline1_iter_130000.replaymemory -max_iter=$MAX_ITER`
#     done
# done

# momentums=".85 .95 .99 .999"
# MAX_ITER=200000
# for momentum in $momentums;
# do
#     JOB="momentum2is$momentum"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -memory_snapshot=state/new_baseline1_iter_130000.replaymemory -momentum2=$momentum -max_iter=$MAX_ITER`
# done

# clips="1 5 10 20"
# MAX_ITER=200000
# for clip in $clips;
# do
#     JOB="clip$clip"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -memory_snapshot=state/new_baseline1_iter_130000.replaymemory -clip_grad=$clip -max_iter=$MAX_ITER`
# done
