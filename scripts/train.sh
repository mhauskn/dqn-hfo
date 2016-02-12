#!/bin/bash

# set -e

# JOB="load_from_replaymem"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -memory_snapshot=state/new_baseline1_iter_130000.replaymemory`

ACTOR_LRS=".00001 .00003 .000003"
CRITIC_LRS=".001 .003 .0003"
MAX_ITER=200000
for actor_lr in $ACTOR_LRS;
do
    for critic_lr in $CRITIC_LRS;
    do
        JOB="alr$actor_lr"_"clr$critic_lr"
        SAVE="state/$JOB"
        PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -actor_lr=$actor_lr -critic_lr=$critic_lr -memory_snapshot=state/new_baseline1_iter_130000.replaymemory -max_iter=$MAX_ITER`
    done
done
