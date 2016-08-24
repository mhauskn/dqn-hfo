#!/bin/bash
set -e

# Function to monitor the running job.
function monitor {
    ACTIVE="~/public_html/exp_vis/active/"$1
    VIS_CMD="./scripts/save.sh "$2"_INFO_* $ACTIVE"
    SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
    FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
    EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
    nohup monitor-condor-job --pid=$3 --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
}



# 8-23-16
# JOB="MoveToBall_NoComm"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 1000000 -offense_agents 2 -tasks move_to_ball`
# monitor $JOB $SAVE $PID

# JOB="MoveToBall_CommAct1"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 1000000 -offense_agents 2 -tasks move_to_ball -comm_actions 1`
# monitor $JOB $SAVE $PID

JOB="SoccerEasy_sanity"
SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 2000000 -offense_agents 1 -tasks soccer_easy`
monitor $JOB $SAVE $PID

# JOB="MirrorActions_NoComm"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 1000000 -offense_agents 2 -tasks mirror_actions`
# monitor $JOB $SAVE $PID

# JOB="MirrorActions_CommAct1"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 1000000 -offense_agents 2 -tasks mirror_actions -comm_actions 1`
# monitor $JOB $SAVE $PID

# JOB="MirrorActions_ShareLayers2_NoComm"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 1000000 -offense_agents 2 -tasks mirror_actions -share_actor_layers 2 -share_critic_layers 2`
# monitor $JOB $SAVE $PID

# JOB="MirrorActions_ShareLayers2_CommAct1"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 1000000 -offense_agents 2 -tasks mirror_actions -share_actor_layers 2 -share_critic_layers 2 -comm_actions 1`
# monitor $JOB $SAVE $PID

# 8-15-16
# JOB="Cross"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 5000000 -offense_agents 2 -tasks cross`
# monitor $JOB $SAVE $PID

# 8-13-16
# JOB="MultiagentPassNaive"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 5000000 -offense_agents 2 -tasks pass`
# monitor $JOB $SAVE $PID

# JOB="MultiagentPassShare2Layers"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 5000000 -offense_agents 2 -tasks pass -share_actor_layers 2 -share_critic_layers 2`
# monitor $JOB $SAVE $PID

# 8-12-16
# JOB="SequentialCurriculum"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 10000000 -tasks move_to_ball,dribble,kick_to_goal,soccer -curriculum sequential`
# monitor $JOB $SAVE $PID

# 8-1-16 Try multiple tasks
# JOB="SingleAgentCurriculum"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 10000000 -tasks move_to_ball,kick_to_goal,dribble,soccer`
# monitor $JOB $SAVE $PID

# JOB="Soccer_sanity"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 5000000 -offense_agents 1 -tasks soccer`
# monitor $JOB $SAVE $PID

# 7-26-16 Sanity check pass task
# JOB="Pass_sanity"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 5000000 -offense_agents 1 -tasks pass`
# monitor $JOB $SAVE $PID

# 7-22-16 Sanity check dribble
# JOB="Dribble_sanity"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 1000000 -tasks dribble`
# monitor $JOB $SAVE $PID

# JOB="KickToGoal_sanity"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 300000 -tasks kick_to_goal`
# monitor $JOB $SAVE $PID

# 7-15-16 Sanity check the task system
# JOB="MoveToBall_sanity"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./bin/dqn -save=$SAVE -max_iter 300000 -tasks move_to_ball`
# monitor $JOB $SAVE $PID

# 7-9-16 Train against a defense chaser
# JOB="Chaser_2v0"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --defense_chasers 1`
# monitor $JOB $SAVE $PID

# 7-9-16 Try pass reward
# JOB="PassReward_2v0"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2`
# monitor $JOB $SAVE $PID

# JOB="PassReward_2v0_share2layers"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 -share_actor_layers 2 -share_critic_layers 2`
# monitor $JOB $SAVE $PID

# JOB="PassReward_2v1"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --defense_npcs 1`
# monitor $JOB $SAVE $PID

# JOB="PassReward_2v1_share2layers"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --defense_npcs 1 -share_actor_layers 2 -share_critic_layers 2`
# monitor $JOB $SAVE $PID

# 6-1-16 Train 2v1 with shared replay memory
# JOB="sharedreplay_2v1"
# SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -offense_agents 2 -defense_npcs 1 -offense_on_ball 10 -ball_x_min 0.6 -share_replay_memory -verbose`
# monitor $JOB $SAVE $PID

# 6-1-16 Train weight sharing on 2v1
# values="1 2 3 4"
# for v in $values;
# do
#     JOB="shareparam_2v1_$v"
#     SAVE="/scratch/cluster/mhauskn/dqn-hfo/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -beta 0.2 -offense_agents 2 -defense_npcs 1 -offense_on_ball 10 -ball_x_min 0.6 -share_actor_layers $v -share_critic_layers $v -verbose`
#     monitor $JOB $SAVE $PID
# done

# 5-25-16 Train 2v0 with shared replay memory
# JOB="sharedreplay_2v0"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -offense_agents 2 -share_replay_memory`
# monitor $JOB $SAVE $PID

# 5-24-16 Weight Sharing Experiment
# values="1 2 3 4"
# for v in $values;
# do
#     JOB="shareparam_2v0_$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -beta 0.2 -offense_agents 2 -offense_on_ball 10 -share_actor_layers $v -share_critic_layers $v`
#     monitor $JOB $SAVE $PID
# done

# Train a 1v1 agent with an offensive dummy
# JOB="1v1"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -beta 0.2 -offense_dummies 1 -defense_npcs 1 -ball_x_min 0.6 -offense_on_ball 1`
# monitor $JOB $SAVE $PID

# ACTOR="state/1v1_agent0_HiScore0.800000_actor_iter_5920022.solverstate"
# CRITIC="state/1v1_agent0_HiScore0.800000_critic_iter_5920022.solverstate"
# JOB="pretrained_2v1"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --explore 1 --offense_agents 2 --ball_x_min 0.6 --offense_on_ball 10 -defense_npcs 1 -actor_snapshot $ACTOR,$ACTOR -critic_snapshot $CRITIC,$CRITIC`
# monitor $JOB $SAVE $PID


# 5-12-16 Combine the two independently learned agents together
# ACTOR0="state/dummy_0_agent0_actor_iter_2720000.solverstate"
# CRITIC0="state/dummy_0_agent0_critic_iter_2720000.solverstate"
# ACTOR1="state/dummy_1_agent0_actor_iter_2730000.solverstate"
# CRITIC1="state/dummy_1_agent0_critic_iter_2730000.solverstate"
# JOB="2v0_pretrained"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --offense_on_ball=10 -defense_dummies 1 -actor_snapshot $ACTOR0,$ACTOR1 -critic_snapshot $CRITIC0,$CRITIC1 -zeta_explore 1`
# monitor $JOB $SAVE $PID
# JOB="2v1_pretrained"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --offense_on_ball=10 -defense_npcs 1 -actor_snapshot $ACTOR0,$ACTOR1 -critic_snapshot $CRITIC0,$CRITIC1 -zeta_explore 1`
# monitor $JOB $SAVE $PID

# 5-10-16 Return to 2v0 with annealed intrinsic rewards
# JOB="2v0_joint"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --offense_on_ball=10`
# monitor $JOB $SAVE $PID
# # 5-10-16 Return to 2v1 with annealed intrinsic rewards
# JOB="2v1_joint"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --defense_npcs 1 --offense_on_ball=10`
# monitor $JOB $SAVE $PID
# # 5-10-16 Learning with a teammate using annealing
# JOB="teammate"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 1 --offense_npcs 1 --offense_on_ball 2`
# monitor $JOB $SAVE $PID
# # 5-10-16 Individual training with dummy agents
# values="0 1"
# for v in $values;
# do
#     JOB="dummy_$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 1 -offense_dummies 1 -defense_dummies 1`
#     monitor $JOB $SAVE $PID
# done

# 5-9-16 Testing annealing of intrinsic reward
# ==== Naive annealing
# JOB="naive_anneal"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 1 --zeta_explore 1000000`
# monitor $JOB $SAVE $PID
# ==== Extrinsic annealing
# JOB="extrinsic_anneal"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 1 --zeta_explore 5000`
# monitor $JOB $SAVE $PID
# ==== Adaptive annealing
# JOB="adaptive_anneal"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 1 --zeta_explore 50`
# monitor $JOB $SAVE $PID

# 5-6-16 Debug testing with dummy teammate and dummy goalie
# RESULT: Passed after fixing the unum assignment
# values="0 1"
# for v in $values;
# do
#     JOB="refactorunum_$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 1 -offense_dummies 1 -defense_dummies 1`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-31-16 Lets try this on 1v1!
# values="0 .2 .5 .8 1"
# for v in $values;
# do
#     JOB="Hybrid1v1_beta$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -beta $v -defense_npcs=1`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-30-16 Hybrid learning with forward sampling rather than random
# values="0 .2 .5 .8 1"
# for v in $values;
# do
#     JOB="ForwardSampling_beta$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -beta $v`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-29-16 Hybrid learning from on-policy and off-policy values
# values="0 .2 .5 .8 1"
# for v in $values;
# do
#     JOB="Hybrid_beta$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -beta $v`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-12-16 Learning with a teammate
# values="0 1"
# for v in $values;
# do
#     JOB="teammate_$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 1 --offense_npcs 1 --defense_npcs 1`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-11-16 2v1 test with naive shaped reward
# values="1 2"
# for v in $values;
# do
#     JOB="2v1_$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2 --defense_npcs 1`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-8-16 2v0 testing
# JOB="2v0"
# SAVE="state/$JOB"
# PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE --offense_agents 2`
# ACTIVE="~/public_html/exp_vis/active/"$JOB
# VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
# SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
# FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
# EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
# nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &

# 3-8-16 Regression testing on single agent learning
# values="1 2"
# for v in $values;
# do
#     JOB="sanity$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-2-16 Testing Samping of actions rather than max
# values=".05 .1"
# for v in $values;
# do
#     JOB="ActionSampling_UpdateRatio$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -update_ratio=$v -sample_actions`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-2-16 Testing Offense against a goalie
# values="1 2"
# for v in $values;
# do
#     JOB="1v1_$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -defense_npcs=1`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 3-1-16 Testing DDQN update
# values=".05 .1 .5"
# for v in $values;
# do
#     JOB="DDQN_UpdateRatio$v"
#     SAVE="state/$JOB"
#     PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -update_ratio=$v`
#     ACTIVE="~/public_html/exp_vis/active/"$JOB
#     VIS_CMD="./scripts/save.sh "$SAVE"_INFO_* $ACTIVE"
#     SUCCESS="mv $ACTIVE* ~/public_html/exp_vis/complete/"
#     FAILURE="mv $ACTIVE* ~/public_html/exp_vis/failed/"
#     EXIT_CMD="if grep termination $PREFIX.log | tail -1 | grep -q Normal; then $SUCCESS; else $FAILURE; fi;"
#     nohup monitor-condor-job --pid=$PID --do="$VIS_CMD" --every=100 --on_exit="$EXIT_CMD" >/dev/null &
# done

# 2-25-16 Again test update ratio. Hopefully no disk crashes this time...
# values=".05 .1 .5"
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
