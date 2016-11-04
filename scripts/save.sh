#!/bin/bash
set -e

if [ $# -lt 2 ]; then
    echo "usage: $0 log_files save_prefix"
    exit
fi

LAST=""
LOGS=""
SAVE=""
for var in "$@"
do
    LOGS+="$SAVE "
    SAVE="$LAST "
    LAST="$var"
done
LOGS+="$SAVE "
SAVE="$LAST"
LAST="$var "
PREFIX="${SAVE##*/}"

# Determine the number of agents playing the game
AGENTS=0
while true;
do
    if grep -q Agent$AGENTS $LOGS; then
        AGENTS=$((AGENTS+1))
    else
        break;
    fi
done

# Determine the tasks being performed
TASKS_STR=`grep "Adding Task" $LOGS | awk 'NF>1{print $NF}' | sort -u`
TASKS=( $TASKS_STR )

MARKERS="-p o- v-"
COLOR="-c Set2"
# Only add a legend if we have two or more agents
LEGEND="--legend br"
if [ "$AGENTS" -lt 2 ]; then
    LEGEND=""
    MARKERS="-p o-"
fi

# Plot Reward
grep "Episode [0-9]*, reward" $LOGS | lmj-plot -m '\[Agent0\] .* reward = (\S+),.*' '\[Agent1\] .* reward = (\S+),.*' --xlabel Episode --ylabel Reward --title $PREFIX -g -T $MARKERS $LEGEND -c Dark2 -o $SAVE"_reward.png" &

# Plot Evaluation Average Steps
grep "Evaluation:" $LOGS | lmj-plot -m '\[Agent0\].*actor_iter = (\d+),.*avg_steps = (\S+),.*steps_std = (\S+),.*' '\[Agent1\].*actor_iter = (\d+),.*avg_steps = (\S+),.*steps_std = (\S+),.*' --xlabel 'Iteration' --ylabel 'Average Steps' --title "$PREFIX Evaluation" -g -T $MARKERS $LEGEND -c Accent -f .5 -o $SAVE"_eval_steps.png" &

# Plot Evaluation Goal Percentage
grep "Evaluation:" $LOGS | lmj-plot -m '\[Agent0\].*actor_iter = (\d+),.*goal_perc = (\S+),.*' '\[Agent1\].*actor_iter = (\d+),.*goal_perc = (\S+),.*' --xlabel 'Iteration' --ylabel 'Goal Percentage' --title "$PREFIX Evaluation" -g -T $MARKERS $LEGEND -c Set3 -f .5 -o $SAVE"_eval_goal_perc.png" &

# Plot Critic Loss
grep "Critic Iteration" $LOGS | lmj-plot -m '\[Agent0\] Critic Iteration (\d+), loss = (\S+)' '\[Agent1\] Critic Iteration (\d+), loss = (\S+)' --num-x-ticks 8 --xlabel 'Iteration' --ylabel 'Critic Average Loss' --title $PREFIX -g -T --log y $MARKERS $LEGEND -c Pastel1 -o $SAVE"_loss.png" &

# Plot Semantic Loss
grep "Semantic Iteration" $LOGS | lmj-plot -m '\[Agent0\] Semantic Iteration (\d+), avg_loss = (\S+)' '\[Agent1\] Semantic Iteration (\d+), avg_loss = (\S+)' --num-x-ticks 8 --xlabel 'Iteration' --ylabel 'Semantic Average Loss' --title $PREFIX -g -T --log y $MARKERS $LEGEND -c Pastel1 -o $SAVE"_semantic_loss.png" &

# Plot avg q_value
grep "Actor Iteration" $LOGS | lmj-plot -m '\[Agent0\] Actor Iteration (\d+),.* avg_q_value = (\S+).*' '\[Agent1\] Actor Iteration (\d+),.* avg_q_value = (\S+).*' --num-x-ticks 8 --xlabel 'Iteration' --ylabel 'Actor Average Q-Value' --title $PREFIX -g -T --log y $MARKERS $LEGEND -c Pastel2 -o $SAVE"_avgq.png" &

if [[ ( "${#TASKS[@]}" -gt 1 ) && ( "$AGENTS" -eq 1 ) ]]
then
    echo "Single agent, many tasks"
    # Plot Eval Performance for across tasks
    for i in `seq 0 $(($AGENTS-1))`;
    do
        PLT="grep 'Evaluation:' $LOGS | lmj-plot -m "
        for task in "${TASKS[@]}";
        do
            PLT+="'\[Agent$i\].*actor_iter = (\d+),.*task = $task, performance = (\S+)' "
        done
        PLT+="--xlabel 'Iteration' --ylabel 'Performance' --title '$PREFIX Agent$i EvalPerf' -g -T $MARKERS --legend br -n $TASKS_STR -c Set3 -f .5 -o ${SAVE}_agent${i}_eval_perf.png &"
        echo "Plot Command: $PLT"
        eval $PLT
    done
fi

if [ "${#TASKS[@]}" -gt 1 ]
then
    # Plot task_id as a function of episode
    grep "Episode [0-9]*, reward" $LOGS | lmj-plot -m '\[Agent0\] .* id = (\d+)' --xlabel Episode --ylabel TaskID --title $PREFIX -g -T $MARKERS $LEGEND -c Dark2 -o $SAVE"_taskID.png" &
fi

if [ "$AGENTS" -gt 1 ]
then
    echo "Single Task, potentially many agents"
    # Plot Eval Reward across agents for single task
    for task in "${TASKS[@]}";
    do
        PLT="grep 'Evaluation:.*task = $task' $LOGS | lmj-plot -m "
        for i in `seq 0 $(($AGENTS-1))`;
        do
            PLT+="'\[Agent$i\].*actor_iter = (\d+),.*avg_reward = (\S+),.*reward_std = (\S+),.*' "
        done
        PLT+="--xlabel 'Iteration' --ylabel 'Average Reward' --title '$PREFIX $task EvalReward' -g -T $MARKERS $LEGEND -c Dark2 -f .5 -o ${SAVE}_${task}_eval_reward.png &"
        eval $PLT
    done

    # Plot Eval Performance across agents for single task
    for task in "${TASKS[@]}";
    do
        PLT="grep 'Evaluation:.*task = $task' $LOGS | lmj-plot -m "
        for i in `seq 0 $(($AGENTS-1))`;
        do
            PLT+="'\[Agent$i\].*actor_iter = (\d+),.*task = $task, performance = (\S+)' "
        done
        PLT+="--xlabel 'Iteration' --ylabel 'Performance' --title '$PREFIX $task EvalPerf' -g -T $MARKERS $LEGEND -c Set2 -f .5 -o ${SAVE}_${task}_eval_perf.png &"
        eval $PLT
    done
fi
