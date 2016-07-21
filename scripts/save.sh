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

# Plot Evaluation Reward
grep "Evaluation:" $LOGS | lmj-plot -m '\[Agent0\].*actor_iter = (\d+),.*avg_reward = (\S+),.*reward_std = (\S+),.*' '\[Agent1\].*actor_iter = (\d+),.*avg_reward = (\S+),.*reward_std = (\S+),.*' --xlabel 'Iteration' --ylabel 'Average Reward' --title "$PREFIX Evaluation" -g -T $MARKERS $LEGEND -c Dark2 -f .5 -o $SAVE"_eval_reward.png" &

# Plot Evaluation Average Steps
grep "Evaluation:" $LOGS | lmj-plot -m '\[Agent0\].*actor_iter = (\d+),.*avg_steps = (\S+),.*steps_std = (\S+),.*' '\[Agent1\].*actor_iter = (\d+),.*avg_steps = (\S+),.*steps_std = (\S+),.*' --xlabel 'Iteration' --ylabel 'Average Steps' --title "$PREFIX Evaluation" -g -T $MARKERS $LEGEND -c Accent -f .5 -o $SAVE"_eval_steps.png" &

# Plot Evaluation Goal Percentage
grep "Evaluation:" $LOGS | lmj-plot -m '\[Agent0\].*actor_iter = (\d+),.*goal_perc = (\S+).*' '\[Agent1\].*actor_iter = (\d+),.*goal_perc = (\S+).*' --xlabel 'Iteration' --ylabel 'Goal Percentage' --title "$PREFIX Evaluation" -g -T $MARKERS $LEGEND -c Set3 -f .5 -o $SAVE"_eval_goal_perc.png" &

# Plot Critic Loss
grep "Critic Iteration" $LOGS | lmj-plot -m '\[Agent0\] Critic Iteration (\d+), loss = (\S+)' '\[Agent1\] Critic Iteration (\d+), loss = (\S+)' --num-x-ticks 8 --xlabel 'Iteration' --ylabel 'Critic Average Loss' --title $PREFIX -g -T --log y $MARKERS $LEGEND -c Pastel1 -o $SAVE"_loss.png" &

# Plot avg q_value
grep "Actor Iteration" $LOGS | lmj-plot -m '\[Agent0\] Actor Iteration (\d+),.* avg_q_value = (\S+).*' '\[Agent1\] Actor Iteration (\d+),.* avg_q_value = (\S+).*' --num-x-ticks 8 --xlabel 'Iteration' --ylabel 'Actor Average Q-Value' --title $PREFIX -g -T --log y $MARKERS $LEGEND -c Pastel2 -o $SAVE"_avgq.png" &

# Plot Zeta
# grep "Zeta" $LOGS | lmj-plot -m 'Zeta = (\S+)' --num-x-ticks 8 --xlabel 'Episode' --ylabel 'Zeta' --title $PREFIX -g -T $MARKERS $LEGEND -c Pastel1 -o $SAVE"_zeta.png" &
