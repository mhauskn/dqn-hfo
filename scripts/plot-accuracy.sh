#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 log_file"
    exit
fi

lmj-plot -m 'Training Set: Iteration (\d+), Accuracy (\S+)' \
         'Test Set: Iteration (\d+), Accuracy (\S+)' \
         'Iteration (\d+), DASH Accuracy: (\S+)' \
         'Iteration (\d+), TURN Accuracy: (\S+)' \
         'Iteration (\d+), KICK Accuracy: (\S+)' \
-n Train Test Dash_Test Turn_Test Kick_Test \
--xlabel Iteration --ylabel Accuracy --title $1 -g -T --input $1 -L
