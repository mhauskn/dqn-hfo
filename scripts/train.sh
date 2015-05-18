#!/bin/bash

# set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 savepath"
    exit
fi

LOGNAME=$1

echo "==> Run server"
/u/chen/robocup_libs/HFO/bin/start.py --offense 1 --defense 0 --headless &
sleep 1
echo "==> Run DQN"
./dqn -save $LOGNAME -memory_threshold 1
sleep 1
echo "==> Done Killing Server!"
killall -9 rcssserver
