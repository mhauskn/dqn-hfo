#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 log_file"
    exit
fi

lmj-plot -m 'train_dash_deviation : (\S+)' \
         'train_dash_deviation2 : (\S+)' \
         'train_turn_deviation : (\S+)' \
         'train_kick_deviation : (\S+)' \
         'train_kick_deviation2 : (\S+)' \
-n dash1 dash2 turn kick1 kick2 \
--xlabel Epochs --ylabel Deviation --title $1 -g -T --input $1 -L
