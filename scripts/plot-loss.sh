#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 log_file"
    exit
fi

lmj-plot -m 'Train set: iteration (\d+), Loss sum = (\S+)' \
         'Test set: iteration (\d+), Loss sum = (\S+)' \
         -n Train Test --xlabel Iteration --ylabel Loss --title $1 -g -T --input $1 -L

