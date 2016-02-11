#!/bin/bash

# set -e

JOB="caffenew_solver"
SAVE="state/$JOB"
PID=`cluster --gpu --prefix $SAVE ./dqn -save=$SAVE -explore=10000 -memory_threshold=1000`
