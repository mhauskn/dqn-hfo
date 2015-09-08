#!/bin/bash

# mimic slow/fast dash agent with separate network para
# remember to set the server cmd to be 1 offense player and no defense player
# PREFIX="state2/job"
# SAVE="state2/"
# MIMICDATA_SLOW="1v0_handcraft_all_slow_dash/agent.log"
# MIMICDATA_FAST="1v0_handcraft/agent.log"
# ACTOR_SOLVER="dqn_actor_solver3.prototxt"
# cluster --gpu --prefix $PREFIX ./dqn -save=$SAVE -mimic -mimic_data=$MIMICDATA_FAST -epochs=30 -actor_solver=$ACTOR_SOLVER


# evaluate the given agent's performance
# PREFIX="state/Handcoded_Evaluate_job"
# SAVE="state/"
# ACTOR_SOLVER="dqn_actor_solver.prototxt"
# ACTOR_WEIGHTS="train_result/8_19_all_fast_dash/_actor_iter_998341.caffemodel"
# CRITIC_WEIGHTS="train_result/8_19_all_fast_dash/_critic_iter_1.caffemodel"
# PORT=10000
# cluster --gpu --prefix $PREFIX ./dqn -evaluate -save=$SAVE -actor_weights=$ACTOR_WEIGHTS -critic_weights=$CRITIC_WEIGHTS -actor_solver=$ACTOR_SOLVER -repeat_games=200 -port=$PORT

PREFIX="state2/job"
SAVE="state2/"
ACTOR_SOLVER="dqn_actor_solver.prototxt"
ACTOR_WEIGHTS="train_result/8_14_current_model_trained_on_1v1/_actor_iter_843751.caffemodel"
CRITIC_WEIGHTS="train_result/8_14_current_model_trained_on_1v1/_critic_iter_1.caffemodel"
PORT=6000
cluster --gpu --prefix $PREFIX ./dqn -evaluate -save=$SAVE -actor_weights=$ACTOR_WEIGHTS -critic_weights=$CRITIC_WEIGHTS -actor_solver=$ACTOR_SOLVER -repeat_games=100 -port=$PORT

# PREFIX="state3/Agent2D_on_1v1_job"
# SAVE="state3/"
# ACTOR_SOLVER="train_result/8_14_current_model_trained_on_1v1/dqn_actor_solver.prototxt"
# ACTOR_WEIGHTS="train_result/8_14_current_model_trained_on_1v1/_actor_iter_843751.caffemodel"
# CRITIC_WEIGHTS="train_result/8_14_current_model_trained_on_1v1/_critic_iter_1.caffemodel"
# PORT=7000
# cluster --gpu --prefix $PREFIX ./dqn -evaluate -save=$SAVE -actor_weights=$ACTOR_WEIGHTS -critic_weights=$CRITIC_WEIGHTS -actor_solver=$ACTOR_SOLVER -repeat_games=200 -port=$PORT 
