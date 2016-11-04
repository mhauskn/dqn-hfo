#include "task.hpp"

using namespace std;
using namespace hfo;

MoveToBall::MoveToBall(int server_port, int offense_agents, int defense_agents,
                       float ball_x_min, float ball_x_max) :
    Task(taskName(), offense_agents, defense_agents),
    old_ball_prox_(offense_agents + defense_agents, 0.),
    ball_prox_delta_(offense_agents + defense_agents, 0.),
    first_step_(offense_agents + defense_agents, true)
{
  int max_steps = 100;
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              max_steps, ball_x_min, ball_x_max);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float MoveToBall::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  const std::vector<float>& current_state = env.getState();
  bool kickable = current_state[12] > 0;
  float ball_proximity = current_state[53];

  // Episode ends once an agent can kick the ball
  if (kickable && !first_step_[tid]) {
    episode_over_ = true;
  }

  if (!first_step_[tid]) {
    ball_prox_delta_[tid] = ball_proximity - old_ball_prox_[tid];
  }
  float reward = ball_prox_delta_[tid];

  old_ball_prox_[tid] = ball_proximity;
  if (episodeOver()) {
    old_ball_prox_[tid] = 0;
    ball_prox_delta_[tid] = 0;
    first_step_[tid] = true;
    reward = 0;
  } else {
    first_step_[tid] = false;
  }

  barrier_.wait();
  return reward;
}

MoveAwayFromBall::MoveAwayFromBall(int server_port, int offense_agents, int defense_agents,
                                   float ball_x_min, float ball_x_max) :
    Task(taskName(), offense_agents, defense_agents),
    old_ball_prox_(offense_agents + defense_agents, 0.),
    first_step_(offense_agents + defense_agents, true)
{
  int max_steps = 100;
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              max_steps, ball_x_min, ball_x_max);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float MoveAwayFromBall::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  const std::vector<float>& current_state = env.getState();
  float ball_proximity = (current_state[53]+1.)/2.;
  float ball_prox_delta = 0;
  if (!first_step_[tid]) {
    ball_prox_delta = ball_proximity - old_ball_prox_[tid];
  }
  float reward = -ball_prox_delta;
  old_ball_prox_[tid] = ball_proximity;
  if (episodeOver()) {
    reward = 0;
    old_ball_prox_[tid] = 0;
    first_step_[tid] = true;
  } else {
    first_step_[tid] = false;
  }
  barrier_.wait();
  return reward;
}

BlindMoveToBall::BlindMoveToBall(int server_port, int offense_agents, int defense_agents,
                                 float ball_x_min, float ball_x_max) :
    Task(taskName(), offense_agents, defense_agents),
    old_ball_prox_(offense_agents + defense_agents, 0.),
    ball_prox_delta_(offense_agents + defense_agents, 0.),
    first_step_(offense_agents + defense_agents, true),
    blind_reward_(2e10)
{
  CHECK_EQ(offense_agents, 2) << "BlindSoccer requires 2 agents.";
  int max_steps = 100;
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              max_steps, ball_x_min, ball_x_max);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

const std::vector<float>& BlindMoveToBall::getState(int tid) {
  CHECK_GT(envs_.size(), tid);
  if (tid == 0) { // Blind agent can't see anything
    dummy_state_ = envs_[tid].getState();
    for (int i = 0; i < dummy_state_.size(); ++i) {
      dummy_state_[i] = 0.;
    }
    return dummy_state_;
  } else {
    return envs_[tid].getState();
  }
}

void BlindMoveToBall::act(int tid, hfo::action_t action, float arg1, float arg2) {
  CHECK_GT(envs_.size(), tid);

  // Sighted agent can't Dash
  if (tid == 1 && action == DASH) {
    envs_[tid].act(action, 0.f, 0.f);
    return;
  }

  envs_[tid].act(action, arg1, arg2);
}

float BlindMoveToBall::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  const std::vector<float>& current_state = env.getState();
  bool kickable = current_state[12] > 0;
  float ball_proximity = current_state[53];
  float ball_prox_delta = ball_proximity - old_ball_prox_[tid];

  // Episode ends once an agent can kick the ball
  if (kickable && !first_step_[tid]) {
    episode_over_ = true;
  }

  float reward = 0;
  if (tid == 0) { // Blind agent
    if (!first_step_[tid]) {
      reward = ball_prox_delta;
    }
    blind_reward_ = reward;
  } else { // Non-blind agent is rewarded as much as blind agent is
    while (blind_reward_ >= 2e10) {
      // Wait for blind agent to get a reward
      little_sleep(std::chrono::microseconds(100));
    }
    reward = blind_reward_;
    blind_reward_ = 2e10;
  }

  old_ball_prox_[tid] = ball_proximity;

  if (episodeOver()) {
    old_ball_prox_[tid] = 0;
    ball_prox_delta_[tid] = 0;
    first_step_[tid] = true;
    //reward = 0;
  } else {
    first_step_[tid] = false;
  }

  barrier_.wait();
  return reward;
}
