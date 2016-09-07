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

  // If there are two agents, only the one closer to the ball is
  // rewarded for approaching the ball.
  if (getNumAgents() > 1) {
    CHECK_GE(current_state.size(), 60);
    float ball_dist = 1. - (ball_proximity+1.)/2.;
    float teammate_proximity = current_state[60];
    float teammate_dist = 1. - (teammate_proximity+1.)/2.;
    float ball_dist_teammate = getDist(
        ball_dist, current_state[51], current_state[52],
        teammate_dist, current_state[58], current_state[59]);
    bool teammate_closer_to_ball = ball_dist > ball_dist_teammate;
    VLOG(1) << "Agent" << tid << "-" << env.getUnum() << " BallDist " << ball_dist
            << " BallDistTeammate " << ball_dist_teammate;
    if (teammate_closer_to_ball) {
      // If teammate closer to ball, you get negative reward for approaching
      reward = std::min(-ball_prox_delta_[tid], 0.f);
    }
  }

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
