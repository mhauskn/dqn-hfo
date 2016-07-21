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
  float kickable = current_state[12];
  float ball_proximity = current_state[53];

  // Episode ends once an agent can kick the ball
  if (kickable > 0 && !first_step_[tid]) {
    episode_over_ = true;
  }

  if (!first_step_[tid]) {
    ball_prox_delta_[tid] = ball_proximity - old_ball_prox_[tid];
  }
  old_ball_prox_[tid] = ball_proximity;
  float reward = ball_prox_delta_[tid];

  if (episodeOver()) {
    old_ball_prox_[tid] = 0;
    ball_prox_delta_[tid] = 0;
    first_step_[tid] = true;
  } else {
    first_step_[tid] = false;
  }

  barrier_.wait();
  return reward;
}
