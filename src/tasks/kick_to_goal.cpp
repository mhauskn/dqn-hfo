#include "task.hpp"
#include <cmath>

using namespace std;
using namespace hfo;

KickToGoal::KickToGoal(int server_port, int offense_agents, int defense_agents,
                       float ball_x_min, float ball_x_max) :
    Task(taskName(), offense_agents, defense_agents),
    old_ball_dist_goal_(offense_agents + defense_agents, 0.),
    first_step_(offense_agents + defense_agents, true)
{
  int player_on_ball = 100; // Random offense agent will be given the ball
  int max_steps = 60;
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              max_steps, ball_x_min, ball_x_max, player_on_ball);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float KickToGoal::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  Player pob = env.playerOnBall();
  const std::vector<float>& current_state = env.getState();
  float ball_proximity = current_state[53];
  float goal_proximity = current_state[15];
  float ball_dist = 1. - (ball_proximity+1.)/2.;
  float goal_dist = 1. - (goal_proximity+1.)/2.;
  float ball_dist_goal = getDist(
      ball_dist, current_state[51], current_state[52],
      goal_dist, current_state[13], current_state[14]);
  CHECK(std::isfinite(ball_dist_goal))
      << "ball_dist=" << ball_dist
      << " goal_dist=" << goal_dist
      << " current_state[51]=" << current_state[51]
      << " current_state[52]=" << current_state[52]
      << " current_state[13]=" << current_state[13]
      << " current_state[14]=" << current_state[14];
  float ball_dist_goal_delta = 0;
  if (!first_step_[tid]) {
    ball_dist_goal_delta = ball_dist_goal - old_ball_dist_goal_[tid];
  }
  float reward = 0;
  if (pob.unum == env.getUnum()) {
    reward = -ball_dist_goal_delta;
  }
  old_ball_dist_goal_[tid] = ball_dist_goal;
  if (episodeOver()) {
    old_ball_dist_goal_[tid] = 0;
    first_step_[tid] = true;
    reward = 0;
  } else {
    first_step_[tid] = false;
  }
  barrier_.wait();
  return reward;
}
