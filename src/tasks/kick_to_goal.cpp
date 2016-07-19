#include "task.hpp"
#include <cmath>

using namespace std;
using namespace hfo;

KickToGoal::KickToGoal(int server_port, int offense_agents, int defense_agents,
                       float ball_x_min, float ball_x_max) :
    Task("kick_to_goal", offense_agents, defense_agents),
    old_ball_dist_goal_(offense_agents + defense_agents, 0.),
    ball_dist_goal_delta_(offense_agents + defense_agents, 0.),
    first_step_(offense_agents + defense_agents, true)
{
  int player_on_ball = 100; // Random offense agent will be given the ball
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              500, ball_x_min, ball_x_max, player_on_ball);
}

float KickToGoal::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  const std::vector<float>& current_state = env.getState();
  float ball_proximity = current_state[53];
  float goal_proximity = current_state[15];
  float ball_dist = 1.0 - ball_proximity;
  float goal_dist = 1.0 - goal_proximity;
  float ball_ang_sin_rad = current_state[51];
  float ball_ang_cos_rad = current_state[52];
  float ball_ang_rad = acos(ball_ang_cos_rad);
  if (ball_ang_sin_rad < 0) { ball_ang_rad *= -1.; }
  float goal_ang_sin_rad = current_state[13];
  float goal_ang_cos_rad = current_state[14];
  float goal_ang_rad = acos(goal_ang_cos_rad);
  if (goal_ang_sin_rad < 0) { goal_ang_rad *= -1.; }
  float alpha = std::max(ball_ang_rad, goal_ang_rad)
      - std::min(ball_ang_rad, goal_ang_rad);
  float ball_dist_goal = sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
                              2.*ball_dist*goal_dist*cos(alpha));

  if (!first_step_[tid]) {
    ball_dist_goal_delta_[tid] = ball_dist_goal - old_ball_dist_goal_[tid];
  }
  old_ball_dist_goal_[tid] = ball_dist_goal;

  if (episodeOver()) {
    old_ball_dist_goal_[tid] = 0;
    ball_dist_goal_delta_[tid] = 0;
    first_step_[tid] = true;
  } else {
    first_step_[tid] = false;
  }
  barrier_.wait();
  return ball_dist_goal_delta_[tid];
}
