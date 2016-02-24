#include <glog/logging.h>
#include <gflags/gflags.h>
#include "hfo_game.hpp"
#include <time.h>
#include <cmath>

using namespace hfo;

DEFINE_int32(offense_agents, 1, "Number of agents playing offense");
DEFINE_int32(offense_npcs, 0, "Number of npcs playing offense");
DEFINE_int32(defense_agents, 0, "Number of agents playing defense");
DEFINE_int32(defense_npcs, 0, "Number of npcs playing defense");
DEFINE_string(server_cmd, "./scripts/HFO --fullstate --frames-per-trial 500",
              "Command executed to start the HFO server.");
DEFINE_string(config_dir, "/u/mhauskn/projects/HFO/bin/teams/base/config/formations-dt",
              "Directory containing HFO config files.");
DEFINE_bool(gui, false, "Open a GUI window.");
DEFINE_bool(log_game, false, "Log the HFO game.");

int NumStateFeatures() {
  return 50 + 8 * (FLAGS_offense_agents + FLAGS_defense_npcs +
                   FLAGS_offense_npcs + FLAGS_defense_agents);
}

HFOEnvironment CreateHFOEnvironment(int port, int unum) {
  std::string cmd = FLAGS_server_cmd + " --port " + std::to_string(port)
      + " --offense-agents " + std::to_string(FLAGS_offense_agents)
      + " --offense-npcs " + std::to_string(FLAGS_offense_npcs)
      + " --defense-agents " + std::to_string(FLAGS_defense_agents)
      + " --defense-npcs " + std::to_string(FLAGS_defense_npcs);
  if (!FLAGS_gui) { cmd += " --headless"; }
  if (!FLAGS_log_game) { cmd += " --no-logging"; }
  cmd += " &";
  LOG(INFO) << "Starting server with command: " << cmd;
  CHECK_EQ(system(cmd.c_str()), 0) << "Unable to start the HFO server.";
  sleep(10);
  HFOEnvironment hfo_env;
  hfo_env.connectToServer(LOW_LEVEL_FEATURE_SET, FLAGS_config_dir, unum,
                          port, "localhost", "base_left", false);
  return hfo_env;
}

Action GetRandomHFOAction(std::mt19937& random_engine) {
  action_t action_indx = (action_t) std::uniform_int_distribution<int>(DASH, KICK)(random_engine);
  float arg1, arg2;
  switch (action_indx) {
    case DASH:
      arg1 = std::uniform_real_distribution<float>(-100.0, 100.0)(random_engine);
      arg2 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      break;
    case TURN:
      arg1 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      arg2 = 0;
      break;
    case TACKLE:
      arg1 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      arg2 = 0;
      break;
    case KICK:
      arg1 = std::uniform_real_distribution<float>(0.0, 100.0)(random_engine);
      arg2 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      break;
    default:
      LOG(FATAL) << "Invalid Action Index: " << action_indx;
      break;
  }
  Action act = {action_indx, arg1, arg2};
  return act;
}

HFOGameState::HFOGameState(HFOEnvironment& hfo) :
    hfo(hfo), old_ball_prox(0), ball_prox_delta(0), old_kickable(0),
    kickable_delta(0), old_ball_dist_goal(0), ball_dist_goal_delta(0),
    steps(0), total_reward(0), status(IN_GAME),
    episode_over(false), got_kickable_reward(false) {
  VLOG(1) << "Creating new HFOGameState";
}

HFOGameState::~HFOGameState() {
  VLOG(1) << "Destroying HFOGameState";
  while (status == IN_GAME) {
    hfo.act(DASH, 0, 0);
    status = hfo.step();
  }
}

void HFOGameState::update(const std::vector<float>& current_state,
                          status_t current_status) {
  if (status == SERVER_DOWN) {
    LOG(FATAL) << "Server Down!";
  }
  status = current_status;
  if (status != IN_GAME) {
    episode_over = true;
  }
  float ball_proximity = current_state[53];
  float goal_proximity = current_state[15];
  float ball_dist = 1.0 - ball_proximity;
  float goal_dist = 1.0 - goal_proximity;
  float kickable = current_state[12];
  float ball_ang_sin_rad = current_state[51];
  float ball_ang_cos_rad = current_state[52];
  float ball_ang_rad = acos(ball_ang_cos_rad);
  if (ball_ang_sin_rad < 0) { ball_ang_rad *= -1.; }
  float goal_ang_sin_rad = current_state[13];
  float goal_ang_cos_rad = current_state[14];
  float goal_ang_rad = acos(goal_ang_cos_rad);
  if (goal_ang_sin_rad < 0) { goal_ang_rad *= -1.; }
  float alpha = std::max(ball_ang_rad, goal_ang_rad) - std::min(ball_ang_rad, goal_ang_rad);
  // By law of cosines. Alpha is angle between ball and goal
  float ball_dist_goal = sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
                              2.*ball_dist*goal_dist*cos(alpha));
  VLOG(1) << "BallProx: " << ball_proximity << " BallDistGoal: " << ball_dist_goal;
  if (steps > 0) {
    ball_prox_delta = ball_proximity - old_ball_prox;
    kickable_delta = kickable - old_kickable;
    ball_dist_goal_delta = ball_dist_goal - old_ball_dist_goal;
  }
  old_ball_prox = ball_proximity;
  old_kickable = kickable;
  old_ball_dist_goal = ball_dist_goal;
  if (episode_over) {
    ball_prox_delta = 0;
    kickable_delta = 0;
    ball_dist_goal_delta = 0;
  }
  steps++;
}

float HFOGameState::reward() {
  float moveToBallReward = move_to_ball_reward();
  float kickToGoalReward = 3. * kick_to_goal_reward();
  float eotReward = 5. * EOT_reward();
  float reward = moveToBallReward + kickToGoalReward + eotReward;
  total_reward += reward;
  VLOG(1) << "Overall_Reward: " << reward << " MTB: " << moveToBallReward
          << " KTG: " << kickToGoalReward << " EOT: " << eotReward;
  return reward;
}

// Reward for moving to ball and getting kickable. Ends episode once
// kickable is attained.
float HFOGameState::move_to_ball_reward() {
  float reward = ball_prox_delta;
  if (kickable_delta >= 1 && !got_kickable_reward) {
    reward += 1.0;
    // episode_over = true;
    got_kickable_reward = true;
  }
  return reward;
}

// Reward for kicking ball towards the goal
float HFOGameState::kick_to_goal_reward() {
  return -ball_dist_goal_delta;
}

float HFOGameState::EOT_reward() {
  if (status == GOAL) {
    VLOG(1) << "GOAL"; return 1;
  }
  // else if (status == CAPTURED_BY_DEFENSE || status == OUT_OF_BOUNDS ||
  //            status == OUT_OF_TIME) {
  //   VLOG(1) << "FAIL"; return -1;
  // }
  return 0;
}
