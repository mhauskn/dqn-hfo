#include "task.hpp"

using namespace std;
using namespace hfo;

DEFINE_bool(gui, false, "Open a GUI window.");
DEFINE_bool(log_game, false, "Log the HFO game.");
DEFINE_bool(verbose, false, "Server prints verbose output.");
DEFINE_int32(message_size, 1000, "Max size of messages.");

bool FINISHED = false;

void ExecuteCommand(string cmd) {
  CHECK_EQ(system(cmd.c_str()), 0) << "Failed to execute command: " << cmd;
}

float Task::getDist(float o1_dist, float o1_ang_sin, float o1_ang_cos,
                    float o2_dist, float o2_ang_sin, float o2_ang_cos) {
  float o1_ang_rad = acos(o1_ang_cos);
  float o2_ang_rad = acos(o2_ang_cos);
  if (o1_ang_sin < 0) {
    o1_ang_rad *= -1.;
  }
  if (o2_ang_sin < 0) {
    o2_ang_rad *= -1.;
  }
  float alpha = std::max(o1_ang_rad, o2_ang_rad) - std::min(o1_ang_rad, o2_ang_rad);
  float o1_dist_o2 = sqrt(o1_dist * o1_dist + o2_dist * o2_dist
                          - 2. * o1_dist * o2_dist * cos(alpha));
  if (!std::isfinite(o1_dist_o2)) {
    LOG(ERROR) << "ERROR: getDist not finite! o1_dist " << o1_dist << " o2_dist " << o2_dist
               << " alpha " << alpha << " o1_ang_rad " << o1_ang_rad
               << " o2_ang_rad " << o2_ang_rad;
    return 0;
  }
  return o1_dist_o2;
}

Task::Task(std::string task_name, int offense_agents, int defense_agents) :
    task_name_(task_name),
    status_(offense_agents + defense_agents, IN_GAME),
    mutexs_(offense_agents + defense_agents),
    cvs_(offense_agents + defense_agents),
    need_to_step_(offense_agents + defense_agents, false),
    episode_reward_(offense_agents + defense_agents, 0),
    last_episode_reward_(offense_agents + defense_agents, 0),
    offense_agents_(offense_agents),
    defense_agents_(defense_agents),
    server_port_(-1),
    episode_over_(false),
    barrier_(offense_agents + defense_agents)
{
  envs_.resize(offense_agents + defense_agents);
}

Task::~Task() {
  FINISHED = true;
  for (int i=0; i<envs_.size(); ++i) {
    envs_[i].act(QUIT);
    stepThread(i);
  }
  for (std::thread& t : threads_) {
    t.join();
  }
}

void Task::act(int tid, action_t action, float arg1, float arg2) {
  CHECK_GT(envs_.size(), tid);
  envs_[tid].act(action, arg1, arg2);
}

void Task::say(int tid, const string& message) {
  CHECK_GT(envs_.size(), tid);
  envs_[tid].say(message);
}

pair<status_t, float> Task::step(int tid) {
  CHECK_GT(envs_.size(), tid);
  status_t old_status = status_[tid];
  if (episode_over_ && old_status == IN_GAME) {
    stepUntilEpisodeEnd(tid);
    old_status = status_[tid];
  }
  stepThread(tid);
  status_t current_status = status_[tid];
  if (current_status == SERVER_DOWN) {
    LOG(FATAL) << "Server Down! Exiting.";
    exit(1);
  }
  if (episode_over_ && old_status != IN_GAME && current_status == IN_GAME) {
    episode_over_ = false;
    for (int i=0; i<envs_.size(); ++i) {
      episode_reward_[i] = 0;
    }
  }
  if (current_status != IN_GAME) {
    episode_over_ = true;
  }
  float reward = getReward(tid);
  CHECK(std::isfinite(reward)) << "Reward not finite! Task=" << getName();
  episode_reward_[tid] += reward;
  if (episode_over_) {
    for (int i=0; i<envs_.size(); ++i) {
      last_episode_reward_[i] = episode_reward_[i];
    }
  }
  return make_pair(current_status, reward);
}

status_t Task::stepUntilEpisodeEnd(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  while (status_[tid] == IN_GAME) {
    env.act(NOOP);
    stepThread(tid);
    if (status_[tid] == SERVER_DOWN) {
      LOG(FATAL) << "Server Down! Exiting.";
      exit(1);
    }
  }
  return status_[tid];
}

status_t Task::getStatus(int tid) const {
  CHECK_GT(status_.size(), tid);
  return status_[tid];
}

void Task::startServer(int port, int offense_agents, int offense_npcs,
                       int defense_agents, int defense_npcs, bool fullstate,
                       int frames_per_trial, float ball_x_min, float ball_x_max,
                       int offense_on_ball) {
  server_port_ = port;
  std::string cmd = "./bin/HFO";
  cmd += " --port " + std::to_string(port)
      + " --frames-per-trial " + std::to_string(frames_per_trial)
      + " --offense-agents " + std::to_string(offense_agents)
      + " --offense-npcs " + std::to_string(offense_npcs)
      + " --defense-agents " + std::to_string(defense_agents)
      + " --defense-npcs " + std::to_string(defense_npcs)
      + " --ball-x-min " + std::to_string(ball_x_min)
      + " --ball-x-max " + std::to_string(ball_x_max)
      + " --offense-on-ball " + std::to_string(offense_on_ball)
      + " --message-size " + std::to_string(FLAGS_message_size)
      + " --log-dir log/" + task_name_;
  if (fullstate) { cmd += " --fullstate"; }
  if (!FLAGS_gui)      { cmd += " --headless"; }
  if (!FLAGS_log_game) { cmd += " --no-logging"; }
  if (FLAGS_verbose)   { cmd += " --verbose"; }
  LOG(INFO) << "Starting server with command: " << cmd;
  threads_.emplace_back(std::thread(ExecuteCommand, cmd));
  sleep(10);
}

void Task::stepThread(int tid) {
  mutexs_[tid].lock();
  need_to_step_[tid] = true;
  cvs_[tid].notify_one();
  mutexs_[tid].unlock();
  while (need_to_step_[tid]) {
    std::this_thread::yield();
  }
}

void EnvConnect(HFOEnvironment* env, int server_port, string team_name,
                bool play_goalie, status_t* status_ptr,
                mutex* mtx, condition_variable* cv, int* need_to_step) {
  env->connectToServer(LOW_LEVEL_FEATURE_SET, "bin/formations-dt", server_port,
                       "localhost", team_name, play_goalie);
  while (!FINISHED) {
    std::unique_lock<std::mutex> lck(*mtx);
    while (!(*need_to_step)) {
      cv->wait(lck);
    }
    *status_ptr = env->step();
    *need_to_step = false;
  }
}

void Task::connectToServer(int tid) {
  CHECK_GT(envs_.size(), tid);
  CHECK_GT(server_port_, 0);
  if (tid < offense_agents_) {
    threads_.emplace_back(EnvConnect, &envs_[tid], server_port_, "base_left",
                          false, &status_[tid],
                          &mutexs_[tid], &cvs_[tid], &need_to_step_[tid]);
  } else {
    bool play_goalie = tid == offense_agents_;
    threads_.emplace_back(EnvConnect, &envs_[tid], server_port_, "base_right",
                          play_goalie, &status_[tid],
                          &mutexs_[tid], &cvs_[tid], &need_to_step_[tid]);
  }
}

HFOEnvironment& Task::getEnv(int tid) {
  CHECK_GT(envs_.size(), tid);
  return envs_[tid];
}


Soccer::Soccer(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents)
{
  int max_steps = 500;
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              max_steps);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float Soccer::getReward(int tid) {
  if (status_[tid] == GOAL) {
    return 1;
  }
  return 0;
}

Soccer1v1::Soccer1v1(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents)
{
  int max_steps = 500;
  CHECK_EQ(offense_agents, 1);
  CHECK_EQ(defense_agents, 0);
  int defense_npcs = 1;
  startServer(server_port, offense_agents, 0, defense_agents, defense_npcs, true,
              max_steps);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float Soccer1v1::getReward(int tid) {
  if (status_[tid] == GOAL) {
    return 1;
  }
  return 0;
}

Soccer2v1::Soccer2v1(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents),
    old_pob_(offense_agents + defense_agents)
{
  int max_steps = 500;
  CHECK_EQ(offense_agents, 2);
  CHECK_EQ(defense_agents, 0);
  int defense_npcs = 1;
  startServer(server_port, offense_agents, 0, defense_agents, defense_npcs, true,
              max_steps);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float Soccer2v1::getReward(int tid) {
  HFOEnvironment& env = envs_[tid];
  Player pob = env.playerOnBall();
  float reward = 0;
  if (status_[tid] == GOAL) {
    if (old_pob_[tid].unum == env.getUnum()) {
      reward = 1.;
    } else {
      reward = .5;
    }
  }
  old_pob_[tid] = pob;
  return reward;
}

SoccerEasy::SoccerEasy(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents),
    first_step_(offense_agents + defense_agents, true),
    old_ball_prox_(offense_agents + defense_agents, 0.),
    old_kickable_(offense_agents + defense_agents, false),
    old_ball_dist_goal_(offense_agents + defense_agents, 0.),
    got_kickable_reward_(offense_agents + defense_agents, false),
    old_pob_(offense_agents + defense_agents)
{
  int max_steps = 500;
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              max_steps);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float SoccerEasy::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  const std::vector<float>& current_state = env.getState();
  Player pob = env.playerOnBall();
  bool kickable = current_state[12] > 0;
  float ball_proximity = current_state[53];
  float ball_dist = 1. - (ball_proximity+1.)/2.;
  float goal_proximity = current_state[15];
  float goal_dist = 1. - (goal_proximity+1.)/2.;
  float ball_dist_goal = getDist(
      ball_dist, current_state[51], current_state[52],
      goal_dist, current_state[13], current_state[14]);

  float ball_prox_delta = 0;
  float kickable_delta = 0;
  float ball_dist_goal_delta = 0;
  if (!first_step_[tid]) {
    ball_prox_delta = ball_proximity - old_ball_prox_[tid];
    kickable_delta = kickable - old_kickable_[tid];
    ball_dist_goal_delta = ball_dist_goal - old_ball_dist_goal_[tid];
  }

  float reward = 0;
  if (!episodeOver()) {
    // Move to ball reward
    if (pob.unum < 0 || pob.unum == env.getUnum()) {
      VLOG(1) << "Unum-" << env.getUnum() << " MoveToBallReward: " << ball_prox_delta;
      reward += ball_prox_delta;
    }
    // Kickable reward
    if (kickable && !old_kickable_[tid] && !got_kickable_reward_[tid]) {
      VLOG(1) << "Unum-" << env.getUnum() << " KickableReward: 1";
      reward += 1.0;
      got_kickable_reward_[tid] = true;
    }
    // Kick to goal reward
    if (pob.unum == env.getUnum()) {
      VLOG(1) << "Unum-" << env.getUnum() << " KickToGoalReward: " << (3. * -ball_dist_goal_delta);
      reward -= 3. * ball_dist_goal_delta;
    }
  }
  // Goal Reward
  if (status_[tid] == GOAL) {
    if (old_pob_[tid].unum == env.getUnum()) {
      VLOG(1) << "Unum-" << env.getUnum() << " GoalReward: 5";
      reward = 5.;
    } else {
      VLOG(1) << "Unum-" << env.getUnum() << " TeammateGoalReward: 1";
      reward = 1.;
    }
  }

  old_ball_prox_[tid] = ball_proximity;
  old_kickable_[tid] = kickable;
  old_ball_dist_goal_[tid] = ball_dist_goal;
  old_pob_[tid] = pob;

  if (episodeOver()) {
    first_step_[tid] = true;
    got_kickable_reward_[tid] = false;
  } else {
    first_step_[tid] = false;
  }
  barrier_.wait();
  return reward;
}

Dribble::Dribble(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents)
{
  int offense_on_ball = 1;
  startServer(server_port, offense_agents, 0, defense_agents, 0, true,
              500, 0, 0.2, offense_on_ball);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float Dribble::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  const std::vector<float>& current_state = env.getState();
  float self_vel_mag = current_state[4];
  float ball_proximity = current_state[53];
  float ball_vel_mag = current_state[55];

  float reward = 0;
  if (ball_proximity >= .94 && ball_vel_mag >= -.8) {
    reward += .1;
  }
  // Don't go out of bounds or lose control of the ball
  status_t s = status_[tid];
  if (s == GOAL || s == OUT_OF_BOUNDS || s == CAPTURED_BY_DEFENSE) {
    // Lose half of the total accrued reward for going out of bounds
    reward -= std::min(1., 0.5 * episode_reward_[tid]);
  }
  barrier_.wait();
  return reward;
}

Pass::Pass(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents),
    pass_active_(offense_agents + defense_agents, false),
    kicker_(offense_agents + defense_agents),
    old_ball_prox_(offense_agents + defense_agents, 0.),
    old_teammate_prox_(offense_agents + defense_agents, 0.),
    old_ball_dist_teammate_(offense_agents + defense_agents, 0.),
    got_kickable_reward_(offense_agents + defense_agents, false)
{
  CHECK_LE(offense_agents, 2);
  int offense_on_ball = 100; // Randomize who gets the ball
  startServer(server_port, 2, 0, defense_agents, 0, true,
              500, 0.5, 0.5, offense_on_ball);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
  // Start a passing teammate
  if (offense_agents == 1) {
    string cmd = "./bin/passer " + std::to_string(server_port) + " base_left false";
    threads_.emplace_back(ExecuteCommand, cmd);
  }
}

float Pass::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  Player pob = env.playerOnBall();
  const std::vector<float>& current_state = env.getState();
  CHECK_GT(current_state.size(), 64) << "Unexpected number of features.";
  float self_vel_mag = (current_state[4]+1.)/2.;
  bool kickable = current_state[12] > 0;
  float ball_proximity = current_state[53];
  float ball_dist = 1. - (ball_proximity+1.)/2.;
  float ball_vel_mag = current_state[55];
  float teammate_proximity = current_state[60];
  float teammate_dist = 1. - (teammate_proximity+1.)/2.;
  float ball_dist_teammate = getDist(
      ball_dist, current_state[51], current_state[52],
      teammate_dist, current_state[58], current_state[59]);
  float ball_dist_teammate_delta = ball_dist_teammate - old_ball_dist_teammate_[tid];
  float ball_prox_delta_ = ball_proximity - old_ball_prox_[tid];
  float teammate_prox_delta_ = teammate_proximity - old_teammate_prox_[tid];

  float reward = 0;

  // Ball has stopped moving or control has switched
  if (pass_active_[tid]) {
    if (pob.unum != kicker_[tid].unum && ball_vel_mag <= -.8) {
      reward += 1.;
      VLOG(1) << "Pass Inactive: Ball Caught! Reward 1";
      pass_active_[tid] = false;
    } else if (ball_vel_mag <= -.8 && (ball_dist < ball_dist_teammate)) {
      reward -= .5;
      VLOG(1) << "Pass Inactive: Ball too slow. Reward -.5";
      pass_active_[tid] = false;
    }
  }

  // Detect a pass and record who kicked
  if (!pass_active_[tid] && ball_vel_mag >= -.6) {
    if (pob.unum == env.getUnum()) {
      reward += 1.;
      VLOG(1) << "Kick Reward 1";
    }
    pass_active_[tid] = true;
    kicker_[tid] = pob;
    VLOG(1) << "Pass Active. Kicker " << pob.unum;
  }

  // Rewarded for going to ball when pass is inactive and we are closer
  if (ball_dist < ball_dist_teammate && !pass_active_[tid]) {
    reward += ball_prox_delta_;
    VLOG(1) << "reward += ball_prox_delta_ " << ball_prox_delta_;
  }

  old_ball_prox_[tid] = ball_proximity;
  old_teammate_prox_[tid] = teammate_proximity;
  old_ball_dist_teammate_[tid] = ball_dist_teammate;
  status_t s = status_[tid];

  // Lose half of the accrued reward for ball out of bounds
  if (s == GOAL || s == OUT_OF_BOUNDS || s == CAPTURED_BY_DEFENSE) {
    reward = -std::max(0., 0.5 * episode_reward_[tid]);
    VLOG(1) << "reward = OOB " << -std::max(0., 0.5 * episode_reward_[tid]);
  }
  if (episodeOver()) {
    pass_active_[tid] = false;
    old_ball_dist_teammate_[tid] = 0;
    old_ball_prox_[tid] = 0;
    old_teammate_prox_[tid] = 0;
    got_kickable_reward_[tid] = false;
  }
  barrier_.wait();
  return reward;
}

KickToTeammate::KickToTeammate(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents),
    kicker_(offense_agents + defense_agents),
    old_ball_prox_(offense_agents + defense_agents, 0.),
    old_teammate_prox_(offense_agents + defense_agents, 0.),
    old_ball_dist_teammate_(offense_agents + defense_agents, 0.)
{
  CHECK_LE(offense_agents, 2);
  int max_steps = 50;
  int offense_on_ball = 100; // Randomize who gets the ball
  if (offense_agents == 1) {
    offense_on_ball = 1;
  }
  startServer(server_port, 2, 0, defense_agents, 0, true,
              max_steps, 0.5, 0.5, offense_on_ball);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
    kicker_[i].unum = -1;
  }
  if (offense_agents == 1) {
    // Start a dummy teammate
    string cmd = "./bin/dummy_teammate " + std::to_string(server_port) + " base_left false";
    threads_.emplace_back(ExecuteCommand, cmd);
    sleep(10);
  }
}

float KickToTeammate::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  Player pob = env.playerOnBall();
  const std::vector<float>& current_state = env.getState();
  CHECK_GT(current_state.size(), 64) << "Unexpected number of features.";
  float ball_proximity = current_state[53];
  float ball_dist = 1. - (ball_proximity+1.)/2.;
  float teammate_proximity = current_state[60];
  float teammate_dist = 1. - (teammate_proximity+1.)/2.;
  float ball_dist_teammate = getDist(
      ball_dist, current_state[51], current_state[52],
      teammate_dist, current_state[58], current_state[59]);
  float ball_dist_teammate_delta = ball_dist_teammate - old_ball_dist_teammate_[tid];
  float ball_prox_delta = ball_proximity - old_ball_prox_[tid];
  float teammate_prox_delta = teammate_proximity - old_teammate_prox_[tid];

  if (pob.unum > 0 && kicker_[tid].unum < 0) {
    kicker_[tid] = pob;
    VLOG(1) << "Unum-" << tid << " BackupKicker " << pob.unum;
    ball_dist_teammate_delta = 0;
    ball_prox_delta = 0;
    teammate_prox_delta = 0;
  }
  float reward = 0;

  if (kicker_[tid].unum == env.getUnum()) {
    // Positive reward for minimizing distance between ball and teammate
    float kick_to_teammate_reward = std::max(0.f, -ball_dist_teammate_delta);
    reward += kick_to_teammate_reward;
    VLOG(1) << "Unum" << env.getUnum() << " KickToTeammate " << kick_to_teammate_reward;
  } else { // We are recieving the kick
    // Positive Reward for getting closer to the ball
    float approach_ball_reward = std::max(0.f, ball_prox_delta);
    reward += approach_ball_reward;
    VLOG(1) << "Unum" << env.getUnum() << " ApproachBall " << approach_ball_reward;
  }
  // Negative reward for moving towards the teammate
  float avoid_teammate_reward = std::min(0.f, -teammate_prox_delta);
  reward += avoid_teammate_reward;
  VLOG(1) << "Unum" << env.getUnum() << " AvoidTeammate " << avoid_teammate_reward;

  old_ball_prox_[tid] = ball_proximity;
  old_teammate_prox_[tid] = teammate_proximity;
  old_ball_dist_teammate_[tid] = ball_dist_teammate;

  if (episodeOver()) {
    reward = 0.;
    VLOG(1) << "Unum-" << tid << " Kicker is " << pob.unum;
    kicker_[tid] = pob;
  }
  barrier_.wait();
  return reward;
}


Cross::Cross(int server_port, int offense_agents, int defense_agents,
             float ball_x_min, float ball_x_max) :
    Task(taskName(), offense_agents, defense_agents),
    initial_pob_(offense_agents + defense_agents)
 {
  CHECK_EQ(offense_agents, 2);
  int offense_on_ball = 100; // Randomize who gets the ball
  int defense_npcs = 1;
  startServer(server_port, 2, 0, defense_agents, defense_npcs, true,
              500, ball_x_min, ball_x_max, offense_on_ball);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float Cross::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  if (initial_pob_[tid].side != LEFT || initial_pob_[tid].unum <= 0) {
    initial_pob_[tid] = env.playerOnBall();
    VLOG(1) << "Initial POB: " << initial_pob_[tid].unum << " side " << initial_pob_[tid].side;
  }

  float reward = 0;

  // Both players get a reward when the player who didn't start with the ball scores
  if (status_[tid] == GOAL && (env.playerOnBall().unum != initial_pob_[tid].unum)) {
    reward += 1;
  }

  if (episodeOver()) {
    initial_pob_[tid].unum = 0;
    initial_pob_[tid].side = NEUTRAL;
  }

  barrier_.wait();
  return reward;
}

MirrorActions::MirrorActions(int server_port, int offense_agents,
                             int defense_agents) :
    Task(taskName(), offense_agents, defense_agents),
    actions_(offense_agents + defense_agents, NOOP),
    old_actions_(offense_agents + defense_agents, NOOP)
{
  CHECK_EQ(offense_agents, 2);
  int max_steps = 100;
  startServer(server_port, 2, 0, defense_agents, 0, true, max_steps);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

void MirrorActions::act(int tid, hfo::action_t action, float arg1, float arg2) {
  CHECK_GT(envs_.size(), tid);
  envs_[tid].act(action, arg1, arg2);
  actions_[tid] = action;
}

float MirrorActions::getReward(int tid) {
  barrier_.wait();

  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  float reward = 0;
  CHECK_EQ(actions_.size(), 2);
  // Agents are rewarded for both doing the same action
  if (actions_[0] == actions_[1]) {
    if (tid == 0) {
      VLOG(1) << "Reward 1: Both agents selected action " << actions_[0];
    }
    reward += .1;
  } else {
    if (tid == 0) {
      VLOG(1) << "Reward 0: Agents selected different actions "
              << actions_[0] << ", " << actions_[1];
    }
  }
  // Agents penalized for doing the same action more than once
  if (actions_[tid] == old_actions_[tid]) {
    VLOG(1) << "Reward -1. Agent" << tid << " twice selected action " << actions_[tid];
    reward -= .1;
  }
  old_actions_[tid] = actions_[tid];
  return reward;
}

SayMyTid::SayMyTid(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents)
{
  CHECK_EQ(offense_agents, 2);
  int max_steps = 100;
  startServer(server_port, 2, 0, defense_agents, 0, true, max_steps);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
}

float SayMyTid::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  std::string msg = env.hear();
  std::stringstream ss(msg);
  float f = 0;
  while (ss >> f) {
    if (ss.peek() == ' ') {
      ss.ignore();
    }
    break;
  }
  float reward = 0;
  if (!msg.empty()) {
    float target = tid == 0 ? -.8 : .8;
    reward = .1 / exp(50. * pow(target - f, 2.f));
    VLOG(1) << "Agent" << tid << " heard " << f << " reward " << reward;
  }
  barrier_.wait();
  return reward;
}

Keepaway::Keepaway(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents)
{
  CHECK_EQ(offense_agents, 2);
  int max_steps = 500;
  int defenders = 1;
  startServer(server_port, 2, 0, defenders, 0, true, max_steps);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
  // Start a chaser
  string cmd = "./bin/chaser " + std::to_string(server_port) + " base_right true &";
  threads_.emplace_back(ExecuteCommand, cmd);
  sleep(5);
}

float Keepaway::getReward(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  float reward = .01;
  status_t s = status_[tid];
  if (status_[tid] == OUT_OF_BOUNDS || status_[tid] == CAPTURED_BY_DEFENSE) {
    reward = -1.;
  }
  barrier_.wait();
  return reward;
}
