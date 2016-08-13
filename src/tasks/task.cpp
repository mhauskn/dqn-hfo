#include "task.hpp"

using namespace std;
using namespace hfo;

DEFINE_bool(gui, false, "Open a GUI window.");
DEFINE_bool(log_game, false, "Log the HFO game.");
DEFINE_bool(verbose, false, "Server prints verbose output.");

bool FINISHED = false;

void ExecuteCommand(string cmd) {
  CHECK_EQ(system(cmd.c_str()), 0) << "Failed to execute command: " << cmd;
}

float getDist(float o1_dist, float o1_ang_sin, float o1_ang_cos,
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
    steps_(offense_agents + defense_agents, 0),
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

pair<status_t, float> Task::step(int tid) {
  CHECK_GT(envs_.size(), tid);
  float reward = getReward(tid);
  episode_reward_[tid] += reward;
  status_t old_status = status_[tid];
  if (episode_over_ && old_status == IN_GAME) {
    stepUntilEpisodeEnd(tid);
  } else {
    stepThread(tid);
  }
  status_t current_status = status_[tid];
  steps_[tid] += 1;
  if (current_status == SERVER_DOWN) {
    LOG(FATAL) << "Server Down! Exiting.";
    exit(1);
  }
  if (episode_over_ && old_status != IN_GAME && current_status == IN_GAME) {
    episode_over_ = false;
    episode_reward_[tid] = 0;
    steps_[tid] = 0;
  }
  if (current_status != IN_GAME) {
    episode_over_ = true;
    last_episode_reward_[tid] = episode_reward_[tid];
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
      + " --log-dir log/" + task_name_;
  if (fullstate) { cmd += " --fullstate"; }
  if (!FLAGS_gui)      { cmd += " --headless"; }
  if (!FLAGS_log_game) { cmd += " --no-logging"; }
  if (FLAGS_verbose)   { cmd += " --verbose"; }
  LOG(INFO) << "Starting server with command: " << cmd;
  threads_.emplace_back(std::thread(ExecuteCommand, cmd));
  sleep(5);
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
    reward -= 0.5 * episode_reward_[tid];
  }
  barrier_.wait();
  return reward;
}

Pass::Pass(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents),
    pass_active_(offense_agents + defense_agents, false),
    kicker_(offense_agents + defense_agents),
    pass_timer_(offense_agents + defense_agents, 0),
    old_ball_prox_(offense_agents + defense_agents, 0.),
    old_teammate_prox_(offense_agents + defense_agents, 0.),
    old_ball_dist_teammate_(offense_agents + defense_agents, 0.),
    first_step_(offense_agents + defense_agents, true),
    got_kickable_reward_(offense_agents + defense_agents, false)
{
  CHECK_EQ(offense_agents, 1);
  int offense_on_ball = 100; // Randomize who gets the ball
  startServer(server_port, offense_agents + 1, 0, defense_agents, 0, true,
              500, 0.5, 0.5, offense_on_ball);
  // Connect the agents to the server
  for (int i=0; i<envs_.size(); ++i) {
    connectToServer(i);
    sleep(5);
  }
  // Start a passing teammate
  string cmd = "./bin/passer " + std::to_string(server_port) + " base_left false &";
  ExecuteCommand(cmd);
  threads_.emplace_back(ExecuteCommand, cmd);
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
  float ball_dist_teammate = getDist(ball_dist, current_state[51], current_state[52],
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
      pass_timer_[tid] = 0;
    } else if (ball_vel_mag <= -.8) {
      reward -= 1.;
      VLOG(1) << "Pass Inactive: Ball too slow. Reward -1";
      pass_active_[tid] = false;
      pass_timer_[tid] = 0;
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
    pass_timer_[tid] = 0;
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
  pass_timer_[tid] += 1;
  status_t s = status_[tid];
  // Lose half of the accrued reward for ball out of bounds
  if (s == GOAL || s == OUT_OF_BOUNDS || s == CAPTURED_BY_DEFENSE) {
    reward -= std::max(0., 0.5 * episode_reward_[tid]);
    VLOG(1) << "reward -= OOB " << -std::max(0., 0.5 * episode_reward_[tid]);
  }
  if (episodeOver()) {
    pass_active_[tid] = false;
    pass_timer_[tid] = 0;
    old_ball_dist_teammate_[tid] = 0;
    old_ball_prox_[tid] = 0;
    old_teammate_prox_[tid] = 0;
    got_kickable_reward_[tid] = false;
  }
  return reward;
}
