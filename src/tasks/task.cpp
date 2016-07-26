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

Task::Task(std::string task_name, int offense_agents, int defense_agents) :
    task_name_(task_name),
    status_(offense_agents + defense_agents, IN_GAME),
    mutexs_(offense_agents + defense_agents),
    cvs_(offense_agents + defense_agents),
    need_to_step_(offense_agents + defense_agents, false),
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

status_t Task::step(int tid) {
  CHECK_GT(envs_.size(), tid);

  status_t old_status = status_[tid];
  if (episode_over_ && old_status == IN_GAME) {
    stepUntilEpisodeEnd(tid);
  } else {
    stepThread(tid);
  }
  status_t current_status = status_[tid];

  if (current_status == SERVER_DOWN) {
    LOG(FATAL) << "Server Down! Exiting.";
    exit(1);
  }

  if (episode_over_ && old_status != IN_GAME && current_status == IN_GAME) {
    episode_over_ = false;
  }

  if (current_status != IN_GAME) {
    episode_over_ = true;
  }

  return current_status;
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
      + " --offense-on-ball " + std::to_string(offense_on_ball);
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


Dribble::Dribble(int server_port, int offense_agents, int defense_agents) :
    Task(taskName(), offense_agents, defense_agents),
    episode_reward_(0)
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

  // LOG(INFO) << "self_vel " << self_vel_mag << " ball_prox " << ball_proximity
  //           << " ball_vel " << ball_vel_mag;

  float reward = 0;
  if (ball_proximity >= .94 && ball_vel_mag >= -.8) {
    reward += .1;
  }
  episode_reward_ += reward;
  // Don't go out of bounds or lose control of the ball
  if (status_[tid] == GOAL || status_[tid] == OUT_OF_BOUNDS ||
      status_[tid] == CAPTURED_BY_DEFENSE) {
    // Lose half of the total accrued reward for going out of bounds
    reward -= 0.5 * episode_reward_;
  }
  if (episodeOver()) {
    episode_reward_ = 0;
  }

  barrier_.wait();
  return reward;
}
