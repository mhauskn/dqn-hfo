#include "task.hpp"

using namespace std;
using namespace hfo;

DEFINE_bool(gui, false, "Open a GUI window.");
DEFINE_bool(log_game, false, "Log the HFO game.");
DEFINE_bool(verbose, false, "Server prints verbose output.");

void ExecuteCommand(string cmd) {
  CHECK_EQ(system(cmd.c_str()), 0) << "Failed to execute command: " << cmd;
}

Task::Task(std::string task_name, int offense_agents, int defense_agents) :
    task_name_(task_name),
    status_(offense_agents + defense_agents, IN_GAME),
    offense_agents_(offense_agents),
    defense_agents_(defense_agents),
    server_port_(-1),
    episode_over_(false),
    barrier_(offense_agents + defense_agents)
{
  envs_.resize(offense_agents + defense_agents);
}

Task::~Task() {
  for (HFOEnvironment& env : envs_) {
    env.act(QUIT);
    env.step();
  }
  for (std::thread& t : threads_) {
    t.join();
  }
}

status_t Task::step(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];

  status_t status;
  if (episode_over_ && status_[tid] == IN_GAME) {
    status = stepUntilEpisodeEnd(tid);
  } else {
    status = env.step();
  }

  if (status == SERVER_DOWN) {
    LOG(FATAL) << "Server Down! Exiting.";
    exit(1);
  }

  if (episode_over_ && status_[tid] != IN_GAME && status == IN_GAME) {
    episode_over_ = false;
  }

  if (status != IN_GAME) {
    episode_over_ = true;
  }

  status_[tid] = status;
  return status;
}

status_t Task::stepUntilEpisodeEnd(int tid) {
  CHECK_GT(envs_.size(), tid);
  HFOEnvironment& env = envs_[tid];
  while (status_[tid] == IN_GAME) {
    env.act(NOOP);
    status_[tid] = env.step();
    if (status_[tid] == SERVER_DOWN) {
      LOG(FATAL) << "Server Down! Exiting.";
      exit(1);
    }
  }
  return status_[tid];
}

// bool Task::episodeOver(int tid) const {
//   CHECK_GT(status_.size(), tid);
//   return status_[tid] != IN_GAME;
// }

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

void Task::connectToServer(int tid) {
  CHECK_GT(envs_.size(), tid);
  CHECK_GT(server_port_, 0);
  if (tid < offense_agents_) {
    envs_[tid].connectToServer(LOW_LEVEL_FEATURE_SET, "bin/formations-dt",
                               server_port_);
  } else {
    envs_[tid].connectToServer(LOW_LEVEL_FEATURE_SET, "bin/formations-dt",
                               server_port_, "localhost", "base_right",
                               tid==offense_agents_);
  }
}

HFOEnvironment& Task::getEnv(int tid) {
  CHECK_GT(envs_.size(), tid);
  return envs_[tid];
}
