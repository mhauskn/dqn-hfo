#include "task.hpp"

using namespace std;
using namespace hfo;

void ExecuteCommand(string cmd) {
  CHECK_EQ(system(cmd.c_str()), 0) << "Failed to execute command: " << cmd;
}

Task::Task(std::string task_name, int offense_agents, int defense_agents) :
    task_name_(task_name),
    status_(offense_agents + defense_agents, IN_GAME),
    offense_agents_(offense_agents),
    defense_agents_(defense_agents),
    server_port_(-1)
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
  status_t status = env.step();
  if (status == SERVER_DOWN) {
    LOG(FATAL) << "Server Down! Exiting.";
    exit(1);
  }
  status_[tid] = status;
  return status;
}

bool Task::episodeOver(int tid) const {
  CHECK_GT(status_.size(), tid);
  return status_[tid] != IN_GAME;
}

status_t Task::getStatus(int tid) const {
  CHECK_GT(status_.size(), tid);
  return status_[tid];
}

void Task::startServer(int port, int offense_agents, int offense_npcs,
                       int defense_agents, int defense_npcs, bool fullstate,
                       int frames_per_trial, float ball_x_min, float ball_x_max,
                       int offense_on_ball, bool gui, bool log_game, bool verbose) {
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
  if (!gui)      { cmd += " --headless"; }
  if (!log_game) { cmd += " --no-logging"; }
  if (verbose)   { cmd += " --verbose"; }
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
