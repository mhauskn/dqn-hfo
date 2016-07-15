#ifndef TASK_HPP_
#define TASK_HPP_

#include <thread>
#include <HFO.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

class Task {
 public:
  Task(std::string task_name, int offense_agents, int defense_agents);
  ~Task();

  virtual void connectToServer(int tid);

  // TODO: Do we want an act()?
  hfo::status_t step(int tid);
  virtual float getReward(int tid) = 0;
  hfo::HFOEnvironment& getEnv(int tid);

  bool episodeOver(int tid) const;
  hfo::status_t getStatus(int tid) const;
  std::string getName() const { return task_name_; }

 protected:
  void startServer(int port, int offense_agents, int offense_npcs,
                   int defense_agents, int defense_npcs, bool fullstate=true,
                   int frames_per_trial=500, float ball_x_min=0., float ball_x_max=0.2,
                   int offense_on_ball=0, bool gui=false, bool log_game=false,
                   bool verbose=true);

 protected:
  std::string task_name_;
  std::vector<std::thread> threads_;
  std::vector<hfo::HFOEnvironment> envs_;
  std::vector<hfo::status_t> status_;
  int offense_agents_, defense_agents_;
  int server_port_;
};

/**
 * MoveToBall task rewards the agent for approaching the ball.
 */
class MoveToBall : public Task {
 public:
  MoveToBall(int server_port, int offense_agents, int defense_agents);
  virtual float getReward(int tid) override;

 protected:
  std::vector<float> old_ball_prox_;
  std::vector<float> ball_prox_delta_;
  std::vector<bool> first_step_;
};

class KickToGoal : public Task {
 public:
  KickToGoal(int server_port, int offense_agents, int defense_agents);
  virtual float getReward(int tid) override;

 protected:
  std::vector<float> old_ball_dist_goal_;
  std::vector<float> ball_dist_goal_delta_;
  std::vector<bool> first_step_;
};

#endif
