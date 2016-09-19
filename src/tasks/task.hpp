#ifndef TASK_HPP_
#define TASK_HPP_

#include <mutex>
#include <condition_variable>
#include <thread>
#include <HFO.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/thread/barrier.hpp>

class Task {
 public:
  Task(std::string task_name, int offense_agents, int defense_agents);
  ~Task();

  // Takes an action in the environment
  virtual void act(int tid, hfo::action_t action, float arg1, float arg2);

  // Say a message in the environment
  virtual void say(int tid, const std::string& message);

  // Advances the environment. Returns new status and reward.
  std::pair<hfo::status_t, float> step(int tid);

  // Returns how much reward an agent can expect to accrue if performing well.
  virtual float getMaxExpectedReward() = 0;

  int getID() { return id_; }
  void setID(int new_id) { id_ = new_id; }
  bool episodeOver() const { return episode_over_; }
  hfo::HFOEnvironment& getEnv(int tid);
  hfo::status_t getStatus(int tid) const;
  std::string getName() const { return task_name_; }
  float getEpisodeReward(int tid) const { return episode_reward_[tid]; }
  float getLastEpisodeReward(int tid) const { return last_episode_reward_[tid]; }
  int getNumAgents() const { return envs_.size(); }

 protected:
  virtual float getReward(int tid) = 0;
  virtual float getNormReward(int tid) { return getReward(tid) / getMaxExpectedReward(); }
  void stepThread(int tid);
  hfo::status_t stepUntilEpisodeEnd(int tid);
  void startServer(int port, int offense_agents, int offense_npcs,
                   int defense_agents, int defense_npcs, bool fullstate=true,
                   int frames_per_trial=500, float ball_x_min=0., float ball_x_max=0.2,
                   int offense_on_ball=0);
  virtual void connectToServer(int tid);
  static float getDist(float o1_dist, float o1_ang_sin, float o1_ang_cos,
                       float o2_dist, float o2_ang_sin, float o2_ang_cos);


 protected:
  std::string task_name_;
  int id_;
  std::vector<std::thread> threads_;
  std::vector<hfo::HFOEnvironment> envs_;
  std::vector<hfo::status_t> status_;
  std::vector<std::mutex> mutexs_;
  std::vector<std::condition_variable> cvs_;
  std::vector<int> need_to_step_;
  std::vector<float> episode_reward_;
  std::vector<float> last_episode_reward_;
  int offense_agents_, defense_agents_;
  int server_port_;
  bool episode_over_;
  boost::barrier barrier_;
};

/**
 * MoveToBall task rewards the agent for approaching the ball.
 */
class MoveToBall : public Task {
 public:
  MoveToBall(int server_port, int offense_agents, int defense_agents,
             float ball_x_min=0.0, float ball_x_max=0.8);
  virtual float getMaxExpectedReward() { return 0.6; }
  static std::string taskName() { return "move_to_ball"; }

 protected:
  virtual float getReward(int tid) override;

  std::vector<float> old_ball_prox_;
  std::vector<float> ball_prox_delta_;
  std::vector<bool> first_step_;
};

class MoveAwayFromBall : public Task {
 public:
  MoveAwayFromBall(int server_port, int offense_agents, int defense_agents,
                   float ball_x_min=0.0, float ball_x_max=0.8);
  virtual float getMaxExpectedReward() { return 0.7; }
  static std::string taskName() { return "move_away_from_ball"; }

 protected:
  virtual float getReward(int tid) override;

  std::vector<float> old_ball_prox_;
  std::vector<bool> first_step_;
};
/**
 * KickToGoal task initializes the agent with the ball and rewards the
 * agent for kicking the ball towards the goal.
 */
class KickToGoal : public Task {
 public:
  KickToGoal(int server_port, int offense_agents, int defense_agents,
             float ball_x_min=0.4, float ball_x_max=0.8);
  virtual float getMaxExpectedReward() { return 0.4; }
  static std::string taskName() { return "kick_to_goal"; }

 protected:
  virtual float getReward(int tid) override;

  std::vector<float> old_ball_dist_goal_;
  std::vector<bool> first_step_;
};

/**
 * Soccer task initializes the agent away from the ball and only
 * rewards the agent for scoring a goal.
 */
class Soccer : public Task {
 public:
  Soccer(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 1; }
  static std::string taskName() { return "soccer"; }

 protected:
  virtual float getReward(int tid) override;
};
class Soccer1v1 : public Task {
 public:
  Soccer1v1(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 1; }
  static std::string taskName() { return "soccer1v1"; }

 protected:
  virtual float getReward(int tid) override;
};
class Soccer2v1 : public Task {
 public:
  Soccer2v1(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 1; }
  static std::string taskName() { return "soccer2v1"; }

 protected:
  virtual float getReward(int tid) override;
  std::vector<hfo::Player> old_pob_;
};

/**
 * The original soccer task features a more informative reward signal
 * that rewards the agent for going to the ball, and kicking to goal.
 */
class SoccerEasy : public Task {
 public:
  SoccerEasy(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 9; }
  static std::string taskName() { return "soccer_easy"; }

 protected:
  virtual float getReward(int tid) override;

  std::vector<bool> first_step_;
  std::vector<float> old_ball_prox_;
  std::vector<bool> old_kickable_;
  std::vector<float> old_ball_dist_goal_;
  std::vector<bool> got_kickable_reward_;
  std::vector<hfo::Player> old_pob_;
};

/**
 * Dribble task encourages the agent to dribble the ball without going
 * out of bounds or losing control. The agent is initialized with the
 * ball.
 */
class Dribble : public Task {
 public:
  Dribble(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 20; }
  static std::string taskName() { return "dribble"; }

 protected:
  virtual float getReward(int tid) override;
};

/**
 * Passing requires the agent to kick
 */
class Pass : public Task {
 public:
  Pass(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 27; }
  static std::string taskName() { return "pass"; }

 protected:
  virtual float getReward(int tid) override;

  std::vector<bool> pass_active_;
  std::vector<hfo::Player> kicker_;
  std::vector<float> old_ball_prox_;
  std::vector<float> old_teammate_prox_;
  std::vector<float> old_ball_dist_teammate_;
  std::vector<bool> got_kickable_reward_;
};


/**
 * Passing requires the agent to kick
 */
class KickToTeammate : public Task {
 public:
  KickToTeammate(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 1; }
  static std::string taskName() { return "kick_to_teammate"; }

 protected:
  virtual float getReward(int tid) override;

  std::vector<hfo::Player> kicker_;
  std::vector<float> old_ball_prox_;
  std::vector<float> old_teammate_prox_;
  std::vector<float> old_ball_dist_teammate_;
};

/**
 * Cross task is a set play that requires the agent to pass to the
 * teammate before scoring.
 */
class Cross : public Task {
 public:
  Cross(int server_port, int offense_agents, int defense_agents,
        float ball_x_min=0.7, float ball_x_max=0.7);
  virtual float getMaxExpectedReward() { return 1; }
  static std::string taskName() { return "cross"; }

 protected:
  virtual float getReward(int tid) override;

  std::vector<hfo::Player> initial_pob_;
};

/**
 * The MirrorActions task rewards one agent for performing a task and
 * the other for mirroring the actions of the first. Agents are
 * penalized for repeating actions.
 */
class MirrorActions : public Task {
 public:
  MirrorActions(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 30; }
  static std::string taskName() { return "mirror_actions"; }

  virtual void act(int tid, hfo::action_t action, float arg1, float arg2) override;

 protected:
  virtual float getReward(int tid) override;

  std::vector<hfo::action_t> actions_;
  std::vector<hfo::action_t> old_actions_;
};

/**
 * SayMyTid rewards each agent when it says the tid of the teammate
 */
class SayMyTid : public Task {
 public:
  SayMyTid(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 10; }
  static std::string taskName() { return "say_my_tid"; }

 protected:
  virtual float getReward(int tid) override;
};

class Keepaway : public Task {
 public:
  Keepaway(int server_port, int offense_agents, int defense_agents);
  virtual float getMaxExpectedReward() { return 1; }
  static std::string taskName() { return "keepaway"; }

 protected:
  virtual float getReward(int tid) override;
};

#endif
