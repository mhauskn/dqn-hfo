#ifndef HFO_GAME_HPP_
#define HFO_GAME_HPP_

#include <HFO.hpp>
#include <random>

struct Action {
  hfo::action_t action;
  float arg1;
  float arg2;
};

const
inline int NumStateFeatures(int num_players) {
  return 50 + 9 * (num_players);
}

constexpr auto kPassVelThreshold = -.5;

// Starts the RCSSSERVER
void StartHFOServer(int port, int offense_agents, int offense_npcs,
                    int defense_agents, int defense_npcs);

void StartDummyTeammate(int port);
void StartDummyGoalie(int port);
void StartChaser(int port, std::string team_name, bool goalie);

void StopHFOServer();

// Creates an interface for a single agent to connect to the server
void ConnectToServer(hfo::HFOEnvironment& hfo_env, int port=6000);

// Returns a random HFO Action
Action GetRandomHFOAction(std::mt19937& random_engine);

class HFOGameState {
 public:
  HFOGameState(int unum);
  ~HFOGameState();
  void update(hfo::HFOEnvironment& hfo);
  float reward();
  float move_to_ball_reward();
  float kick_to_goal_reward();
  float EOT_reward();
  float pass_reward();

 public:
  float old_ball_prox, ball_prox_delta;
  float old_kickable, kickable_delta;
  float old_ball_dist_goal, ball_dist_goal_delta;
  int steps;
  double total_reward;
  double extrinsic_reward;
  hfo::status_t status;
  bool episode_over;
  bool got_kickable_reward;
  hfo::Player old_player_on_ball, player_on_ball;
  int our_unum;
  bool pass_active;
};

#endif /* HFO_GAME_HPP_ */
