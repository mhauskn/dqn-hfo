#ifndef HFO_GAME_HPP_
#define HFO_GAME_HPP_

#include <HFO.hpp>
#include <random>

hfo::HFOEnvironment CreateHFOEnvironment();
hfo::Action GetRandomHFOAction(std::mt19937& random_engine);

class HFOGameState {
 public:
  HFOGameState(hfo::HFOEnvironment& hfo);
  ~HFOGameState();
  void update(const std::vector<float>& current_state,
              hfo::status_t current_status);
  float reward();
  float move_to_ball_reward();
  float kick_to_goal_reward();
  float EOT_reward();

 public:
  hfo::HFOEnvironment& hfo;
  float old_ball_prox, ball_prox_delta;
  float old_kickable, kickable_delta;
  float old_ball_dist_goal, ball_dist_goal_delta;
  int steps;
  double total_reward;
  hfo::status_t status;
  bool episode_over;
  bool got_kickable_reward;
};

#endif /* HFO_GAME_HPP_ */
