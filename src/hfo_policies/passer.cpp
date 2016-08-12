#include <algorithm>
#include <iostream>
#include <vector>
#include <HFO.hpp>
#include <cstdlib>
#include <math.h>

using namespace std;
using namespace hfo;

// Before running this program, first Start HFO server:
// $./bin/HFO --offense-agents 1

// Server Connection Options. See printouts from bin/HFO.
feature_set_t features = LOW_LEVEL_FEATURE_SET;
string config_dir = "/u/mhauskn/projects/HFO/bin/teams/base/config/formations-dt";
int port = 6000;
string server_addr = "localhost";
string team_name = "base_left";
bool goalie = false;

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

std::pair<float, float> makeSpace(const vector<float>& current_state) {
  float tl_dist = 1. - (current_state[36]+1.)/2.;
  float tr_dist = 1. - (current_state[39]+1.)/2.;
  float br_dist = 1. - (current_state[42]+1.)/2.;
  float bl_dist = 1. - (current_state[45]+1.)/2.;
  float teammate_dist = 1. - (current_state[60]+1.)/2.;
  float tl_dist_teammate = getDist(tl_dist, current_state[34], current_state[35],
                                   teammate_dist, current_state[58], current_state[59]);
  float tr_dist_teammate = getDist(tr_dist, current_state[37], current_state[38],
                                   teammate_dist, current_state[58], current_state[59]);
  float br_dist_teammate = getDist(br_dist, current_state[40], current_state[41],
                                   teammate_dist, current_state[58], current_state[59]);
  float bl_dist_teammate = getDist(bl_dist, current_state[43], current_state[44],
                                   teammate_dist, current_state[58], current_state[59]);
  float A[4] = { tl_dist_teammate, tr_dist_teammate, br_dist_teammate, bl_dist_teammate };
  int max_indx = std::distance(A, std::max_element(A, A + 4));
  switch (max_indx) {
    case 0:
      return std::make_pair<float, float>(-.8, -.9);
    case 1:
      return std::make_pair<float, float>(.8, -.9);
    case 2:
      return std::make_pair<float, float>(.8, .9);
    case 3:
      return std::make_pair<float, float>(-.8, .9);
  }
  return std::make_pair<float, float>(0, 0);
}

float TMT_DIST = 0.3;

int main(int argc, char** argv) {
  if (argc < 4) {
    cout << "Usage: " << argv[0] << " port team_name goalie" << endl;
    exit(0);
  }
  port = atoi(argv[1]);
  team_name = argv[2];
  goalie = atoi(argv[3]);
  cout << port << " " << team_name << " " << goalie << endl;
  HFOEnvironment hfo;
  hfo.connectToServer(features, config_dir, port, server_addr,
                      team_name, goalie);
  status_t status = IN_GAME;
  for (int episode = 0; status != SERVER_DOWN; episode++) {
    status = IN_GAME;
    int kick_timer = 100;
    while (status == IN_GAME) {
      // Get the vector of state features for the current state
      const vector<float>& current_state = hfo.getState();
      bool kickable = current_state[12] > 0;
      float ball_dist = 1. - (current_state[53]+1.)/2.;
      float ball_vel_mag = current_state[55];
      float teammate_dist = 1. - (current_state[60]+1.)/2.;
      float ball_dist_teammate = getDist(ball_dist, current_state[51], current_state[52],
                                         teammate_dist, current_state[58], current_state[59]);
      if (kickable) {
        if (teammate_dist >= TMT_DIST) {
          float teammate_ang = acos(current_state[59]) * (180. / 3.14159265);
          if (current_state[58] < 0) { teammate_ang *= -1; };
          float power = std::min(100., 250. * teammate_dist);
          hfo.act(KICK, power, teammate_ang);
          kick_timer = 0;
        } else {
          std::pair<float, float> p = makeSpace(current_state);
          hfo.act(DRIBBLE_TO, p.first, p.second);
        }
      } else {
        if (ball_dist < ball_dist_teammate && kick_timer > 20) {
          // We are closer: Go to the ball
          hfo.act(MOVE);
        } else { // Teammate is closer to ball
          if (teammate_dist < TMT_DIST) {
            std::pair<float, float> p = makeSpace(current_state);
            hfo.act(MOVE_TO, p.first, p.second);
          } else {
            hfo.act(NOOP);
          }
        }
      }
      // Advance the environment and get the game status
      status = hfo.step();
      kick_timer++;
    }
  }
  hfo.act(QUIT);
};
