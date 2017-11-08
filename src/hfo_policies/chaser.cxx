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
feature_set_t features = HIGH_LEVEL_FEATURE_SET;
string config_dir = "${HFO_ROOT_DIR}/bin/teams/base/config/formations-dt";
int port = 6000;
string server_addr = "localhost";
string team_name = "base_right";
bool goalie = true;

int main(int argc, char** argv) {
  if (argc < 4) {
    cout << "Usage: " << argv[0] << " port team_name goalie" << endl;
    exit(0);
  }
  port = atoi(argv[1]);
  team_name = argv[2];
  goalie = atoi(argv[3]);
  cout << port << " " << team_name << " " << goalie << endl;
  // Create the HFO environment
  HFOEnvironment hfo;
  // Connect to the server and request high-level feature set. See
  // manual for more information on feature sets.
  hfo.connectToServer(features, config_dir, port, server_addr,
                      team_name, goalie);
  status_t status = IN_GAME;
  for (int episode = 0; status != SERVER_DOWN; episode++) {
    status = IN_GAME;
    while (status == IN_GAME) {
      // Get the vector of state features for the current state
      const vector<float>& feature_vec = hfo.getState();
      float orientation = feature_vec[2];
      float ball_prox = feature_vec[3];
      float ball_ang = feature_vec[4];
      if (feature_vec[5] == 1) {
        hfo.act(CATCH);
      } else if (fabs(ball_ang - orientation) > .1) {
        hfo.act(TURN, 90.0 * (ball_ang - orientation));
      } else {
        hfo.act(DASH, 100., 0.);
      }
      // Advance the environment and get the game status
      status = hfo.step();
    }
  }
  hfo.act(QUIT);
};
