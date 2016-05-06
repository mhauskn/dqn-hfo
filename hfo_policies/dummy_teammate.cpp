#include <iostream>
#include <vector>
#include <HFO.hpp>
#include <cstdlib>

using namespace std;
using namespace hfo;

feature_set_t features = HIGH_LEVEL_FEATURE_SET;
string config_dir = "/u/mhauskn/projects/HFO/bin/teams/base/config/formations-dt";
int port = 6000;
string server_addr = "localhost";
string team_name = "base_left";
bool goalie = false;

int main(int argc, char** argv) {
  if (argc > 1) {
    port = atoi(argv[1]);
  }

  HFOEnvironment hfo;
  hfo.connectToServer(features, config_dir, port, server_addr,
                      team_name, goalie);
  status_t status = IN_GAME;
  while (status != SERVER_DOWN) {
    hfo.act(NOOP);
    status = hfo.step();
  }
  hfo.act(QUIT);
};
