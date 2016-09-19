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
  while (status != SERVER_DOWN) {
    hfo.act(NOOP);
    status = hfo.step();
  }
  hfo.act(QUIT);
};
