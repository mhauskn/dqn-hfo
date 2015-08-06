#include <cmath>
#include <iostream>
#include <HFO.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "dqn.hpp"
#include <boost/filesystem.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdlib.h>

using namespace boost::filesystem;
using namespace hfo;

DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_bool(benchmark, false, "Benchmark the network and exit");
// Load/Save Args
DEFINE_string(save, "", "Prefix for saving snapshots");
DEFINE_bool(resume, true, "Automatically resume training from latest snapshot.");
DEFINE_string(actor_solver, "actor_solver.prototxt", "Actor solver (*.prototxt)");
DEFINE_string(critic_solver, "critic_solver.prototxt", "Critic solver (*.prototxt)");
DEFINE_string(actor_weights, "", "The actor pretrained weights load (*.caffemodel).");
DEFINE_string(critic_weights, "", "The critic pretrained weights load (*.caffemodel).");
DEFINE_string(actor_snapshot, "", "The actor solver state to load (*.solverstate).");
DEFINE_string(critic_snapshot, "", "The critic solver state to load (*.solverstate).");
DEFINE_string(memory_snapshot, "", "The replay memory to load (*.replaymemory).");
// Epsilon-Greedy Args
DEFINE_int32(explore, 1000000, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, .1, "Value of epsilon after explore iterations.");
DEFINE_double(evaluate_with_epsilon, 0, "Epsilon value to be used in evaluation mode");
// Evaluation Args
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_int32(evaluate_freq, 10000, "Frequency (steps) between evaluations");
DEFINE_int32(repeat_games, 10, "Number of games played in evaluation mode");
// HFO Args
DEFINE_string(server_cmd, "./scripts/start.py --offense-agents 1 --fullstate",
              "Command executed to start the HFO server.");
DEFINE_int32(port, -1, "Port to use for server/client.");


double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - (1.0 - FLAGS_epsilon) * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return FLAGS_epsilon;
  }
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(HFOEnvironment& hfo, dqn::DQN& dqn, const double epsilon,
                      const bool update) {
  hfo.act({DASH, 0, 0});
  std::deque<dqn::StateDataSp> past_states;
  double total_score = 0;
  status_t status = IN_GAME;
  while (status == IN_GAME) {
    const std::vector<float>& current_state = hfo.getState();
    CHECK_EQ(current_state.size(), dqn::kStateSize);
    dqn::StateDataSp current_state_sp = std::make_shared<dqn::StateData>();
    std::copy(current_state.begin(), current_state.end(), current_state_sp->begin());
    past_states.push_back(current_state_sp);
    if (past_states.size() < dqn::kStateInputCount) {
      status = hfo.act({DASH, 0., 0.});
    } else {
      while (past_states.size() > dqn::kStateInputCount) {
        past_states.pop_front();
      }
      dqn::InputStates input_states;
      std::copy(past_states.begin(), past_states.end(), input_states.begin());
      const Action action = dqn.SelectAction(input_states, epsilon);
      float q_value = dqn.EvaluateAction(input_states, action);
      status = hfo.act(action);
      float reward = 0;
      if (status == GOAL) {
        reward = 1;
      } else if (status == CAPTURED_BY_DEFENSE || status == OUT_OF_BOUNDS ||
                 status == OUT_OF_TIME) {
        reward = -1;
      }
      total_score += reward;
      if (update) {
        const std::vector<float>& next_state = hfo.getState();
        CHECK_EQ(next_state.size(), dqn::kStateSize);
        dqn::StateDataSp next_state_sp = std::make_shared<dqn::StateData>();
        std::copy(next_state.begin(), next_state.end(), next_state_sp->begin());
        const auto transition = (status == IN_GAME) ?
            dqn::Transition(input_states, action, reward, next_state_sp):
            dqn::Transition(input_states, action, reward, boost::none);
        dqn.AddTransition(transition);
        dqn.Update();
      }
    }
  }
  return total_score;
}

/**
 * Evaluate the current player
 */
double Evaluate(HFOEnvironment& hfo, dqn::DQN& dqn) {
  std::vector<double> scores;
  for (int i = 0; i < FLAGS_repeat_games; ++i) {
    double score = PlayOneEpisode(hfo, dqn, FLAGS_evaluate_with_epsilon, false);
    scores.push_back(score);
  }
  double total_score = 0.0;
  for (auto score : scores) {
    total_score += score;
  }
  const auto avg_score = total_score / static_cast<double>(scores.size());
  double stddev = 0.0; // Compute the sample standard deviation
  for (auto i=0; i<scores.size(); ++i) {
    stddev += (scores[i] - avg_score) * (scores[i] - avg_score);
  }
  stddev = sqrt(stddev / static_cast<double>(FLAGS_repeat_games - 1));
  LOG(INFO) << "Evaluation avg_score = " << avg_score << " std = " << stddev;
  return avg_score;
}

int main(int argc, char** argv) {
  std::string usage(argv[0]);
  usage.append(" -[evaluate|save [path]]");
  gflags::SetUsageMessage(usage);
  gflags::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  fLI::FLAGS_logbuflevel = -1;
  if (FLAGS_evaluate) {
    google::LogToStderr();
  }
  if (FLAGS_save.empty() && !FLAGS_evaluate) {
    LOG(ERROR) << "Save path (or evaluate) required but not set.";
    LOG(ERROR) << "Usage: " << gflags::ProgramUsage();
    exit(1);
  }
  path save_path(FLAGS_save);
  // Set the logging destinations
  google::SetLogDestination(google::GLOG_INFO,
                            (save_path.native() + "_INFO_").c_str());
  google::SetLogDestination(google::GLOG_WARNING,
                            (save_path.native() + "_WARNING_").c_str());
  google::SetLogDestination(google::GLOG_ERROR,
                            (save_path.native() + "_ERROR_").c_str());
  google::SetLogDestination(google::GLOG_FATAL,
                            (save_path.native() + "_FATAL_").c_str());

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  // Look for a recent snapshot to resume
  LOG(INFO) << "Save path: " << save_path.native();
  if (FLAGS_resume && FLAGS_actor_snapshot.empty()
      && FLAGS_critic_snapshot.empty() && FLAGS_memory_snapshot.empty()) {
    dqn::FindLatestSnapshot(save_path.native(), FLAGS_actor_snapshot,
                            FLAGS_critic_snapshot, FLAGS_memory_snapshot);
  }

  CHECK((FLAGS_critic_snapshot.empty() || FLAGS_critic_weights.empty()) &&
        (FLAGS_actor_snapshot.empty() || FLAGS_actor_weights.empty()))
      << "Give a snapshot or weights but not both.";

  // Start rcssserver3d
  if (FLAGS_port < 0) {
    srand(std::hash<std::string>()(save_path.native()));
    FLAGS_port = rand() % 40000 + 20000;
  }
  std::string cmd = FLAGS_server_cmd + " --port " + std::to_string(FLAGS_port);
  if (!FLAGS_gui) { cmd += " --headless"; }
  if (!FLAGS_evaluate) { cmd += " --no-logging"; }
  cmd += " &";
  LOG(INFO) << "Starting server with command: " << cmd;
  CHECK_EQ(system(cmd.c_str()), 0) << "Unable to start the HFO server.";

  HFOEnvironment hfo;
  hfo.connectToAgentServer(FLAGS_port, LOW_LEVEL_FEATURE_SET);

  // Construct the solver
  caffe::SolverParameter actor_solver_param;
  caffe::SolverParameter critic_solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_actor_solver, &actor_solver_param);
  caffe::ReadProtoFromTextFileOrDie(FLAGS_critic_solver, &critic_solver_param);
  actor_solver_param.set_snapshot_prefix((save_path.native() + "_actor").c_str());
  critic_solver_param.set_snapshot_prefix((save_path.native() + "_critic").c_str());

  dqn::DQN dqn(actor_solver_param, critic_solver_param);

  // Load actor/critic/memory
  if (!FLAGS_actor_snapshot.empty()) {
    dqn.RestoreActorSolver(FLAGS_actor_snapshot);
  } else if (!FLAGS_actor_weights.empty()) {
    dqn.LoadActorWeights(FLAGS_actor_weights);
  }
  if (!FLAGS_critic_snapshot.empty()) {
    dqn.RestoreCriticSolver(FLAGS_critic_snapshot);
  } else if (!FLAGS_critic_weights.empty()) {
    dqn.LoadCriticWeights(FLAGS_critic_weights);
  }
  if (!FLAGS_memory_snapshot.empty()) {
    dqn.LoadReplayMemory(FLAGS_memory_snapshot);
  }

  if (FLAGS_evaluate) {
    Evaluate(hfo, dqn);
    return 0;
  }

  if (FLAGS_benchmark) {
    double score = PlayOneEpisode(hfo, dqn, FLAGS_evaluate_with_epsilon, true);
    dqn.Benchmark(1000);
    return 0;
  }

  int last_eval_iter = 0;
  int episode = 0;
  double best_score = std::numeric_limits<double>::min();
  while (dqn.current_iteration() < actor_solver_param.max_iter()) {
    double epsilon = CalculateEpsilon(dqn.current_iteration());
    double score = PlayOneEpisode(hfo, dqn, epsilon, true);
    LOG(INFO) << "Episode " << episode << " score = " << score
              << ", epsilon = " << epsilon
              << ", iter = " << dqn.current_iteration()
              << ", replay_mem_size = " << dqn.memory_size();
    episode++;

    if (dqn.current_iteration() >= last_eval_iter + FLAGS_evaluate_freq) {
      double avg_score = Evaluate(hfo, dqn);
      if (avg_score > best_score) {
        LOG(INFO) << "iter " << dqn.current_iteration()
                  << " New High Score: " << avg_score;
        best_score = avg_score;
        std::string fname = save_path.native() + "_HiScore" +
            std::to_string(avg_score);
        dqn.Snapshot(fname, false, false);
      }
      dqn.Snapshot(save_path.native(), true, true);
      last_eval_iter = dqn.current_iteration();
    }
  }
  if (dqn.current_iteration() >= last_eval_iter) {
    Evaluate(hfo, dqn);
  }
};
