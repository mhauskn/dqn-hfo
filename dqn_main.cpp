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

using namespace boost::filesystem;

// DQN Parameters
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(save, "", "Prefix for saving snapshots");
DEFINE_int32(memory, 400000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, .1, "Value of epsilon after explore iterations.");
DEFINE_double(gamma, .99, "Discount factor of future rewards (0,1]");
DEFINE_int32(clone_freq, 10000, "Frequency (steps) of cloning the target network.");
DEFINE_int32(memory_threshold, 50000, "Number of transitions to start learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_string(weights, "", "The pretrained weights load (*.caffemodel).");
DEFINE_string(snapshot, "", "The solver state to load (*.solverstate).");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, 0, "Epsilon value to be used in evaluation mode");
DEFINE_int32(evaluate_freq, 250000, "Frequency (steps) between evaluations");
DEFINE_int32(repeat_games, 32, "Number of games played in evaluation mode");
DEFINE_string(solver, "dqn_solver.prototxt", "Solver parameter file (*.prototxt)");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - (1.0 - FLAGS_epsilon) * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return FLAGS_epsilon;
  }
}

/**
 * Converts a discrete action into a continuous HFO-action
 */
Action GetAction(int action_idx) {
  CHECK_LT(action_idx, dqn::kOutputCount);
  Action a;
  switch (action_idx) {
    case 0:
      a = {KICK, 100., 0.};
      break;
    case 1:
      a = {KICK, 100., -10.};
      break;
    case 2:
      a = {KICK, 100., 10.};
      break;
    case 3:
      a = {KICK, 100., -20.};
      break;
    case 4:
      a = {KICK, 100., 20.};
      break;
    case 5:
      a = {KICK, 100., -30.};
      break;
    case 6:
      a = {KICK, 100., 30.};
      break;
    case 7:
      a = {KICK, 100., -45.};
      break;
    case 8:
      a = {KICK, 100., 45.};
      break;
    default:
      LOG(FATAL) << "Unknown action requested: " << action_idx;
  }
  return a;
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(HFOEnvironment& hfo, dqn::DQN& dqn, const double epsilon,
                      const bool update) {
  std::deque<dqn::StateDataSp> past_states;
  double total_score;
  hfo_status_t status = IN_GAME;
  while (status == IN_GAME) {
    const std::vector<float>& current_state = hfo.getState();
    CHECK_EQ(current_state.size(),dqn::kStateDataSize);
    dqn::StateDataSp current_state_sp = std::make_shared<dqn::StateData>();
    std::copy(current_state.begin(), current_state.end(), current_state_sp->begin());
    past_states.push_back(current_state_sp);
    if (past_states.size() < dqn::kInputCount) {
      // If there are not past states enough for DQN input, just select DASH
      Action a;
      a = {DASH, 0., 0.};
      status = hfo.act(a);
    } else {
      while (past_states.size() > dqn::kInputCount) {
        past_states.pop_front();
      }
      dqn::InputStates input_states;
      std::copy(past_states.begin(), past_states.end(), input_states.begin());
      const auto action_idx = dqn.SelectAction(input_states, epsilon);
      Action action = GetAction(action_idx);
      status = hfo.act(action);
      // Rewards for DQN are normalized as follows:
      // 1 for scoring a goal, -1 for captured by defense, out of bounds, out of time
      // 0 for other middle states
      float reward = 0;
      if (status == GOAL) {
        reward = 1;
      } else if (status == CAPTURED_BY_DEFENSE || status == OUT_OF_BOUNDS ||
               status == OUT_OF_TIME) {
        reward = -1;
      }
      total_score = reward;
      if (update) {
        // Add the current transition to replay memory
        const std::vector<float>& next_state = hfo.getState();
        CHECK_EQ(next_state.size(),dqn::kStateDataSize);
        dqn::StateDataSp next_state_sp = std::make_shared<dqn::StateData>();
        std::copy(next_state.begin(), next_state.end(), next_state_sp->begin());
        const auto transition = (status == IN_GAME) ?
            dqn::Transition(input_states, action_idx, reward, next_state_sp):
            dqn::Transition(input_states, action_idx, reward, boost::none);
        dqn.AddTransition(transition);
        // If the size of replay memory is large enough, update DQN
        if (dqn.memory_size() > FLAGS_memory_threshold) {
          dqn.Update();
        }
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
  usage.append(" -[evaluate|save path]");
  gflags::SetUsageMessage(usage);
  gflags::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  // google::LogToStderr();

  if (!is_regular_file(FLAGS_solver)) {
    LOG(ERROR) << "Invalid solver: " << FLAGS_solver;
    exit(1);
  }
  if (FLAGS_save.empty() && !FLAGS_evaluate) {
    LOG(ERROR) << "Save path (or evaluate) required but not set.";
    LOG(ERROR) << "Usage: " << gflags::ProgramUsage();
    exit(1);
  }
  path save_path(FLAGS_save);
  path snapshot_dir(current_path());
  // Check for files that may be overwritten
  assert(is_directory(snapshot_dir));
  LOG(INFO) << "Snapshots Prefix: " << save_path;
  directory_iterator end;
  for(directory_iterator it(snapshot_dir); it!=end; ++it) {
    if(boost::filesystem::is_regular_file(it->status())) {
      std::string save_path_str = save_path.stem().native();
      std::string other_str = it->path().filename().native();
      auto res = std::mismatch(save_path_str.begin(),
                               save_path_str.end(),
                               other_str.begin());
      if (res.first == save_path_str.end()) {
        LOG(ERROR) << "Existing file " << it->path()
                   << " conflicts with save path " << save_path;
        LOG(ERROR) << "Please remove this file or specify another save path.";
        exit(1);
      }
    }
  }
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

  HFOEnvironment hfo;
  hfo.connectToAgentServer(6008);

  // Get the vector of legal actions
  std::vector<int> legal_actions(dqn::kOutputCount);
  std::iota(legal_actions.begin(), legal_actions.end(), 0);

  CHECK(FLAGS_snapshot.empty() || FLAGS_weights.empty())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  // Construct the solver
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  solver_param.set_snapshot_prefix(save_path.c_str());

  dqn::DQN dqn(legal_actions, solver_param, FLAGS_memory, FLAGS_gamma,
               FLAGS_clone_freq);
  dqn.Initialize();

  if (!FLAGS_snapshot.empty()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    dqn.RestoreSolver(FLAGS_snapshot);
  } else if (!FLAGS_weights.empty()) {
    LOG(INFO) << "Finetuning from " << FLAGS_weights;
    dqn.LoadTrainedModel(FLAGS_weights);
  }

  if (FLAGS_evaluate) {
    if (FLAGS_gui) {
      auto score = PlayOneEpisode(hfo, dqn, FLAGS_evaluate_with_epsilon, false);
      LOG(INFO) << "Score " << score;
    } else {
      Evaluate(hfo, dqn);
    }
    return 0;
  }

  int last_eval_iter = 0;
  int episode = 0;
  double best_score = std::numeric_limits<double>::min();
  while (dqn.current_iteration() < solver_param.max_iter()) {
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
        dqn.Snapshot();
      }
      last_eval_iter = dqn.current_iteration();
    }
  }
  if (dqn.current_iteration() >= last_eval_iter) {
    Evaluate(hfo, dqn);
  }
};
