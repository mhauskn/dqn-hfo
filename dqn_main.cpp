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

// DQN Parameters
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(save, "state/", "Prefix for saving snapshots");
DEFINE_int32(epochs, 30, "Epochs for training mimic data");
DEFINE_int32(memory, 400000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, .1, "Value of epsilon after explore iterations.");
DEFINE_double(gamma, .99, "Discount factor of future rewards (0,1]");
DEFINE_int32(clone_freq, 10000, "Frequency (steps) of cloning the target network.");
DEFINE_int32(memory_threshold, 50000, "Number of transitions to start learning");
DEFINE_int32(updates_per_action, 1, "Updates done after each action taken");
DEFINE_string(actor_weights, "", "The actor pretrained weights load (*.caffemodel).");
DEFINE_string(critic_weights, "", "The critic pretrained weights load (*.caffemodel).");
DEFINE_string(actor_snapshot, "", "The actor solver state to load (*.solverstate).");
DEFINE_string(critic_snapshot, "", "The critic solver state to load (*.solverstate).");
DEFINE_string(memory_snapshot, "", "The replay memory to load (*.replaymemory).");
DEFINE_bool(resume, true, "Automatically resume training from latest snapshot.");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_bool(delay_reward, true, "If false will skip the timesteps between shooting EOT. ");
DEFINE_double(evaluate_with_epsilon, 0, "Epsilon value to be used in evaluation mode");
DEFINE_int32(evaluate_freq, 2500000, "Frequency (steps) between evaluations");
DEFINE_int32(repeat_games, 32, "Number of games played in evaluation mode");
DEFINE_int32(actor_update_factor, 1, "Number of actor updates per critic update");
DEFINE_string(actor_solver, "dqn_actor_solver.prototxt",
              "Actor solver parameter file (*.prototxt)");
DEFINE_string(critic_solver, "dqn_critic_solver.prototxt",
              "Critic solver parameter file (*.prototxt)");
DEFINE_string(server_cmd,
              "./scripts/start.py --offense-agents 1 --offense-npcs 0 --defense-agents 0 --defense-npcs 0 --record",
              "Command executed to start the HFO server.");
DEFINE_int32(port, -1, "Port to use for server/client.");
DEFINE_string(mimic_data, "1.log", "The mimic state-action train data to load (*.log)");
DEFINE_bool(mimic, false, "Mimic mode: mimic agent2D by training the network with mimic_data");

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
  std::deque<dqn::ActorStateDataSp> past_states;
  double total_score = 0;
  hfo_status_t status = IN_GAME;
  while (status == IN_GAME) {
    const std::vector<float>& current_state = hfo.getState();
    CHECK_EQ(current_state.size(), dqn::kStateDataSize);
    dqn::ActorStateDataSp current_state_sp = std::make_shared<dqn::ActorStateData>();
    std::copy(current_state.begin(), current_state.end(), current_state_sp->begin());
    past_states.push_back(current_state_sp);
    if (past_states.size() < dqn::kStateInputCount) {
      // If there are not past states enough for DQN input, just select DASH
      status = hfo.act({DASH, 0., 0.});
    } else {
      while (past_states.size() > dqn::kStateInputCount) {
        past_states.pop_front();
      }
      dqn::ActorInputStates input_states;
      std::copy(past_states.begin(), past_states.end(), input_states.begin());
      const Action action = dqn.SelectAction(input_states, epsilon);
      status = hfo.act(action);
      if (!FLAGS_delay_reward) { // Skip to EOT if not delayed reward
        while (status == IN_GAME) {
          status = hfo.act(action);
        }
      }
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

      // if (update) {
      //   // Add the current transition to replay memory
      //   const std::vector<float>& next_state = hfo.getState();
      //   CHECK_EQ(next_state.size(), dqn::kStateDataSize);
      //   dqn::ActorStateDataSp next_state_sp = std::make_shared<dqn::ActorStateData>();
      //   std::copy(next_state.begin(), next_state.end(), next_state_sp->begin());
      //   const auto transition = (status == IN_GAME) ?
      //       dqn::Transition(input_states, kickangle, reward, next_state_sp):
      //       dqn::Transition(input_states, kickangle, reward, boost::none);
      //   if (!FLAGS_delay_reward) {
      //     CHECK(!std::get<3>(transition)) << "Expected no next state...";
      //   }
      //   dqn.AddTransition(transition);
      //   // If the size of replay memory is large enough, update DQN
      //   if (dqn.memory_size() > FLAGS_memory_threshold) {
      //     for (int i = 0; i < FLAGS_updates_per_action; ++i) {
      //       dqn.UpdateCritic();
      //       for (int u = 0; u < FLAGS_actor_update_factor; ++u) {
      //         dqn.UpdateActor();
      //       }
      //     }
      //   }
      // }
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

void TrainMimic(HFOEnvironment& hfo, dqn::DQN& dqn, path save_path) {
  LOG(INFO) << "Begin training with mimic data.";
  LOG(INFO) << "The replay memory has " << dqn.memory_size() << " transitions.";
  int epochs = 0;
  while (epochs++ < FLAGS_epochs) {
    LOG(INFO) << "Epoch: " << epochs;
    int threshold = 0.9 * dqn.memory_size() / dqn::kMinibatchSize;
    int i = 0;
    int test_times = 0, train_times = 0;;
    float euclideanloss = 0;
    float softmaxloss = 0;
    std::vector<std::pair<int, int>> accuracy_train, accuracy_test;
    std::vector<float> deviation_train, deviation_test;
    for (int i = 0; i < 4; ++i) {
      accuracy_train.push_back(std::make_pair(0,0));
      accuracy_test.push_back(std::make_pair(0,0));
    }
    for (int i =0; i < 6; ++i) {
      deviation_train.push_back(0);
      deviation_test.push_back(0);
    }
    for (; i < threshold; ++i) {
      std::pair<float,float> loss = dqn.UpdateActor(i, true, accuracy_train,
                                                    deviation_train);
      euclideanloss += loss.first;
      softmaxloss += loss.second;
    }
    euclideanloss = euclideanloss / i;
    softmaxloss = softmaxloss / i;
    LOG(INFO) <<  "Train set: iteration "
              << epochs * threshold
              << ", Loss sum = " << euclideanloss + softmaxloss;
    LOG(INFO) << "  Euclideanloss = " << euclideanloss;
    LOG(INFO) << "  Softmaxloss = " << softmaxloss;
    euclideanloss = 0;
    softmaxloss = 0;
    int accuracy_train_1 = accuracy_train[0].first + accuracy_train[1].first
        + accuracy_train[2].first + accuracy_train[3].first;
    int accuracy_train_2 = accuracy_train[0].second + accuracy_train[1].second
        + accuracy_train[2].second +accuracy_train[3].second;
    LOG(INFO) << "Training Set: Iteration " << epochs * threshold << ", Accuracy "
              << accuracy_train_1 / (float)accuracy_train_2 << " ("
              << accuracy_train_1 << "/" << accuracy_train_2 << ")";
    LOG(INFO) << "  Iteration " << epochs * threshold <<", Dash Accuracy: "
              << accuracy_train[0].first / (float)accuracy_train[0].second
              << " (" << accuracy_train[0].first << "/" << accuracy_train[0].second
              << ") train_dash_deviation : "
              << deviation_train[0] / accuracy_train[0].first
              << " train_dash_deviation2 : "
              << deviation_train[1] / accuracy_train[0].first;
    LOG(INFO) << "  Iteration " << epochs * threshold <<", Turn Accuracy: "
              << accuracy_train[1].first / (float)accuracy_train[1].second
              << " (" << accuracy_train[1].first << "/" << accuracy_train[1].second
              << ") train_turn_deviation : "
              << deviation_train[2] / accuracy_train[1].first;
    LOG(INFO) << "  Iteration " << epochs * threshold <<", Tackle Accuracy: "
              << accuracy_train[2].first / (float)accuracy_train[2].second
              << " (" << accuracy_train[2].first << "/" << accuracy_train[2].second
              << ") train_tackle_deviation : "
              << deviation_train[3] / accuracy_train[2].first;
    LOG(INFO) << "  Iteration " << epochs * threshold <<", Kick Accuracy: "
              << accuracy_train[3].first / (float)accuracy_train[3].second
              << " (" << accuracy_train[3].first << "/" << accuracy_train[3].second
              << ") train_kick_deviation : "
              << deviation_train[4] / accuracy_train[3].first
              << " train_kick_deviation2 : " << deviation_train[5] / accuracy_train[3].first;
    for (; i < dqn.memory_size() / dqn::kMinibatchSize; ++i) {
      test_times++;
      std::pair<float,float> loss = dqn.UpdateActor(i, false,
                                                    accuracy_test, deviation_test);
      euclideanloss += loss.first;
      softmaxloss += loss.second;
    }
    euclideanloss = euclideanloss / test_times;
    softmaxloss = softmaxloss / test_times;
    LOG(INFO) << "Test set: iteration "
              << epochs * threshold
              << ", Loss sum = " << euclideanloss + softmaxloss;
    LOG(INFO) << "  Euclideanloss = " << euclideanloss;
    LOG(INFO) << "  Softmaxloss = " << softmaxloss;
    int accuracy_test_1 = accuracy_test[0].first + accuracy_test[1].first
        + accuracy_test[2].first + accuracy_test[3].first;
    int accuracy_test_2 = accuracy_test[0].second + accuracy_test[1].second
        + accuracy_test[2].second +accuracy_test[3].second;
    LOG(INFO) << "Test Set: Iteration " << epochs * threshold << ", Accuracy "
              << accuracy_test_1 / (float)accuracy_test_2 << " ("
              << accuracy_test_1 << "/" << accuracy_test_2 << ")";
    LOG(INFO) << "  Iteration " << epochs * threshold <<", DASH Accuracy: "
              << accuracy_test[0].first / (float)accuracy_test[0].second
              << " (" << accuracy_test[0].first << "/" << accuracy_test[0].second
              << ") test_dash_deviation : "
              << deviation_test[0] / accuracy_test[0].first
              << " test_dash_deviation2 : "
              << deviation_test[1] / accuracy_test[0].first;
    LOG(INFO) << "  Iteration " << epochs * threshold <<", TURN Accuracy: "
              << accuracy_test[1].first / (float)accuracy_test[1].second
              << " (" << accuracy_test[1].first << "/" << accuracy_test[1].second
              << ") test_turn_deviation : "
              << deviation_test[2] / accuracy_test[1].first;
    LOG(INFO) << "  Iteration " << epochs * threshold <<", TACKLE Accuracy: "
              << accuracy_test[2].first / (float)accuracy_test[2].second
              << " (" << accuracy_test[2].first << "/" << accuracy_test[2].second
              << ") test_tackle_deviation : "
              << deviation_test[3] / accuracy_test[2].first;
    LOG(INFO) << "  Iteration " << epochs * threshold <<", KICK Accuracy: "
              << accuracy_test[3].first / (float)accuracy_test[3].second
              << " (" << accuracy_test[3].first << "/" << accuracy_test[3].second
              << ") test_kick_deviation : "
              << deviation_test[4] / accuracy_test[3].first
              << " test_kick_deviation2 : "
              << deviation_test[5] / accuracy_test[3].first;
  }
  dqn.Snapshot(save_path.native(), false, false);
}

int main(int argc, char** argv) {
  std::string usage(argv[0]);
  usage.append(" -[evaluate|save path]");
  gflags::SetUsageMessage(usage);
  gflags::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  fLI::FLAGS_logbuflevel = -1;
  if (FLAGS_evaluate) {
    google::LogToStderr();
  }
  if (!is_regular_file(FLAGS_actor_solver)) {
    LOG(ERROR) << "Invalid solver: " << FLAGS_actor_solver;
    exit(1);
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

  if (FLAGS_port < 0) {
    srand(std::hash<std::string>()(save_path.native()));
    FLAGS_port = rand() % 40000 + 20000;
  }
  std::string cmd = FLAGS_server_cmd + " --port " + std::to_string(FLAGS_port);
  if (!FLAGS_gui) { cmd += " --headless"; }
  if (!FLAGS_evaluate) { cmd += " --no-logging"; }
  if (FLAGS_evaluate) { cmd += " --record"; }
  cmd += " &";
  LOG(INFO) << "Starting server with command: " << cmd;
  CHECK_EQ(system(cmd.c_str()), 0) << "Unable to start the HFO server.";

  HFOEnvironment hfo;
  hfo.connectToAgentServer(FLAGS_port, LOW_LEVEL_FEATURE_SET);

  // Get the vector of legal actions
  std::vector<int> legal_actions(dqn::kActionCount);
  std::iota(legal_actions.begin(), legal_actions.end(), 0);

  CHECK((FLAGS_critic_snapshot.empty() || FLAGS_critic_weights.empty()) &&
        (FLAGS_actor_snapshot.empty() || FLAGS_actor_weights.empty()))
      << "Give a snapshot to resume training or weights to finetune " <<
      "but not both.";

  // Construct the solver
  caffe::SolverParameter actor_solver_param;
  caffe::SolverParameter critic_solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_actor_solver, &actor_solver_param);
  caffe::ReadProtoFromTextFileOrDie(FLAGS_critic_solver, &critic_solver_param);
  actor_solver_param.set_snapshot_prefix((save_path.native() + "_actor").c_str());
  critic_solver_param.set_snapshot_prefix((save_path.native() + "_critic").c_str());

  dqn::DQN dqn(legal_actions, actor_solver_param, critic_solver_param,
               FLAGS_memory, FLAGS_gamma, FLAGS_clone_freq);
  dqn.Initialize();

  if (!FLAGS_critic_snapshot.empty() && !FLAGS_actor_snapshot.empty()) {
    CHECK(is_regular_file(FLAGS_memory_snapshot))
        << "Unable to find .replaymemory: " << FLAGS_memory_snapshot;
    LOG(INFO) << "Actor solver state resuming from " << FLAGS_actor_snapshot;
    LOG(INFO) << "Critic solver state resuming from " << FLAGS_critic_snapshot;
    dqn.RestoreSolver(FLAGS_actor_snapshot, FLAGS_critic_snapshot);
    LOG(INFO) << "Loading replay memory from " << FLAGS_memory_snapshot;
    dqn.LoadReplayMemory(FLAGS_memory_snapshot);
  } else if (!FLAGS_critic_weights.empty() && !FLAGS_actor_weights.empty()) {
    LOG(INFO) << "Actor weights finetuning from " << FLAGS_actor_weights;
    LOG(INFO) << "Critic weights finetuning from " << FLAGS_critic_weights;
    dqn.LoadTrainedModel(FLAGS_actor_weights, FLAGS_critic_weights);
  }

  if (FLAGS_mimic) {
    if (!FLAGS_mimic_data.empty()) {
      LOG(INFO) << "Loading mimic data into replay memory from mimic_data/";
      dqn.LoadMimicData(FLAGS_mimic_data);
      LOG(INFO) << "Successfully load mimic data into replay memory!";
      TrainMimic(hfo, dqn, save_path);
      LOG(INFO) << "Successfully train the network with mimic data.";
      return 0;
    }
    else {
      LOG(INFO) << "Please ensure mimic_data/" << FLAGS_mimic_data
                << " are ready to load.";
      return 0;
    }
  }

  if (FLAGS_evaluate) {
    Evaluate(hfo, dqn);
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
