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
DEFINE_string(save, "", "Prefix for saving snapshots");
DEFINE_int32(memory, 400000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, .1, "Value of epsilon after explore iterations.");
DEFINE_double(gamma, .99, "Discount factor of future rewards (0,1]");
DEFINE_int32(clone_freq, 10000, "Frequency (steps) of cloning the target network.");
DEFINE_int32(memory_threshold, 50000, "Number of transitions to start learning");
DEFINE_string(actor_weights, "", "The actor pretrained weights load (*.caffemodel).");
DEFINE_string(critic_weights, "", "The critic pretrained weights load (*.caffemodel).");
DEFINE_string(actor_snapshot, "", "The actor solver state to load (*.solverstate).");
DEFINE_string(critic_snapshot, "", "The critic solver state to load (*.solverstate).");
DEFINE_string(memory_snapshot, "", "The replay memory to load (*.replaymemory).");
DEFINE_bool(resume, true, "Automatically resume training from latest snapshot.");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_bool(delay_reward, true, "If false will skip the timesteps between shooting EOT. ");
DEFINE_double(evaluate_with_epsilon, 0, "Epsilon value to be used in evaluation mode");
DEFINE_int32(evaluate_freq, 250000, "Frequency (steps) between evaluations");
DEFINE_int32(repeat_games, 32, "Number of games played in evaluation mode");
DEFINE_int32(actor_update_factor, 4, "Number of actor updates per critic update");
DEFINE_string(actor_solver, "dqn_actor_solver.prototxt", "Actor solver parameter file (*.prototxt)");
DEFINE_string(critic_solver, "dqn_critic_solver.prototxt", "Critic solver parameter file (*.prototxt)");
DEFINE_string(server_cmd, "./scripts/start.py --offense 1 --defense 0 --headless &",
                "Command executed to start the HFO server.");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - (1.0 - FLAGS_epsilon) * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return FLAGS_epsilon;
  }
}

/**
 * Converts a discrete action into a continuous HFO-action
 x*/
Action GetAction(float kickangle) {
  // CHECK_LT(kickangle, 90);
  // CHECK_GT(kickangle, -90);
  Action a;
  a = {KICK, 100., kickangle};
  return a;
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(HFOEnvironment& hfo, dqn::DQN& dqn, const double epsilon,
                      const bool update) {
  std::deque<dqn::ActorStateDataSp> past_states;
  double total_score;
  hfo_status_t status = IN_GAME;
  while (status == IN_GAME) {
    const std::vector<float>& current_state = hfo.getState();
    CHECK_EQ(current_state.size(),dqn::kStateDataSize);
    dqn::ActorStateDataSp current_state_sp = std::make_shared<dqn::ActorStateData>();
    std::copy(current_state.begin(), current_state.end(), current_state_sp->begin());
    past_states.push_back(current_state_sp);
    if (past_states.size() < dqn::kStateInputCount) {
      // If there are not past states enough for DQN input, just select DASH
      Action a;
      a = {DASH, 0., 0.};
      status = hfo.act(a);
    } else {
      while (past_states.size() > dqn::kStateInputCount) {
        past_states.pop_front();
      }
      dqn::ActorInputStates input_states;
      std::copy(past_states.begin(), past_states.end(), input_states.begin());
      const float kickangle = dqn.SelectAction(input_states, epsilon);
      Action action = GetAction(kickangle);
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
      if (update) {
        // Add the current transition to replay memory
        const std::vector<float>& next_state = hfo.getState();
        CHECK_EQ(next_state.size(),dqn::kStateDataSize);
        dqn::ActorStateDataSp next_state_sp = std::make_shared<dqn::ActorStateData>();
        std::copy(next_state.begin(), next_state.end(), next_state_sp->begin());
        const auto transition = (status == IN_GAME) ?
            dqn::Transition(input_states, kickangle, reward, next_state_sp):
            dqn::Transition(input_states, kickangle, reward, boost::none);
        dqn.AddTransition(transition);
        // If the size of replay memory is large enough, update DQN
        if (dqn.memory_size() > FLAGS_memory_threshold) {
          dqn.UpdateCritic();
          for (int u = 0; u < FLAGS_actor_update_factor; ++u) {
            dqn.UpdateActor();
          }
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
    std::tuple<std::string,std::string,std::string> snapshot =
        dqn::FindLatestSnapshot(save_path.native());
    FLAGS_actor_snapshot = std::get<0>(snapshot);
    FLAGS_critic_snapshot = std::get<1>(snapshot);
    FLAGS_memory_snapshot = std::get<2>(snapshot);
  }

  // Start the server
  CHECK_EQ(system(FLAGS_server_cmd.c_str()), 0) << "Unable to start the HFO server.";

  HFOEnvironment hfo;
  hfo.connectToAgentServer(6008);

  // Get the vector of legal actions
  std::vector<int> legal_actions(dqn::kOutputCount);
  std::iota(legal_actions.begin(), legal_actions.end(), 0);

  CHECK((FLAGS_critic_snapshot.empty() || FLAGS_critic_weights.empty()) &&
        (FLAGS_actor_snapshot.empty() || FLAGS_actor_weights.empty()))
      << "Give a snapshot to resume training or weights to finetune "
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
  } else if (!FLAGS_critic_weights.empty() || !FLAGS_actor_weights.empty()) {
    LOG(INFO) << "Actor weights finetuning from " << FLAGS_actor_weights;
    LOG(INFO) << "Critic weights finetuning from " << FLAGS_critic_weights;
    dqn.LoadTrainedModel(FLAGS_actor_weights, FLAGS_critic_weights);
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
            std::to_string(int(avg_score));
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
