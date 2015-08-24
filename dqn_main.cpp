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
DEFINE_string(memory_snapshot1, "", "The replay memory to load (*.replaymemory).");
DEFINE_string(memory_snapshot2, "", "The replay memory to load (*.replaymemory).");
// Solver Args
DEFINE_int32(actor_max_iter, 0, "Custom max iter of the actor.");
DEFINE_int32(critic_max_iter, 0, "Custom max iter of the critic.");
// Epsilon-Greedy Args
DEFINE_int32(explore, 0, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, 0, "Value of epsilon after explore iterations.");
DEFINE_double(evaluate_with_epsilon, 0, "Epsilon value to be used in evaluation mode");
// Evaluation Args
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_int32(evaluate_freq, 10000, "Frequency (steps) between evaluations");
DEFINE_int32(snapshot_freq, 10000, "Frequency (steps) snapshots");
DEFINE_int32(repeat_games, 10, "Number of games played in evaluation mode");
// HFO Args
DEFINE_string(server_cmd, "./scripts/start.py --offense-agents 1 --fullstate",
              "Command executed to start the HFO server.");
DEFINE_int32(port, 7000, "Port to use for server/client.");
// Misc Args
DEFINE_bool(warp_action, false, "Warp actions in direction of critic improvment.");
// Train Only With ReplayMemory
DEFINE_bool(pure_train, false, "Only update with the existing replaymemory");
// Evaluate critic critic network
DEFINE_bool(evaluate_critic, false, "evaluate the avg_q_values of fast and slow episodes");


double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - (1.0 - FLAGS_epsilon) * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return FLAGS_epsilon;
  }
}

/**
 * Play one episode and return the total score and number of steps
 */
std::pair<double, int> PlayOneEpisode(HFOEnvironment& hfo, dqn::DQN& dqn,
                                      const double epsilon,
                                      const bool update, float warp_action) {
  hfo.act({DASH, 0, 0});
  std::deque<dqn::StateDataSp> past_states;
  double total_score = 0;
  int steps = 0;
  status_t status = IN_GAME;
  while (status == IN_GAME) {
    const std::vector<float>& current_state = hfo.getState();
    CHECK_EQ(current_state.size(), dqn::kStateSize);
    dqn::StateDataSp current_state_sp = std::make_shared<dqn::StateData>();
    std::copy(current_state.begin(), current_state.end(), current_state_sp->begin());
    past_states.push_back(current_state_sp);
    if (past_states.size() < dqn::kStateInputCount) {
      status = hfo.act({DASH, 0, 0});
    } else {
      while (past_states.size() > dqn::kStateInputCount) {
        past_states.pop_front();
      }
      dqn::InputStates input_states;
      std::copy(past_states.begin(), past_states.end(), input_states.begin());
      dqn::ActorOutput actor_output = dqn.SelectAction(input_states, epsilon);
      Action action = dqn::GetAction(actor_output);
      VLOG(1) << "q_value: " << dqn.EvaluateAction(input_states, actor_output)
              << " Action: " << hfo.ActionToString(action);
      if (FLAGS_warp_action && warp_action > 0) {
        const dqn::ActorOutput warped_output = dqn.WarpAction(
            input_states, actor_output, 0.0, warp_action);
        const Action warped_action = dqn::GetAction(warped_output);
        VLOG(1) << "Warped Action: " << hfo.ActionToString(warped_action);
        actor_output = warped_output;
        action = warped_action;
      }
      status = hfo.act(action);
      float reward = 0;
      if (status == GOAL) {
        reward = 1; VLOG(1) << "GOAL";
      } else if (status == CAPTURED_BY_DEFENSE || status == OUT_OF_BOUNDS ||
                 status == OUT_OF_TIME) {
        reward = -1; VLOG(1) << "FAIL";
      }
      total_score += reward;
      steps++;
      if (update) {
        const std::vector<float>& next_state = hfo.getState();
        CHECK_EQ(next_state.size(), dqn::kStateSize);
        dqn::StateDataSp next_state_sp = std::make_shared<dqn::StateData>();
        std::copy(next_state.begin(), next_state.end(), next_state_sp->begin());
        const auto transition = (status == IN_GAME) ?
            dqn::Transition(input_states, actor_output, reward, next_state_sp):
            dqn::Transition(input_states, actor_output, reward, boost::none);
        dqn.AddTransition(transition);
        dqn.Update();
      }
    }
  }
  return std::make_pair(total_score, steps);
}

/**
 * Evaluate the current player
 */
double Evaluate(HFOEnvironment& hfo, dqn::DQN& dqn) {
  LOG(INFO) << "Evaluating for " << FLAGS_repeat_games
            << " episodes with epsilon = " << FLAGS_evaluate_with_epsilon;
  std::vector<double> scores;
  std::vector<int> steps;
  for (int i = 0; i < FLAGS_repeat_games; ++i) {
    std::pair<double,int> result = PlayOneEpisode(hfo, dqn, FLAGS_evaluate_with_epsilon, false, -1);
    scores.push_back(result.first);
    steps.push_back(result.second);
  }
  double total_score = 0.0;
  double total_steps = 0.0;
  for (int i = 0; i < scores.size(); ++i) {
    total_score += scores[i];
    total_steps += steps[i];
  }
  const auto avg_score = total_score / static_cast<double>(scores.size());
  const auto avg_steps = total_steps / static_cast<double>(steps.size());
  double score_stddev = 0.0; // Compute the sample standard deviation
  double steps_stddev = 0.0;
  for (int i = 0; i < scores.size(); ++i) {
    score_stddev += (scores[i] - avg_score) * (scores[i] - avg_score);
    steps_stddev += (steps[i] - avg_steps) * (steps[i] - avg_steps);
  }
  score_stddev = sqrt(score_stddev / static_cast<double>(FLAGS_repeat_games - 1));
  steps_stddev = sqrt(steps_stddev / static_cast<double>(FLAGS_repeat_games - 1));
  LOG(INFO) << "Evaluation avg_score = " << avg_score << ", std = " << score_stddev
            << ", avg_steps = " << avg_steps << ", std = " << steps_stddev;
  return avg_score;
}

void StartServer(path save_path) {
  // if (FLAGS_port < 0) {
  //   srand(std::hash<std::string>()(save_path.native()));
  //   FLAGS_port = rand() % 40000 + 20000;
  // }
  FLAGS_port += 300;
  if (FLAGS_port >= 40000)
    FLAGS_port = 7000;
  std::string cmd = FLAGS_server_cmd + " --port " + std::to_string(FLAGS_port);
  if (!FLAGS_gui) { cmd += " --headless"; }
  if (!FLAGS_evaluate) { cmd += " --no-logging"; }
  cmd += " &";
  LOG(INFO) << "Starting server with command: " << cmd;
  CHECK_EQ(system(cmd.c_str()), 0) << "Unable to start the HFO server.";
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
      && FLAGS_critic_snapshot.empty() && FLAGS_memory_snapshot1.empty()) {
    dqn::FindLatestSnapshot(save_path.native(), FLAGS_actor_snapshot,
                            FLAGS_critic_snapshot, FLAGS_memory_snapshot1);
  }

  CHECK((FLAGS_critic_snapshot.empty() || FLAGS_critic_weights.empty()) &&
        (FLAGS_actor_snapshot.empty() || FLAGS_actor_weights.empty()))
      << "Give a snapshot or weights but not both.";

  // Start rcssserver3d
  StartServer(save_path);

  HFOEnvironment hfo;
  hfo.connectToAgentServer(FLAGS_port, LOW_LEVEL_FEATURE_SET);

  // Construct the solver
  caffe::SolverParameter actor_solver_param;
  caffe::SolverParameter critic_solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_actor_solver, &actor_solver_param);
  caffe::ReadProtoFromTextFileOrDie(FLAGS_critic_solver, &critic_solver_param);
  actor_solver_param.set_snapshot_prefix((save_path.native() + "_actor").c_str());
  critic_solver_param.set_snapshot_prefix((save_path.native() + "_critic").c_str());
  if (FLAGS_actor_max_iter > 0) {
    actor_solver_param.set_max_iter(FLAGS_actor_max_iter);
  }
  if (FLAGS_critic_max_iter > 0) {
    critic_solver_param.set_max_iter(FLAGS_critic_max_iter);
  }

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
  if (!FLAGS_memory_snapshot1.empty()) {
    dqn.ClearReplayMemory();
    dqn.LoadReplayMemory(FLAGS_memory_snapshot1);
    if (!FLAGS_memory_snapshot2.empty()) {
    dqn.LoadReplayMemory(FLAGS_memory_snapshot2);
    }
  }

  if (FLAGS_evaluate) {
    Evaluate(hfo, dqn);
    return 0;
  }

  if (FLAGS_benchmark) {
    PlayOneEpisode(hfo, dqn, FLAGS_evaluate_with_epsilon, true, -1);
    dqn.Benchmark(1000);
    return 0;
  }

  if (FLAGS_evaluate_critic) {
    dqn.EvaluateCritic();
    return 0;
  }

  if (FLAGS_pure_train) {
    // LOG(INFO) << "Critic train begins!";
    // double smoothed_critic_loss = 0;
    // while (dqn.critic_iter() < critic_solver_param.max_iter()) {
    //   double loss = dqn.UpdateCritic();
    //   //     LOG(INFO) << "loss = " << loss;
    //   if (dqn.critic_iter() % 1000 == 0) {
    //     LOG(INFO) << "Critic Iteration " << dqn.critic_iter()
    //               << ", loss = " << smoothed_critic_loss
    //               << ", Optimism = " << dqn.AssessOptimism();
    //     smoothed_critic_loss = 0;
    //   }
    //   smoothed_critic_loss += loss / 1000;
    // }
    // LOG(INFO) << "Critic train finishes!";
    // dqn.Snapshot(save_path.native(), false, false);
    LOG(INFO) << "Actor train begins!";
    int iter = dqn.actor_iter();
    int episode = 0;
    while (dqn.actor_iter() < iter + actor_solver_param.max_iter()) {
      std::pair<double,int> result = PlayOneEpisode(hfo, dqn, 0, true, 0);
      LOG(INFO) << "Episode " << ++episode
                << " score = " << result.first
                << ", steps = " << result.second
                << ", actor_iter = " << dqn.actor_iter()
                << ", critic_iter = " << dqn.critic_iter()
                << ", replay_mem_size = " << dqn.memory_size();
      // float avg_q = dqn.UpdateActor();
      // if ((dqn.actor_iter() - iter) % 1000 == 0) {
      //   LOG(INFO) << "Actor Iteration " << dqn.actor_iter() - iter
      //             << ", avg_q_value = " << avg_q;
      // }
      // if ((dqn.actor_iter() - iter) % 10000 == 0) {
      //   StartServer(save_path);
      //   HFOEnvironment hfo_ev;
      //   hfo_ev.connectToAgentServer(FLAGS_port, LOW_LEVEL_FEATURE_SET);
      //   Evaluate(hfo_ev,dqn);
      // }
    }
    LOG(INFO) << "Actor train finishes!";
    Evaluate(hfo,dqn);
    dqn.Snapshot(save_path.native(), false, false);
    LOG(INFO) << "Successfully finish finetuning actor!";
    return 0;
  }

  int last_eval_iter = 0;
  int last_snapshot_iter = 0;
  int episode = 0;
  float warp_level = FLAGS_warp_action ? 1.0 : -1.0; // How much to warp actions by
  double best_score = std::numeric_limits<double>::min();
  while (dqn.actor_iter() < actor_solver_param.max_iter() &&
         dqn.critic_iter() < critic_solver_param.max_iter() &&
         dqn.memory_size() < 500010) {
    double epsilon = CalculateEpsilon(dqn.max_iter());
    std::pair<double,int> result = PlayOneEpisode(hfo, dqn, epsilon, true, warp_level);
    LOG(INFO) << "Episode " << episode << " score = " << result.first
              << ", steps = " << result.second
              << ", epsilon = " << epsilon
              << ", actor_iter = " << dqn.actor_iter()
              << ", critic_iter = " << dqn.critic_iter()
              << ", replay_mem_size = " << dqn.memory_size();
    if (FLAGS_warp_action) {
      LOG(INFO) << "Episode " << episode << ", warp_level = " << warp_level;
      warp_level *= result.first > 0 ? 2.0 : 0.5; // Warp to 50% success rate
    }
    episode++;
    if (episode % 1000 == 0)
      dqn.Snapshot(save_path.native(), true, true);
    // if (dqn.actor_iter() >= last_eval_iter + FLAGS_evaluate_freq) {
    //   double avg_score = Evaluate(hfo, dqn);
    //   if (avg_score > best_score) {
    //     LOG(INFO) << "New High Score: " << avg_score
    //               << ", actor_iter = " << dqn.actor_iter()
    //               << ", critic_iter = " << dqn.critic_iter();
    //     best_score = avg_score;
    //     std::string fname = save_path.native() + "_HiScore" +
    //         std::to_string(avg_score);
    //     dqn.Snapshot(fname, false, false);
    //   }
    //   last_eval_iter = dqn.actor_iter();
    // }
    // if (dqn.max_iter() >= last_snapshot_iter + FLAGS_snapshot_freq) {
    //   dqn.Snapshot(save_path.native(), true, true);
    //   last_snapshot_iter = dqn.max_iter();
    // }
  }
  dqn.Snapshot(save_path.native(), true, true);
  // if (dqn.actor_iter() > last_eval_iter) {
  //   Evaluate(hfo, dqn);
  // }
  // if (dqn.max_iter() > last_snapshot_iter) {
  //   dqn.Snapshot(save_path.native(), true, true);
  // }
};
