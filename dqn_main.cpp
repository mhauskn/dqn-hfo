#include <iostream>
#include <HFO.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "dqn.hpp"
#include "hfo_game.hpp"
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
DEFINE_bool(benchmark, false, "Benchmark the network and exit");
// Load/Save Args
DEFINE_string(save, "", "Prefix for saving snapshots");
DEFINE_string(resume, "", "Prefix for resuming from. Default=save_path");
DEFINE_string(actor_solver, "actor_solver.prototxt", "Actor solver (*.prototxt)");
DEFINE_string(critic_solver, "critic_solver.prototxt", "Critic solver (*.prototxt)");
DEFINE_string(actor_weights, "", "The actor pretrained weights load (*.caffemodel).");
DEFINE_string(critic_weights, "", "The critic pretrained weights load (*.caffemodel).");
DEFINE_string(actor_snapshot, "", "The actor solver state to load (*.solverstate).");
DEFINE_string(critic_snapshot, "", "The critic solver state to load (*.solverstate).");
DEFINE_string(memory_snapshot, "", "The replay memory to load (*.replaymemory).");
// Solver Args
DEFINE_string(solver, "Adam", "Solver Type.");
DEFINE_double(momentum, .95, "Solver momentum.");
DEFINE_double(momentum2, .999, "Solver momentum2.");
DEFINE_double(actor_lr, .00001, "Solver learning rate.");
DEFINE_double(critic_lr, .001, "Solver learning rate.");
DEFINE_double(clip_grad, 10, "Clip gradients.");
DEFINE_string(lr_policy, "fixed", "LR Policy.");
DEFINE_int32(max_iter, 10000000, "Custom max iter.");
// Epsilon-Greedy Args
DEFINE_int32(explore, 10000, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, .1, "Value of epsilon after explore iterations.");
DEFINE_double(evaluate_with_epsilon, 0, "Epsilon value to be used in evaluation mode");
// Evaluation Args
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_int32(evaluate_freq, 10000, "Frequency (steps) between evaluations");
DEFINE_int32(repeat_games, 10, "Number of games played in evaluation mode");
// Misc Args
DEFINE_double(update_ratio, 0.1, "Ratio of new experiences to updates.");
// Game configuration
DEFINE_int32(offense_agents, 1, "Number of agents playing offense");
DEFINE_int32(offense_npcs, 0, "Number of npcs playing offense");
DEFINE_int32(defense_agents, 0, "Number of agents playing defense");
DEFINE_int32(defense_npcs, 0, "Number of npcs playing defense");


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
                                      const bool update) {
  HFOGameState game(hfo);
  hfo.act(DASH, 0, 0);
  game.update(hfo.getState(), hfo.step());
  std::deque<dqn::StateDataSp> past_states;
  while (!game.episode_over) {
    const std::vector<float>& current_state = hfo.getState();
    CHECK_EQ(current_state.size(), dqn.state_size());
    dqn::StateDataSp current_state_sp
        = std::make_shared<dqn::StateData>(dqn.state_size());
    std::copy(current_state.begin(), current_state.end(), current_state_sp->begin());
    past_states.push_back(current_state_sp);
    if (past_states.size() < dqn::kStateInputCount) {
      hfo.act(DASH, 0, 0);
    } else {
      while (past_states.size() > dqn::kStateInputCount) {
        past_states.pop_front();
      }
      dqn::InputStates input_states;
      std::copy(past_states.begin(), past_states.end(), input_states.begin());
      dqn::ActorOutput actor_output = dqn.SelectAction(input_states, epsilon);
      VLOG(1) << "Actor_output: " << dqn::PrintActorOutput(actor_output);
      Action action = dqn::GetAction(actor_output);
      VLOG(1) << "q_value: " << dqn.EvaluateAction(input_states, actor_output)
              << " Action: " << hfo::ActionToString(action.action);
      hfo.act(action.action, action.arg1, action.arg2);
      status_t status = hfo.step();
      game.update(current_state, status);
      float reward = game.reward();
      if (update) {
        const std::vector<float>& next_state = hfo.getState();
        CHECK_EQ(next_state.size(), dqn.state_size());
        dqn::StateDataSp next_state_sp
            = std::make_shared<dqn::StateData>(dqn.state_size());
        std::copy(next_state.begin(), next_state.end(), next_state_sp->begin());
        const auto transition = (game.status == IN_GAME) ?
            dqn::Transition(input_states, actor_output, reward, next_state_sp):
            dqn::Transition(input_states, actor_output, reward, boost::none);
        dqn.AddTransition(transition);
      }
    }
  }
  LOG(INFO) << "Status " << game.status;
  return std::make_pair(game.total_reward, game.steps);
}

template <class T>
std::pair<double,double> get_avg_std(std::vector<T> data) {
  double sum = 0;
  for (int i = 0; i < data.size(); ++i) {
    sum += data[i];
  }
  double avg = sum / static_cast<double>(data.size());
  double std = 0;
  for (int i = 0; i < data.size(); ++i) {
    std += (data[i] - avg) * (data[i] - avg);
  }
  std = sqrt(std / static_cast<double>(data.size() - 1));
  return std::make_pair(avg, std);
}

/**
 * Evaluate the current player
 */
double Evaluate(HFOEnvironment& hfo, dqn::DQN& dqn) {
  LOG(INFO) << "Evaluating for " << FLAGS_repeat_games
            << " episodes with epsilon = " << FLAGS_evaluate_with_epsilon;
  std::vector<double> scores;
  std::vector<int> steps;
  std::vector<int> successful_trial_steps;
  for (int i = 0; i < FLAGS_repeat_games; ++i) {
    std::pair<double,int> result = PlayOneEpisode(
        hfo, dqn, FLAGS_evaluate_with_epsilon, false);
    scores.push_back(result.first);
    steps.push_back(result.second);
    if (result.first > 0) {
      successful_trial_steps.push_back(result.second);
    }
  }
  std::pair<double, double> score_dist = get_avg_std(scores);
  std::pair<double, double> steps_dist = get_avg_std(steps);
  std::pair<double, double> succ_steps_dist = get_avg_std(successful_trial_steps);
  LOG(INFO) << "Evaluation: "
            << " actor_iter = " << dqn.actor_iter()
            << ", avg_score = " << score_dist.first
            << ", score_std = " << score_dist.second
            << ", avg_steps = " << steps_dist.first
            << ", steps_std = " << steps_dist.second
            << ", success_avg_steps = " << succ_steps_dist.first
            << ", success_steps_std = " << succ_steps_dist.second;
  return score_dist.first;
}

void KeepPlayingGames(int tid, std::string save_prefix, int port, int unum) {
  LOG(INFO) << "Thread " << tid << ", port=" << port << ", unum=" << unum
            << ", save_prefix=" << save_prefix;
  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
  // Look for a recent snapshot to resume
  std::string resume_path = FLAGS_resume.empty() ? save_prefix : FLAGS_resume;
  std::string last_actor_snapshot, last_critic_snapshot, last_memory_snapshot;
  dqn::FindLatestSnapshot(resume_path, last_actor_snapshot,
                          last_critic_snapshot, last_memory_snapshot);
  LOG(INFO) << "Found Resumable(s): [" << resume_path << "] "
            << last_actor_snapshot << ", " << last_critic_snapshot
            << ", " << last_memory_snapshot;
  if (FLAGS_critic_snapshot.empty() && FLAGS_critic_weights.empty()) {
    FLAGS_critic_snapshot = last_critic_snapshot;
  }
  if (FLAGS_actor_snapshot.empty() && FLAGS_actor_weights.empty()) {
    FLAGS_actor_snapshot = last_actor_snapshot;
  }
  if (FLAGS_memory_snapshot.empty()) {
    FLAGS_memory_snapshot = last_memory_snapshot;
  }
  CHECK((FLAGS_critic_snapshot.empty() || FLAGS_critic_weights.empty()) &&
        (FLAGS_actor_snapshot.empty() || FLAGS_actor_weights.empty()))
      << "Give a snapshot or weights but not both.";
  int num_features = NumStateFeatures(FLAGS_offense_agents,
                                      FLAGS_offense_npcs,
                                      FLAGS_defense_agents,
                                      FLAGS_defense_npcs);
  // Construct the solver
  caffe::SolverParameter actor_solver_param;
  caffe::SolverParameter critic_solver_param;
  caffe::NetParameter* actor_net_param = actor_solver_param.mutable_net_param();
  std::string actor_net_filename = save_prefix + "_actor.prototxt";
  if (boost::filesystem::is_regular_file(actor_net_filename)) {
    caffe::ReadProtoFromTextFileOrDie(actor_net_filename.c_str(), actor_net_param);
  } else {
    actor_net_param->CopyFrom(dqn::CreateActorNet(num_features));
    WriteProtoToTextFile(*actor_net_param, actor_net_filename.c_str());
  }
  caffe::NetParameter* critic_net_param = critic_solver_param.mutable_net_param();
  std::string critic_net_filename = save_prefix + "_critic.prototxt";
  if (boost::filesystem::is_regular_file(critic_net_filename)) {
    caffe::ReadProtoFromTextFileOrDie(critic_net_filename.c_str(), critic_net_param);
  } else {
    critic_net_param->CopyFrom(dqn::CreateCriticNet(num_features));
    WriteProtoToTextFile(*critic_net_param, critic_net_filename.c_str());
  }
  actor_solver_param.set_snapshot_prefix((save_prefix + "_actor").c_str());
  critic_solver_param.set_snapshot_prefix((save_prefix + "_critic").c_str());
  actor_solver_param.set_max_iter(FLAGS_max_iter);
  critic_solver_param.set_max_iter(FLAGS_max_iter);
  actor_solver_param.set_type(FLAGS_solver);
  critic_solver_param.set_type(FLAGS_solver);
  actor_solver_param.set_base_lr(FLAGS_actor_lr);
  critic_solver_param.set_base_lr(FLAGS_critic_lr);
  actor_solver_param.set_lr_policy(FLAGS_lr_policy);
  critic_solver_param.set_lr_policy(FLAGS_lr_policy);
  actor_solver_param.set_momentum(FLAGS_momentum);
  critic_solver_param.set_momentum(FLAGS_momentum);
  actor_solver_param.set_momentum2(FLAGS_momentum2);
  critic_solver_param.set_momentum2(FLAGS_momentum2);
  actor_solver_param.set_clip_gradients(FLAGS_clip_grad);
  critic_solver_param.set_clip_gradients(FLAGS_clip_grad);

  dqn::DQN* dqn = new dqn::DQN(actor_solver_param, critic_solver_param,
                                 save_prefix, num_features);
  // TODO: Agent-specific loading of actor/critic/memory
  // Load actor/critic/memory
  if (!FLAGS_actor_snapshot.empty()) {
    dqn->RestoreActorSolver(FLAGS_actor_snapshot);
  } else if (!FLAGS_actor_weights.empty()) {
    dqn->LoadActorWeights(FLAGS_actor_weights);
  }
  if (!FLAGS_critic_snapshot.empty()) {
    dqn->RestoreCriticSolver(FLAGS_critic_snapshot);
  } else if (!FLAGS_critic_weights.empty()) {
    dqn->LoadCriticWeights(FLAGS_critic_weights);
  }
  if (!FLAGS_memory_snapshot.empty()) {
    dqn->LoadReplayMemory(FLAGS_memory_snapshot);
  }

  HFOEnvironment env;
  ConnectToServer(env, port, unum);
  if (FLAGS_evaluate) {
    Evaluate(env, *dqn);
    return;
  }
  if (FLAGS_benchmark) {
    PlayOneEpisode(env, *dqn, FLAGS_evaluate_with_epsilon, true);
    dqn->Benchmark(1000);
    return;
  }
  int last_eval_iter = 0;
  double best_score = std::numeric_limits<double>::min();
  for (int episode = 0; dqn->max_iter() < FLAGS_max_iter; ++episode) {
    double epsilon = CalculateEpsilon(dqn->max_iter());
    std::pair<double,int> result = PlayOneEpisode(env, *dqn, epsilon, true);
    int steps = result.second;
    LOG(INFO) << "[Agent" << tid <<"] Episode " << episode
              << " score = " << result.first;
              // << ", steps = " << result.second
              // << ", epsilon = " << epsilon
              // << ", actor_iter = " << dqn->actor_iter()
              // << ", critic_iter = " << dqn->critic_iter()
              // << ", replay_mem_size = " << dqn->memory_size();
    int n_updates = int(steps * FLAGS_update_ratio);
    for (int i=0; i<n_updates; ++i) {
      dqn->Update();
    }
    if (dqn->actor_iter() >= last_eval_iter + FLAGS_evaluate_freq) {
      double avg_score = Evaluate(env, *dqn);
      if (avg_score > best_score) {
        LOG(INFO) << "[Agent " << tid << "] New High Score: " << avg_score
                  << ", actor_iter = " << dqn->actor_iter()
                  << ", critic_iter = " << dqn->critic_iter();
        best_score = avg_score;
        dqn::RemoveFilesMatchingRegexp(dqn->save_path() + "_HiScore.*");
        std::string fname = dqn->save_path() + "_HiScore" + std::to_string(avg_score);
        dqn->Snapshot(fname, false, false);
      }
      last_eval_iter = dqn->actor_iter();
    }
  }
  dqn->Snapshot();
  Evaluate(env, *dqn);
  delete dqn;
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
  google::SetLogDestination(google::GLOG_INFO, (save_path.native() + "_INFO_").c_str());
  google::SetLogDestination(google::GLOG_WARNING, (save_path.native() + "_WARNING_").c_str());
  google::SetLogDestination(google::GLOG_ERROR, (save_path.native() + "_ERROR_").c_str());
  google::SetLogDestination(google::GLOG_FATAL, (save_path.native() + "_FATAL_").c_str());
  srand(std::hash<std::string>()(save_path.native()));
  int port = rand() % 40000 + 20000;
  StartHFOServer(port, FLAGS_offense_agents, FLAGS_offense_npcs,
                 FLAGS_defense_agents, FLAGS_defense_npcs);
  std::thread player_threads[FLAGS_offense_agents];
  std::vector<int> offense_unums = {11,7,8,9,10,6,3,2,4,5};
  std::sort(offense_unums.begin(), offense_unums.begin() + FLAGS_offense_agents);
  for (int i=0; i<FLAGS_offense_agents; ++i) {
    std::string save_prefix = save_path.native() + "_agent" + std::to_string(i);
    player_threads[i] = std::thread(KeepPlayingGames, i, save_prefix, port,
                                    offense_unums[i]);
    sleep(10);
  }
  for (int i = 0; i < FLAGS_offense_agents; ++i) {
    player_threads[i].join();
  }
  StopHFOServer();
};
