#include <iostream>
#include <HFO.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "dqn.hpp"
#include "tasks/task.hpp"
#include "curriculum.hpp"
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
DEFINE_bool(learn_offline, false, "Just do updates on a fixed replaymemory.");
// Load/Save Args
DEFINE_string(save, "", "Prefix for saving snapshots");
DEFINE_string(resume, "", "Prefix for resuming from. Default=save_path");
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
DEFINE_int32(repeat_games, 100, "Number of games played in evaluation mode");
// Misc Args
DEFINE_double(update_ratio, 0.1, "Ratio of new experiences to updates.");
// Sharing
DEFINE_int32(share_actor_layers, 0, "Share layers between actor networks.");
DEFINE_int32(share_critic_layers, 0, "Share layers between critic networks.");
DEFINE_bool(share_replay_memory, false, "Shares replay memory between agents.");
// Game configuration
DEFINE_int32(offense_agents, 1, "Number of agents playing offense");
DEFINE_int32(offense_npcs, 0, "Number of npcs playing offense");
DEFINE_int32(defense_agents, 0, "Number of agents playing defense");
DEFINE_int32(defense_npcs, 0, "Number of npcs playing defense");
DEFINE_int32(offense_dummies, 0, "Number of dummy npcs playing offense");
DEFINE_int32(defense_dummies, 0, "Number of dummy npcs playing defense");
DEFINE_int32(defense_chasers, 0, "Number of chasers playing defense");
DEFINE_string(tasks, "", "Tasks to use.");
DEFINE_string(curriculum, "random", "Curriculum to use.");
// Communication
DEFINE_int32(comm_actions, 0, "Number of continuous actions used for communication.");
DEFINE_bool(teammate_comm_gradients, false, "Teammate specifies gradients to comm actions");

// Global Variables Shared Between Threads
dqn::DQN* DQNS[12]; // Pointers to all DQNs. We will never have >12 players
std::mutex MTX;
std::vector<float> EVAL_PERF; // Used to compute an average evaluation performance
boost::barrier* BAR; // Barrier pointer
std::vector<int> INT_VEC;
std::vector<float*> FLOAT_VEC;

int NumStateFeatures(int num_players) {
  return 50 + 8 * (num_players);
}

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - (1.0 - FLAGS_epsilon) * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return FLAGS_epsilon;
  }
}

std::string GetArg(std::string input, int indx) {
  std::istringstream ss(input);
  std::string token;
  int i = 0;
  while(std::getline(ss, token, ',')) {
    if (i++ == indx) {
      return token;
    }
  }
  return "";
}

/**
 * Play one episode and return the total score and number of steps
 */
std::tuple<float, int, status_t> PlayOneEpisode(dqn::DQN& dqn,
                                                const double epsilon,
                                                const bool update,
                                                const int tid,
                                                Task& task) {
  std::vector<dqn::Transition> episode;
  int task_id = task.getID();
  VLOG(1) << "Task " << task.getName() << " ID " << task_id;
  HFOEnvironment& env = task.getEnv(tid);
  env.act(NOOP);
  task.step(tid);
  CHECK(!task.episodeOver()) << "Task " << task.getName() << " episode over!";
  std::deque<dqn::StateDataSp> past_states;
  int steps = 0;
  while (!task.episodeOver()) {
    const std::vector<float>& current_state = task.getState(tid);
    CHECK_LE(current_state.size(), dqn.state_size());
    dqn::StateDataSp current_state_sp
        = std::make_shared<dqn::StateData>(dqn.state_size());
    std::copy(current_state.begin(), current_state.end(), current_state_sp->begin());
    if (FLAGS_comm_actions > 0) {
      const std::vector<float>& hear_state = dqn.GetHearFeatures(env);
      CHECK_EQ(hear_state.size(), FLAGS_comm_actions);
      std::copy(hear_state.begin(), hear_state.end(),
                current_state_sp->end() - FLAGS_comm_actions);
    }
    past_states.push_back(current_state_sp);
    CHECK_GE(past_states.size(), dqn::kStateInputCount);
    while (past_states.size() > dqn::kStateInputCount) {
      past_states.pop_front();
    }
    dqn::InputStates input_states;
    std::copy(past_states.begin(), past_states.end(), input_states.begin());
    dqn::ActorOutput actor_output = dqn.SelectAction(input_states, task_id, epsilon);
    VLOG(1) << "Actor_output: " << dqn.PrintActorOutput(actor_output);
    Action action = dqn.GetAction(actor_output);
    task.act(tid, action.action, action.arg1, action.arg2);
    if (FLAGS_comm_actions > 0) {
      task.say(tid, dqn.GetSayMsg(actor_output));
    }
    float reward = task.step(tid).second;
    steps++;
    VLOG(1) << "q_value: " << dqn.EvaluateAction(input_states, task_id, actor_output)
            << " Action: " << hfo::ActionToString(action.action)
            << " Reward: " << reward;
    if (update) {
      const std::vector<float>& next_state = task.getState(tid);
      CHECK_LE(next_state.size(), dqn.state_size());
      dqn::StateDataSp next_state_sp
          = std::make_shared<dqn::StateData>(dqn.state_size(), 0);
      std::copy(next_state.begin(), next_state.end(), next_state_sp->begin());
      if (FLAGS_comm_actions > 0) {
        const std::vector<float>& hear_state = dqn.GetHearFeatures(env);
        CHECK_EQ(hear_state.size(), FLAGS_comm_actions);
        std::copy(hear_state.begin(), hear_state.end(),
                  next_state_sp->end() - FLAGS_comm_actions);
      }
      const auto transition = task.episodeOver() ?
          dqn::Transition(input_states, task_id, actor_output, reward, 0, boost::none) :
          dqn::Transition(input_states, task_id, actor_output, reward, 0, next_state_sp);
      episode.push_back(transition);
    }
  }
  if (update) {
    if (FLAGS_share_replay_memory) { MTX.lock(); }
    dqn.LabelTransitions(episode);
    dqn.AddTransitions(episode);
    if (FLAGS_share_replay_memory) { MTX.unlock(); }
  }
  CHECK(task.episodeOver());
  return std::make_tuple(task.getEpisodeReward(tid),
                         steps,
                         task.getStatus(tid));
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

float Evaluate(dqn::DQN& dqn, int tid, Task& task) {
  LOG(INFO) << "[Agent" << tid << "] Evaluating for " << FLAGS_repeat_games
            << " episodes with epsilon = " << FLAGS_evaluate_with_epsilon
            << " task = " << task.getName();
  std::vector<double> scores;
  std::vector<int> steps;
  std::vector<int> successful_trial_steps;
  int goals = 0;
  for (int i = 0; i < FLAGS_repeat_games; ++i) {
    auto result = PlayOneEpisode(dqn, FLAGS_evaluate_with_epsilon, false, tid, task);
    double trial_reward = std::get<0>(result);
    int trial_steps = std::get<1>(result);
    status_t trial_status = std::get<2>(result);
    scores.push_back(trial_reward);
    steps.push_back(trial_steps);
    if (trial_status == GOAL) {
      goals++;
      successful_trial_steps.push_back(trial_steps);
    }
  }
  std::pair<double, double> score_dist = get_avg_std(scores);
  std::pair<double, double> steps_dist = get_avg_std(steps);
  std::pair<double, double> succ_steps_dist = get_avg_std(successful_trial_steps);
  double avg_reward = score_dist.first;
  float goal_percent = goals / float(FLAGS_repeat_games);
  float perf = avg_reward / task.getMaxExpectedReward();
  LOG(INFO) << "[Agent" << tid << "] Evaluation: "
            << "actor_iter = " << dqn.actor_iter()
            << ", avg_reward = " << avg_reward
            << ", reward_std = " << score_dist.second
            << ", avg_steps = " << steps_dist.first
            << ", steps_std = " << steps_dist.second
            << ", success_steps = " << succ_steps_dist.first
            << ", success_std = " << succ_steps_dist.second
            << ", goal_perc = " << goal_percent
            << ", task = " << task.getName()
            << ", performance = " << perf;
  return perf;
}

double Evaluate(dqn::DQN& dqn, int tid, Curriculum& tasks) {
  std::vector<double> task_perf;
  for (Task* t : tasks.getTasks()) {
    float perf = Evaluate(dqn, tid, *t);
    tasks.addEvalPerf(*t, perf, tid);
    task_perf.push_back(perf);
  }
  return get_avg_std(task_perf).first;
}

void KeepPlayingGames(int tid, std::string save_prefix, Curriculum& tasks) {
  LOG(INFO) << "Thread " << tid << ", save_prefix=" << save_prefix;

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }
  // Look for a recent snapshot to resume
  std::string resume_path = GetArg(FLAGS_resume, tid).empty() ?
      save_prefix : GetArg(FLAGS_resume, tid);
  std::string last_actor_snapshot, last_critic_snapshot, last_memory_snapshot;
  dqn::FindLatestSnapshot(resume_path, last_actor_snapshot,
                          last_critic_snapshot, last_memory_snapshot);
  LOG(INFO) << "Found Resumable(s): [" << resume_path << "] "
            << last_actor_snapshot << ", " << last_critic_snapshot
            << ", " << last_memory_snapshot;
  CHECK((FLAGS_critic_snapshot.empty() || FLAGS_critic_weights.empty()) &&
        (FLAGS_actor_snapshot.empty() || FLAGS_actor_weights.empty()))
      << "Give a snapshot or weights but not both.";
  int num_features = NumStateFeatures(3) + FLAGS_comm_actions; // Enough for 2 agents and comm
  int num_discrete_actions = 3; // Dash, Turn, Kick
  int num_continuous_actions = dqn::kHFOParams + FLAGS_comm_actions;
  int num_tasks = tasks.getTasks().size();
  // Construct the solver
  caffe::SolverParameter actor_solver_param;
  caffe::SolverParameter critic_solver_param;
  caffe::NetParameter* actor_net_param = actor_solver_param.mutable_net_param();
  std::string actor_net_filename = save_prefix + "_actor.prototxt";
  if (boost::filesystem::is_regular_file(actor_net_filename)) {
    caffe::ReadProtoFromTextFileOrDie(actor_net_filename.c_str(), actor_net_param);
  } else {
    actor_net_param->CopyFrom(dqn::CreateActorNet(
        num_features, num_discrete_actions, num_continuous_actions, num_tasks));
    WriteProtoToTextFile(*actor_net_param, actor_net_filename.c_str());
  }
  caffe::NetParameter* critic_net_param = critic_solver_param.mutable_net_param();
  std::string critic_net_filename = save_prefix + "_critic.prototxt";
  if (boost::filesystem::is_regular_file(critic_net_filename)) {
    caffe::ReadProtoFromTextFileOrDie(critic_net_filename.c_str(), critic_net_param);
  } else {
    critic_net_param->CopyFrom(dqn::CreateCriticNet(
        num_features, num_discrete_actions, num_continuous_actions, num_tasks));
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

  std::unique_ptr<dqn::DQN> dqn(new dqn::DQN(actor_solver_param,
                                             critic_solver_param,
                                             save_prefix, num_features, tid,
                                             num_discrete_actions,
                                             num_continuous_actions));

  // Load actor/critic/memory. Try to load from resumables
  // first. Otherwise load from the args.
  if (!last_actor_snapshot.empty()) {
    dqn->RestoreActorSolver(last_actor_snapshot);
  } else if (!GetArg(FLAGS_actor_snapshot, tid).empty()) {
    dqn->RestoreActorSolver(GetArg(FLAGS_actor_snapshot, tid));
  } else if (!GetArg(FLAGS_actor_weights, tid).empty()) {
    dqn->LoadActorWeights(GetArg(FLAGS_actor_weights, tid));
  }
  if (!last_critic_snapshot.empty()) {
    dqn->RestoreCriticSolver(last_critic_snapshot);
  } else if (!GetArg(FLAGS_critic_snapshot, tid).empty()) {
    dqn->RestoreCriticSolver(GetArg(FLAGS_critic_snapshot, tid));
  } else if (!GetArg(FLAGS_critic_weights, tid).empty()) {
    dqn->LoadCriticWeights(GetArg(FLAGS_critic_weights, 0));
  }
  if (!last_memory_snapshot.empty()) {
    dqn->LoadReplayMemory(last_memory_snapshot);
  } else if (!GetArg(FLAGS_memory_snapshot, tid).empty()) {
    dqn->LoadReplayMemory(GetArg(FLAGS_memory_snapshot, tid));
  }

  // Wait for all DQNs to connect and be ready
  DQNS[tid] = dqn.get();
  bool all_dqns_ready = false;
  while (!all_dqns_ready) {
    little_sleep(std::chrono::microseconds(100));
    all_dqns_ready = true;
    for (int i=0; i<FLAGS_offense_agents; ++i) {
      if (DQNS[i] == NULL) {
        all_dqns_ready = false;
      }
    }
  }
  // TID 0 is the Master and will do the sharing
  if (tid == 0) {
    if (FLAGS_share_actor_layers > 0 || FLAGS_share_critic_layers > 0) {
      for (int i=1; i<FLAGS_offense_agents; ++i) {
        dqn::DQN* teammate = DQNS[i];
        CHECK_NOTNULL(teammate);
        dqn->ShareParameters(*teammate,
                             FLAGS_share_actor_layers,
                             FLAGS_share_critic_layers);
      }
    }
    if (FLAGS_share_replay_memory) {
      for (int i=1; i<FLAGS_offense_agents; ++i) {
        dqn::DQN* teammate = DQNS[i];
        CHECK_NOTNULL(teammate);
        dqn->ShareReplayMemory(*teammate);
      }
    }
  }

  if (FLAGS_evaluate) {
    Evaluate(*dqn, tid, tasks);
    return;
  }
  if (FLAGS_benchmark) {
    PlayOneEpisode(*dqn, FLAGS_evaluate_with_epsilon, true, tid, tasks.getTask(tid));
    dqn->Benchmark(1000);
    return;
  }
  if (FLAGS_learn_offline) {
    while (dqn->max_iter() < FLAGS_max_iter) {
      dqn->Update();
    }
    dqn->Snapshot();
    return;
  }
  int last_eval_iter = dqn->max_iter();
  float best_perf = std::numeric_limits<float>::min();
  for (int episode = 0; dqn->max_iter() < FLAGS_max_iter; ++episode) {
    double epsilon = CalculateEpsilon(dqn->max_iter());
    Task& task = tasks.getTask(tid);
    auto result = PlayOneEpisode(*dqn, epsilon, true, tid, task);
    LOG(INFO) << "[Agent" << tid <<"] Episode " << episode
              << ", reward = " << std::get<0>(result)
              << ", steps = " << std::get<1>(result)
              << ", status = " << hfo::StatusToString(std::get<2>(result))
              << ", task = " << task.getName()
              << ", id = " << task.getID();
    int steps = std::get<1>(result);
    int n_updates = int(steps * FLAGS_update_ratio);
    if (FLAGS_share_replay_memory) { MTX.lock(); }
    for (int i=0; i<n_updates; ++i) {
      if (FLAGS_teammate_comm_gradients) {
        dqn->SynchronizedUpdate(*BAR, INT_VEC, FLOAT_VEC);
      } else {
        dqn->Update();
      }
    }
    if (FLAGS_share_replay_memory) { MTX.unlock(); }
    if (dqn->actor_iter() >= last_eval_iter + FLAGS_evaluate_freq) {
      float perf = Evaluate(*dqn, tid, tasks);
      MTX.lock(); // Accumulate Evaluation Scores for all agents
      EVAL_PERF.push_back(perf);
      MTX.unlock();
      if (tid == 0) { // Tid[0] in charge of saving hi-scores
        while (EVAL_PERF.size() < FLAGS_offense_agents) {
          little_sleep(std::chrono::microseconds(100));
        }
        float avg_perf = std::accumulate(EVAL_PERF.begin(), EVAL_PERF.end(), 0.0)
            / float(EVAL_PERF.size());
        EVAL_PERF.clear();
        if (avg_perf > best_perf) {
          LOG(INFO) << "NewHighPerf: " << avg_perf
                    << ", actor_iter = " << dqn->actor_iter()
                    << ", critic_iter = " << dqn->critic_iter();
          best_perf = avg_perf;
          for (int i=0; i<FLAGS_offense_agents; ++i) {
            dqn::DQN* d = DQNS[i];
            dqn::RemoveFilesMatchingRegexp(d->save_path() + "_HiPerf_.*");
            std::string fname = d->save_path() + "_HiPerf_" + std::to_string(avg_perf);
            d->Snapshot(fname, false, false);
          }
        }
      }
      last_eval_iter = dqn->actor_iter();
    }
  }
  dqn->Snapshot();
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
  for (int i=0; i<12; ++i) { // Make the global pointers all null
    DQNS[i] = NULL;
  }
  BAR = new boost::barrier(FLAGS_offense_agents + FLAGS_defense_agents);
  unsigned int seed = std::hash<std::string>()(save_path.native());
  srand(seed);
  Curriculum* tasks = Curriculum::getCurriculum(
      FLAGS_curriculum, FLAGS_offense_agents + FLAGS_defense_agents, seed);
  CHECK(!FLAGS_tasks.empty()) << "Specify one or more tasks to perform.";
  for (int i=0; !GetArg(FLAGS_tasks, i).empty(); ++i) {
    int port = rand() % 40000 + 20000;
    std::string task = GetArg(FLAGS_tasks, i);
    LOG(INFO) << "Adding Task " << task;
    tasks->addTask(task, port, FLAGS_offense_agents, FLAGS_defense_agents);
  }
  std::vector<std::thread> player_threads;
  for (int i=0; i<FLAGS_offense_agents; ++i) {
    std::string save_prefix = save_path.native() + "_agent" + std::to_string(i);
    player_threads.emplace_back(KeepPlayingGames, i, save_prefix, std::ref(*tasks));
    sleep(5);
  }
  for (std::thread& t : player_threads) {
    t.join();
  }
  delete tasks;
  delete BAR;
};
