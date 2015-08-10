#include "dqn.hpp"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <boost/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <glog/logging.h>
#include <chrono>

namespace dqn {

using namespace hfo;

// DQN Args
DEFINE_int32(seed, 0, "Seed the RNG. Default: time");
DEFINE_int32(clone_freq, 10000, "Frequency (steps) of cloning the target network.");
DEFINE_double(gamma, .99, "Discount factor of future rewards (0,1]");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(memory_threshold, 10000, "Number of transitions to start learning");
DEFINE_int32(loss_display_iter, 1000, "Frequency of loss display");
DEFINE_bool(update_actor, true, "Perform updates on actor.");
DEFINE_bool(update_critic, true, "Perform updates on critic.");

template <typename Dtype>
void HasBlobSize(caffe::Net<Dtype>& net,
                 const std::string& blob_name,
                 const std::vector<int> expected_shape) {
  CHECK(net.has_blob(blob_name)) << "Net does not have blob named " << blob_name;
  const caffe::Blob<Dtype>& blob = *net.blob_by_name(blob_name);
  const std::vector<int>& blob_shape = blob.shape();
  CHECK_EQ(blob_shape.size(), expected_shape.size())
      << "Blob \"" << blob_name << "\" has " << blob_shape.size()
      << " dimensions. Expected " << expected_shape.size();
  CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
                   expected_shape.begin()))
      << "Blob \"" << blob_name << "\" failed dimension check.";
}

// Returns the index of the layer matching the given layer_name or -1
// if no such layer exists.
template <typename Dtype>
int GetLayerIndex(caffe::Net<Dtype>& net, const std::string& layer_name) {
  if (!net.has_layer(layer_name)) {
    return -1;
  }
  const std::vector<std::string>& layer_names = net.layer_names();
  int indx = std::distance(
      layer_names.begin(),
      std::find(layer_names.begin(), layer_names.end(), layer_name));
  return indx;
}

// Zeros the gradients accumulated by each forward/backward pass.
template <typename Dtype>
void ZeroGradParameters(caffe::Net<Dtype>& net) {
  for (int i = 0; i < net.params().size(); ++i) {
    caffe::shared_ptr<caffe::Blob<Dtype> > blob = net.params()[i];
    switch (caffe::Caffe::mode()) {
      case caffe::Caffe::CPU:
        caffe::caffe_set(blob->count(), static_cast<Dtype>(0),
                         blob->mutable_cpu_diff());
        break;
      case caffe::Caffe::GPU:
        caffe::caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                             blob->mutable_gpu_diff());
        break;
    }
  }
}

int ParseIterFromSnapshot(const std::string& snapshot) {
  unsigned start = snapshot.find_last_of("_");
  unsigned end = snapshot.find_last_of(".");
  return std::stoi(snapshot.substr(start+1, end-start-1));
}

int ParseScoreFromSnapshot(const std::string& snapshot) {
  unsigned start = snapshot.find("_HiScore");
  unsigned end = snapshot.find("_iter_");
  return std::stoi(snapshot.substr(start+8, end-start-1));
}

void RemoveSnapshots(const std::string& regexp, int min_iter) {
  for (const std::string& f : FilesMatchingRegexp(regexp)) {
    int iter = ParseIterFromSnapshot(f);
    if (iter < min_iter) {
      LOG(INFO) << "Removing " << f;
      CHECK(boost::filesystem::is_regular_file(f));
      boost::filesystem::remove(f);
    }
  }
}

int FindGreatestIter(const std::string& regexp) {
  int max_iter = -1;
  std::vector<std::string> matching_files = FilesMatchingRegexp(regexp);
  for (const std::string& f : matching_files) {
    int iter = ParseIterFromSnapshot(f);
    if (iter > max_iter) {
      max_iter = iter;
    }
  }
  return max_iter;
}

void FindLatestSnapshot(const std::string& snapshot_prefix,
                        std::string& actor_snapshot,
                        std::string& critic_snapshot,
                        std::string& memory_snapshot) {
  std::string actor_regexp(snapshot_prefix + "_actor_iter_[0-9]+\\.solverstate");
  std::string critic_regexp(snapshot_prefix + "_critic_iter_[0-9]+\\.solverstate");
  std::string memory_regexp(snapshot_prefix + "_iter_[0-9]+\\.replaymemory");
  int actor_max_iter = FindGreatestIter(actor_regexp);
  int critic_max_iter = FindGreatestIter(critic_regexp);
  int memory_max_iter = FindGreatestIter(memory_regexp);
  if (actor_max_iter > 0) {
    actor_snapshot = snapshot_prefix + "_actor_iter_"
        + std::to_string(actor_max_iter) + ".solverstate";
  }
  if (critic_max_iter > 0) {
    critic_snapshot = snapshot_prefix + "_critic_iter_"
        + std::to_string(critic_max_iter) + ".solverstate";
  }
  if (memory_max_iter > 0) {
    memory_snapshot = snapshot_prefix + "_iter_"
        + std::to_string(memory_max_iter) + ".replaymemory";
  }
}

int FindHiScore(const std::string& snapshot_prefix) {
  using namespace boost::filesystem;
  std::string regexp(snapshot_prefix + "_HiScore[-]?[0-9]+_iter_[0-9]+\\.caffemodel");
  std::vector<std::string> matching_files = FilesMatchingRegexp(regexp);
  int max_score = std::numeric_limits<int>::lowest();
  for (const std::string& f : matching_files) {
    int score = ParseScoreFromSnapshot(f);
    if (score > max_score) {
      max_score = score;
    }
  }
  return max_score;
}

// Get the offset for a param of a given action. Returns -1 if a
// non-existant offset is requested.
int GetParamOffset(const action_t action, const int arg_num = 0) {
  if (arg_num < 0 || arg_num > 1) {
    return -1;
  }
  switch (action) {
    case DASH:
      return arg_num;
    case TURN:
      return arg_num == 0 ? 2 : -1;
    case TACKLE:
      return arg_num == 0 ? 3 : -1;
    case KICK:
      return 4 + arg_num;
    default:
      LOG(FATAL) << "Unrecognized action: " << action;
  }
}

DQN::DQN(caffe::SolverParameter& actor_solver_param,
         caffe::SolverParameter& critic_solver_param) :
        actor_solver_param_(actor_solver_param),
        critic_solver_param_(critic_solver_param),
        replay_memory_capacity_(FLAGS_memory),
        gamma_(FLAGS_gamma),
        clone_frequency_(FLAGS_clone_freq),
        random_engine(),
        smoothed_critic_loss_(0),
        smoothed_actor_loss_(0) {
  if (FLAGS_seed <= 0) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    LOG(INFO) << "Seeding RNG to time (seed = " << seed << ")";
    random_engine.seed(seed);
  } else {
    LOG(INFO) << "Seeding RNG with seed = " << FLAGS_seed;
    random_engine.seed(FLAGS_seed);
  }
  Initialize();
}

void DQN::Benchmark(int iterations) {
  LOG(INFO) << "*** Benchmark begins ***";
  caffe::Timer critic_timer;
  critic_timer.Start();
  for (int i=0; i<iterations; ++i) {
    UpdateCritic();
  }
  critic_timer.Stop();
  LOG(INFO) << "Average Critic Update: "
            << critic_timer.MilliSeconds()/iterations << " ms.";
  caffe::Timer actor_timer;
  actor_timer.Start();
  for (int i=0; i< iterations; ++i) {
    UpdateActor();
  }
  actor_timer.Stop();
  LOG(INFO) << "Average Actor Update: "
            << actor_timer.MilliSeconds()/iterations << " ms.";
  LOG(INFO) << "*** Benchmark ends ***";
}

// Randomly sample the replay memory n times, returning the indexes
std::vector<int> DQN::SampleTransitionsFromMemory(int n) {
  std::vector<int> transitions(n);
  for (int i = 0; i < n; ++i) {
    transitions[i] =
        std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
            random_engine);
  }
  return transitions;
}

std::vector<InputStates> DQN::SampleStatesFromMemory(int n) {
  std::vector<InputStates> states_batch(n);
  std::vector<int> transitions = SampleTransitionsFromMemory(n);
  for (int i = 0; i < n; ++i) {
    const auto& transition = replay_memory_[transitions[i]];
    InputStates last_states;
    for (int j = 0; j < kStateInputCount; ++j) {
      last_states[j] = std::get<0>(transition)[j];
    }
    states_batch[i] = last_states;
  }
  return states_batch;
}

void DQN::LoadActorWeights(const std::string& actor_weights) {
  CHECK(boost::filesystem::is_regular_file(actor_weights))
      << "Invalid file: " << actor_weights;
  LOG(INFO) << "Actor weights finetuning from " << actor_weights;
  actor_net_->CopyTrainedLayersFrom(actor_weights);
}

void DQN::LoadCriticWeights(const std::string& critic_weights) {
  CHECK(boost::filesystem::is_regular_file(critic_weights))
      << "Invalid file: " << critic_weights;
  LOG(INFO) << "Critic weights finetuning from " << critic_weights;
  critic_net_->CopyTrainedLayersFrom(critic_weights);
  CloneNet(critic_net_, critic_target_net_);
}

void DQN::RestoreActorSolver(const std::string& actor_solver) {
  CHECK(boost::filesystem::is_regular_file(actor_solver))
      << "Invalid file: " << actor_solver;
  LOG(INFO) << "Actor solver state resuming from " << actor_solver;
  actor_solver_->Restore(actor_solver.c_str());
}

void DQN::RestoreCriticSolver(const std::string& critic_solver) {
  CHECK(boost::filesystem::is_regular_file(critic_solver))
      << "Invalid file: " << critic_solver;
  LOG(INFO) << "Critic solver state resuming from " << critic_solver;
  critic_solver_->Restore(critic_solver.c_str());
  CloneNet(critic_net_, critic_target_net_);
}

std::vector<std::string> FilesMatchingRegexp(const std::string& regexp) {
  using namespace boost::filesystem;
  path search_stem(regexp);
  path search_dir(current_path());
  if (search_stem.has_parent_path()) {
    search_dir = search_stem.parent_path();
    search_stem = search_stem.filename();
  }
  const boost::regex expression(search_stem.native());
  std::vector<std::string> matching_files;
  directory_iterator end;
  for(directory_iterator it(search_dir); it != end; ++it) {
    if (is_regular_file(it->status())) {
      path p(it->path());
      boost::smatch what;
      if (boost::regex_match(p.filename().native(), what, expression)) {
        matching_files.push_back(p.native());
      }
    }
  }
  return matching_files;
}

void DQN::Snapshot(const std::string& snapshot_prefix, bool remove_old,
                   bool snapshot_memory) {
  using namespace boost::filesystem;
  actor_solver_->Snapshot(snapshot_prefix + "_actor");
  critic_solver_->Snapshot(snapshot_prefix + "_critic");
  int actor_iter = actor_solver_->iter() + 1;
  std::string fname = snapshot_prefix + "_actor_iter_" + std::to_string(actor_iter);
  CHECK(is_regular_file(fname + ".caffemodel"));
  CHECK(is_regular_file(fname + ".solverstate"));
  int critic_iter = critic_solver_->iter() + 1;
  fname = snapshot_prefix + "_critic_iter_" + std::to_string(critic_iter);
  CHECK(is_regular_file(fname + ".caffemodel"));
  CHECK(is_regular_file(fname + ".solverstate"));
  if (snapshot_memory) {
    fname = snapshot_prefix + "_iter_" + std::to_string(critic_iter);
    std::string mem_fname = fname + ".replaymemory";
    LOG(INFO) << "Snapshotting memory to " << mem_fname;
    SnapshotReplayMemory(mem_fname);
    CHECK(is_regular_file(mem_fname));
  }
  if (remove_old) {
    RemoveSnapshots(snapshot_prefix + "_actor_iter_[0-9]+"
                    "\\.(caffemodel|solverstate)", actor_iter);
    RemoveSnapshots(snapshot_prefix + "_critic_iter_[0-9]+"
                    "\\.(caffemodel|solverstate)", critic_iter);
    RemoveSnapshots(snapshot_prefix + "_iter_[0-9]+\\.replaymemory", critic_iter);
  }
}

void DQN::Initialize() {
#ifndef NDEBUG
  actor_solver_param_.set_debug_info(true);
  critic_solver_param_.set_debug_info(true);
#endif
  // Initialize net and solver
  actor_solver_.reset(caffe::GetSolver<float>(actor_solver_param_));
  critic_solver_.reset(caffe::GetSolver<float>(critic_solver_param_));
  actor_net_ = actor_solver_->net();
  critic_net_ = critic_solver_->net();
#ifndef NDEBUG
  actor_net_->set_debug_info(true);
  critic_net_->set_debug_info(true);
#endif
  // Check that nets have the necessary layers and blobs
  HasBlobSize(*actor_net_, states_blob_name,
              {kMinibatchSize, kStateInputCount, kStateSize, 1});
  HasBlobSize(*actor_net_, actions_blob_name,
              {kMinibatchSize, kActionSize});
  HasBlobSize(*actor_net_, action_params_blob_name,
              {kMinibatchSize, kActionParamSize});
  HasBlobSize(*critic_net_, states_blob_name,
              {kMinibatchSize, kStateInputCount, kStateSize, 1});
  HasBlobSize(*critic_net_, actions_blob_name,
              {kMinibatchSize, 1, kActionSize, 1});
  HasBlobSize(*critic_net_, action_params_blob_name,
              {kMinibatchSize, 1, kActionParamSize, 1});
  HasBlobSize(*critic_net_, targets_blob_name,
              {kMinibatchSize, 1, 1, 1});
  HasBlobSize(*critic_net_, q_values_blob_name,
              {kMinibatchSize, 1});
  // HasBlobSize(*critic_net_, loss_blob_name, {1});
  CHECK(actor_net_->has_layer(state_input_layer_name));
  CHECK(critic_net_->has_layer(state_input_layer_name));
  CHECK(critic_net_->has_layer(action_input_layer_name));
  CHECK(critic_net_->has_layer(action_params_input_layer_name));
  CHECK(critic_net_->has_layer(target_input_layer_name));
  CHECK(critic_net_->has_layer(q_values_layer_name));
  CloneNet(critic_net_, critic_target_net_);
}

Action DQN::GetRandomHFOAction() {
  action_t action_indx = (action_t) std::uniform_int_distribution<int>(DASH, KICK)(random_engine);
  float arg1, arg2;
  switch (action_indx) {
    case DASH:
      arg1 = std::uniform_real_distribution<float>(-100.0, 100.0)(random_engine);
      arg2 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      break;
    case TURN:
      arg1 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      arg2 = 0;
      break;
    case TACKLE:
      arg1 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      arg2 = 0;
      break;
    case KICK:
      arg1 = std::uniform_real_distribution<float>(0.0, 100.0)(random_engine);
      arg2 = std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
      break;
    default:
      LOG(FATAL) << "Invalid Action Index: " << action_indx;
      break;
  }
  Action act = {action_indx, arg1, arg2};
  return act;
}

Action DQN::SelectAction(const InputStates& last_states, const double epsilon) {
  return SelectActions(std::vector<InputStates>{{last_states}}, epsilon)[0];
}

float DQN::EvaluateAction(const InputStates& input_states, const Action& action) {
  return CriticForward(*critic_net_,
                       std::vector<InputStates>{{input_states}},
                       std::vector<Action>{{action}})[0];
}

std::vector<Action> DQN::SelectActions(const std::vector<InputStates>& states_batch,
                                       const double epsilon) {
  CHECK(epsilon >= 0.0 && epsilon <= 1.0);
  CHECK_LE(states_batch.size(), kMinibatchSize);
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < epsilon) {
    // Select randomly
    std::vector<Action> actions(states_batch.size());
    for (int i=0; i<actions.size(); ++i) {
      actions[i] = GetRandomHFOAction();
    }
    return actions;
  } else {
    // Select greedily
    return SelectActionGreedily(*actor_net_, states_batch);
  }
}

Action DQN::SelectActionGreedily(caffe::Net<float>& actor,
                                const InputStates& last_states) {
  return SelectActionGreedily(
      actor, std::vector<InputStates>{{last_states}}).front();
}

std::vector<Action>
DQN::SelectActionGreedily(caffe::Net<float>& actor,
                          const std::vector<InputStates>& states_batch) {
  DLOG(INFO) << "  [Forward] Actor";
  CHECK(actor.has_blob(actions_blob_name));
  CHECK(actor.has_blob(action_params_blob_name));
  CHECK_LE(states_batch.size(), kMinibatchSize);
  std::vector<float> states_input(kStateInputDataSize, 0.0f);
  const auto states_blob = actor.blob_by_name(states_blob_name);
  for (int n = 0; n < states_batch.size(); ++n) {
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = states_batch[n][c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + states_blob->offset(n,c,0,0));
    }
  }
  InputDataIntoLayers(actor, states_input.data(), NULL, NULL, NULL, NULL);
  actor.ForwardPrefilled(nullptr);
  std::vector<Action> actions(states_batch.size());
  const auto actions_blob = actor.blob_by_name(actions_blob_name);
  const auto action_params_blob = actor.blob_by_name(action_params_blob_name);
  for (int n = 0; n < states_batch.size(); ++n) {
    std::vector<float> action_values(kActionSize);
    for (int c = 0; c < kActionSize; ++c) {
      action_values[c] = actions_blob->data_at(n,c,0,0);
    }
    action_t max_act = (action_t) std::distance(action_values.begin(), std::max_element(
        action_values.begin(), action_values.end()));
    actions[n].action = max_act;
    int arg1_offset = GetParamOffset(max_act, 0); CHECK_GE(arg1_offset, 0);
    actions[n].arg1 = action_params_blob->data_at(n,arg1_offset,0,0);
    int arg2_offset = GetParamOffset(max_act, 1);
    actions[n].arg2 = arg2_offset < 0 ? 0 :
        action_params_blob->data_at(n,arg2_offset,0,0);
  }
  return actions;
}

void DQN::AddTransition(const Transition& transition) {
  if (replay_memory_.size() == replay_memory_capacity_) {
    replay_memory_.pop_front();
  }
  replay_memory_.push_back(transition);
}

void DQN::Update() {
  if (memory_size() < FLAGS_memory_threshold) {
    return;
  }
  if (FLAGS_update_critic) {
    float loss = UpdateCritic();
    if (critic_iter() % FLAGS_loss_display_iter == 0) {
      LOG(INFO) << "Critic Iteration " << critic_iter()
                << ", loss = " << smoothed_critic_loss_
                << ", Optimism = " << AssessOptimism();
      smoothed_critic_loss_ = 0;
    }
    smoothed_critic_loss_ += loss / float(FLAGS_loss_display_iter);
  }
  if (FLAGS_update_actor) {
    float loss = UpdateActor();
    if (actor_iter() % FLAGS_loss_display_iter == 0) {
      LOG(INFO) << "Actor Iteration " << actor_iter()
                << ", loss = " << smoothed_actor_loss_;
      smoothed_actor_loss_ = 0;
    }
    smoothed_actor_loss_ += loss / float(FLAGS_loss_display_iter);
  }
}

float DQN::AssessOptimism(int n_states, int n_samples_per_state) {
  std::vector<float> results;
  for (int i=0; i<n_states; ++i) {
    std::vector<InputStates> states = SampleStatesFromMemory(kMinibatchSize);
    std::vector<float> opt = AssessOptimism(states, n_samples_per_state);
    results.insert(results.end(), opt.begin(), opt.end());
  }
  return std::accumulate(results.begin(), results.end(), 0.0) / float(results.size());
}

float DQN::AssessOptimism(const InputStates& input_states, int n_samples) {
  return AssessOptimism(
      std::vector<InputStates>{{input_states}}, n_samples).front();
}

std::vector<float>
DQN::AssessOptimism(const std::vector<InputStates>& states_batch, int n_samples) {
  std::vector<float> opt(states_batch.size(), 0.0);
  // Q-Values for the actions that actor_net_ recommends
  std::vector<float> actor_q_vals = CriticForwardThroughActor(
      *critic_net_, states_batch);
  std::vector<hfo::Action> rand_action_batch(states_batch.size());
  for (int n=0; n<n_samples; ++n) {
    for (int i=0; i<states_batch.size(); ++i) {
      rand_action_batch[i] = GetRandomHFOAction();
    }
    // The Q-Values for random actions
    std::vector<float> rand_q_vals = CriticForward(
        *critic_net_, states_batch, rand_action_batch);
    for (int i=0; i<states_batch.size(); ++i) {
      if (rand_q_vals[i] > actor_q_vals[i]) {
        opt[i] += 1.0 / float(n_samples);
      }
    }
  }
  return opt;
}


float DQN::UpdateCritic() {
  DLOG(INFO) << "[Update] Critic";
  CHECK(critic_net_->has_blob(states_blob_name));
  CHECK(critic_net_->has_blob(actions_blob_name));
  CHECK(critic_net_->has_blob(action_params_blob_name));
  CHECK(critic_net_->has_blob(targets_blob_name));
  CHECK(critic_net_->has_blob(loss_blob_name));
  const auto states_blob = critic_net_->blob_by_name(states_blob_name);
  const auto action_blob = critic_net_->blob_by_name(actions_blob_name);
  const auto action_params_blob = critic_net_->blob_by_name(action_params_blob_name);
  const auto target_blob = critic_net_->blob_by_name(targets_blob_name);
  const auto loss_blob = critic_net_->blob_by_name(loss_blob_name);
  // Every clone_iters steps, update the clone_net_ to equal the primary net
  if (critic_iter() % clone_frequency_ == 0) {
    LOG(INFO) << "Critic Iter " << critic_iter() << ": Updating Clone Net";
    CloneNet(critic_net_, critic_target_net_);
  }
  // Collect a batch of next-states used to generate target_q_values
  std::vector<int> transitions = SampleTransitionsFromMemory(kMinibatchSize);
  std::vector<InputStates> target_states_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    if (!std::get<3>(transition)) {
      continue; // terminal state
    }
    InputStates target_states;
    for (int i = 0; i < kStateInputCount - 1; ++i) {
      target_states[i] = std::get<0>(transition)[i + 1];
    }
    target_states[kStateInputCount - 1] = std::get<3>(transition).get();
    target_states_batch.push_back(target_states);
  }
  // Generate target_q_values using the critic_target_net_
  const std::vector<float> target_q_values =
      CriticForwardThroughActor(*critic_target_net_, target_states_batch);
  std::vector<float> states_input(kStateInputDataSize, 0.0f);
  std::vector<float> action_input(kActionInputDataSize, 0.0f);
  std::vector<float> action_params_input(kActionParamsInputDataSize, 0.0f);
  std::vector<float> target_input(kTargetInputDataSize, 0.0f);
  auto target_value_idx = 0;
  for (int n = 0; n < kMinibatchSize; ++n) {
    const auto& transition = replay_memory_[transitions[n]];
    const Action action = std::get<1>(transition);
    const float reward = std::get<2>(transition);
    CHECK(reward >= -1.0 && reward <= 1.0);
    const auto target = std::get<3>(transition) ?
        reward + gamma_ * target_q_values[target_value_idx++] : reward;
    CHECK(std::isfinite(target)) << "Target not finite!";
    target_input[target_blob->offset(n,0,0,0)] = target;
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = std::get<0>(transition)[c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + states_blob->offset(n,c,0,0));
    }
    action_input[action_blob->offset(n,0,action.action,0)] = 1.0;
    int arg1_offset = GetParamOffset(action.action, 0); CHECK_GE(arg1_offset, 0);
    action_params_input[action_params_blob->offset(n,0,arg1_offset,0)] = action.arg1;
    int arg2_offset = GetParamOffset(action.action, 1);
    if (arg2_offset > 0) {
      action_params_input[action_params_blob->offset(n,0,arg2_offset,0)] = action.arg2;
    }
  }
  InputDataIntoLayers(*critic_net_, states_input.data(), action_input.data(),
                      action_params_input.data(), target_input.data(), NULL);
  DLOG(INFO) << " [Step] Critic";
  critic_solver_->Step(1);
  // Return the loss
  float loss = loss_blob->data_at(0,0,0,0);
  CHECK(std::isfinite(loss)) << "Critic loss not finite!";
  return loss;
}

float DQN::UpdateActor() {
  return UpdateActor(*critic_net_);
}

float DQN::UpdateActor(caffe::Net<float>& critic) {
  DLOG(INFO) << "[Update] Actor";
  CHECK(critic.has_blob(q_values_blob_name));
  CHECK(critic.has_blob(states_blob_name));
  CHECK(critic.has_blob(actions_blob_name));
  CHECK(critic.has_blob(action_params_blob_name));
  CHECK(actor_net_->has_blob(actions_blob_name));
  CHECK(actor_net_->has_blob(action_params_blob_name));
  const auto q_values_blob = critic.blob_by_name(q_values_blob_name);
  const auto critic_actions_blob = critic.blob_by_name(actions_blob_name);
  const auto critic_action_params_blob = critic.blob_by_name(action_params_blob_name);
  const auto actor_actions_blob = actor_net_->blob_by_name(actions_blob_name);
  const auto actor_action_params_blob = actor_net_->blob_by_name(action_params_blob_name);

  // Prevent accumulation of actor gradients
  ZeroGradParameters(*actor_net_);

  std::vector<InputStates> states_batch = SampleStatesFromMemory(kMinibatchSize);
  std::vector<float> q_values = CriticForwardThroughActor(critic, states_batch);

  // Set the critic diff and run backward
  float* q_values_diff = q_values_blob->mutable_cpu_diff();
  for (int n = 0; n < kMinibatchSize; n++) {
    q_values_diff[q_values_blob->offset(n,0,0,0)] = -1.0;
  }
  DLOG(INFO) << " [Backwards] " << critic.name();
  critic.BackwardFrom(GetLayerIndex(critic, q_values_layer_name));

  float action_diff = critic_actions_blob->asum_diff()
      / critic_actions_blob->count();
  // LOG(INFO) << "Critic Action Blob Diff: " << diff_abs_val_mean;
  float action_param_diff = critic_action_params_blob->asum_diff()
      / critic_action_params_blob->count();
  // LOG(INFO) << "Critic Action Params Blob Diff: " << diff_abs_val_mean;

  // Transfer input-level diffs from Critic to Actor
  actor_actions_blob->ShareDiff(*critic_actions_blob);
  actor_action_params_blob->ShareDiff(*critic_action_params_blob);

  DLOG(INFO) << " [Backwards] " << actor_net_->name();
  actor_net_->Backward();
  actor_solver_->ComputeUpdateValue();
  actor_solver_->set_iter(actor_solver_->iter() + 1);
  actor_net_->Update();

  std::vector<float> new_q_values = CriticForwardThroughActor(critic, states_batch);
  float avg_q = 0, post_avg_q = 0, avg_q_diff = 0;
  float sz = float(q_values.size());
  for (int i=0; i<q_values.size(); ++i) {
    avg_q += q_values[i] / sz;
    post_avg_q += new_q_values[i] / sz;
    avg_q_diff = (new_q_values[i] - q_values[i]) / sz;
  }
  VLOG(1) << "Iter " << actor_iter()
          << ", PreUpdateAvgQ = " << avg_q
          << ", PostUpdateAvgQ = " << post_avg_q
          << ", AvgQDiff = " << avg_q_diff;

  return action_diff;
}

std::vector<float> DQN::CriticForwardThroughActor(
    caffe::Net<float>& critic, const std::vector<InputStates>& states_batch) {
  DLOG(INFO) << " [Forward] " << critic.name() << " Through " << actor_net_->name();
  // Use this to run the actor forward. Don't care about returned actions
  SelectActionGreedily(*actor_net_, states_batch);
  const auto actor_actions_blob = actor_net_->blob_by_name(actions_blob_name);
  const auto actor_action_params_blob = actor_net_->blob_by_name(action_params_blob_name);
  DLOG(INFO) << "  [Forward] " << critic.name();
  CHECK(critic.has_blob(states_blob_name));
  CHECK(critic.has_blob(actions_blob_name));
  CHECK(critic.has_blob(action_params_blob_name));
  CHECK(critic.has_blob(q_values_blob_name));
  CHECK_LE(states_batch.size(), kMinibatchSize);
  const auto states_blob = critic.blob_by_name(states_blob_name);
  const auto actions_blob = critic.blob_by_name(actions_blob_name);
  const auto action_params_blob = critic.blob_by_name(action_params_blob_name);
  std::vector<float> states_input(kStateInputDataSize, 0.0f);
  std::vector<float> target_input(kTargetInputDataSize, 0.0f);
  for (int n = 0; n < states_batch.size(); ++n) {
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = states_batch[n][c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + states_blob->offset(n,c,0,0));
    }
  }
  // Transfer actions + params from actor to critic
  CHECK(critic.has_layer(action_input_layer_name));
  const auto action_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          critic.layer_by_name(action_input_layer_name));
  action_input_layer->Reset(actor_actions_blob->mutable_cpu_data(),
                            actor_actions_blob->mutable_cpu_data(),
                            action_input_layer->batch_size());
  CHECK(critic.has_layer(action_params_input_layer_name));
  const auto action_params_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          critic.layer_by_name(action_params_input_layer_name));
  action_params_input_layer->Reset(actor_action_params_blob->mutable_cpu_data(),
                                   actor_action_params_blob->mutable_cpu_data(),
                                   action_params_input_layer->batch_size());
  InputDataIntoLayers(critic, states_input.data(), NULL, NULL, target_input.data(), NULL);
  critic.ForwardPrefilled(nullptr);
  const auto q_values_blob = critic.blob_by_name(q_values_blob_name);
  std::vector<float> q_values(states_batch.size());
  for (int n = 0; n < states_batch.size(); ++n) {
    q_values[n] = q_values_blob->data_at(n,0,0,0);
  }
  return q_values;
}

std::vector<float> DQN::CriticForward(caffe::Net<float>& critic,
                                      const std::vector<InputStates>& states_batch,
                                      const std::vector<Action>& action_batch) {
  DLOG(INFO) << "  [Forward] " << critic.name();
  CHECK(critic.has_blob(states_blob_name));
  CHECK(critic.has_blob(actions_blob_name));
  CHECK(critic.has_blob(action_params_blob_name));
  CHECK(critic.has_blob(q_values_blob_name));
  CHECK_LE(states_batch.size(), kMinibatchSize);
  CHECK_EQ(states_batch.size(), action_batch.size());
  const auto states_blob = critic.blob_by_name(states_blob_name);
  const auto actions_blob = critic.blob_by_name(actions_blob_name);
  const auto action_params_blob = critic.blob_by_name(action_params_blob_name);
  std::vector<float> states_input(kStateInputDataSize, 0.0f);
  std::vector<float> action_input(kActionInputDataSize, 0.0f);
  std::vector<float> action_params_input(kActionParamsInputDataSize, 0.0f);
  std::vector<float> target_input(kTargetInputDataSize, 0.0f);
  for (int n = 0; n < states_batch.size(); ++n) {
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = states_batch[n][c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + states_blob->offset(n,c,0,0));
    }
    Action act = action_batch[n];
    action_input[actions_blob->offset(n,0,act.action,0)] = 1.0;
    int arg1_offset = GetParamOffset(act.action, 0); CHECK_GE(arg1_offset, 0);
    action_params_input[action_params_blob->offset(n,0,arg1_offset,0)] = act.arg1;
    int arg2_offset = GetParamOffset(act.action, 1);
    if (arg2_offset > 0) {
      action_params_input[action_params_blob->offset(n,0,arg2_offset,0)] = act.arg2;
    }
  }
  InputDataIntoLayers(critic, states_input.data(), action_input.data(),
                      action_params_input.data(), target_input.data(), NULL);
  critic.ForwardPrefilled(nullptr);
  const auto q_values_blob = critic.blob_by_name(q_values_blob_name);
  std::vector<float> q_values(states_batch.size());
  for (int n = 0; n < states_batch.size(); ++n) {
    q_values[n] = q_values_blob->data_at(n,0,0,0);
  }
  return q_values;
}

void DQN::CloneNet(NetSp& net_from, NetSp& net_to) {
  caffe::NetParameter net_param;
  net_from->ToProto(&net_param);
  net_param.set_name(net_param.name() + "Clone");
  net_param.set_force_backward(true);
#ifndef NDEBUG
  net_param.set_debug_info(true);
#endif
  if (!net_to) {
    net_to.reset(new caffe::Net<float>(net_param));
  } else {
    net_to->CopyTrainedLayersFrom(net_param);
  }
}

void DQN::InputDataIntoLayers(caffe::Net<float>& net,
                              float* states_input,
                              float* actions_input,
                              float* action_params_input,
                              float* target_input,
                              float* filter_input) {
  if (states_input != NULL) {
    const auto state_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name(state_input_layer_name));
    CHECK(state_input_layer);
    state_input_layer->Reset(states_input, states_input,
                             state_input_layer->batch_size());
  }
  if (actions_input != NULL) {
    const auto action_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name(action_input_layer_name));
    CHECK(action_input_layer);
    action_input_layer->Reset(actions_input, actions_input,
                              action_input_layer->batch_size());
  }
  if (action_params_input != NULL) {
    const auto action_params_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name(action_params_input_layer_name));
    CHECK(action_params_input_layer);
    action_params_input_layer->Reset(action_params_input, action_params_input,
                                     action_params_input_layer->batch_size());
  }
  if (target_input != NULL) {
    const auto target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name(target_input_layer_name));
    CHECK(target_input_layer);
    target_input_layer->Reset(target_input, target_input,
                              target_input_layer->batch_size());
  }
  if (filter_input != NULL) {
    const auto filter_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name(filter_input_layer_name));
    CHECK(filter_input_layer);
    filter_input_layer->Reset(filter_input, filter_input,
                              filter_input_layer->batch_size());
  }
}

void DQN::SnapshotReplayMemory(const std::string& filename) {
  std::ofstream ofile(filename.c_str(),
                      std::ios_base::out | std::ofstream::binary);
  boost::iostreams::filtering_ostream out;
  out.push(boost::iostreams::gzip_compressor());
  out.push(ofile);
  int num_transitions = memory_size();
  out.write((char*)&num_transitions, sizeof(int));
  int episodes = 0;
  bool terminal = true;
  for (const Transition& t : replay_memory_) {
    const InputStates& states = std::get<0>(t);
    if (terminal) { // Save the history of states
      for (int i = 0; i < kStateInputCount - 1; ++i) {
        const StateDataSp state = states[i];
        out.write((char*)state->begin(), kStateSize * sizeof(float));
      }
    }
    const StateDataSp curr_state = states.back();
    out.write((char*)curr_state->begin(), kStateSize * sizeof(float));
    const Action& action = std::get<1>(t);
    out.write((char*)&action, sizeof(Action));
    const float& reward = std::get<2>(t);
    out.write((char*)&reward, sizeof(float));
    terminal = !std::get<3>(t);
    out.write((char*)&terminal, sizeof(bool));
    if (terminal) { episodes++; }
  }
  LOG(INFO) << "Saved memory of size " << memory_size() << " with "
            << episodes << " episodes";
}

void DQN::LoadReplayMemory(const std::string& filename) {
  CHECK(boost::filesystem::is_regular_file(filename)) << "Invalid file: " << filename;
  LOG(INFO) << "Loading replay memory from " << filename;
  ClearReplayMemory();
  std::ifstream ifile(filename.c_str(),
                      std::ios_base::in | std::ofstream::binary);
  boost::iostreams::filtering_istream in;
  in.push(boost::iostreams::gzip_decompressor());
  in.push(ifile);
  int num_transitions;
  in.read((char*)&num_transitions, sizeof(int));
  replay_memory_.resize(num_transitions);
  std::deque<dqn::StateDataSp> past_states;
  int episodes = 0;
  bool terminal = true;
  for (int i = 0; i < num_transitions; ++i) {
    Transition& t = replay_memory_[i];
    if (terminal) {
      past_states.clear();
      for (int i = 0; i < kStateInputCount - 1; ++i) {
        StateDataSp state = std::make_shared<StateData>();
        in.read((char*)state->begin(), kStateSize * sizeof(float));
        past_states.push_back(state);
      }
    }
    StateDataSp state = std::make_shared<StateData>();
    in.read((char*)state->begin(), kStateSize * sizeof(float));
    past_states.push_back(state);
    while (past_states.size() > kStateInputCount) {
      past_states.pop_front();
    }
    CHECK_EQ(past_states.size(), kStateInputCount);
    InputStates& states = std::get<0>(t);
    std::copy(past_states.begin(), past_states.end(), states.begin());
    in.read((char*)&std::get<1>(t), sizeof(Action));
    in.read((char*)&std::get<2>(t), sizeof(float));
    std::get<3>(t) = boost::none;
    if (i > 0 && !terminal) { // Set the next state for the last transition
      std::get<3>(replay_memory_[i-1]) = state;
    }
    in.read((char*)&terminal, sizeof(bool));
    if (terminal) { episodes++; };
  }
  LOG(INFO) << "replay_mem_size = " << memory_size() << " with "
            << episodes << " episodes";
}

}
