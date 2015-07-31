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

namespace dqn {

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
                   expected_shape.begin()));
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

// Randomly sample the replay memory n times, returning the indexes
std::vector<int> DQN::SampleReplayMemory(int n) {
  std::vector<int> transitions(n);
  for (int i = 0; i < n; ++i) {
    transitions[i] =
        std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
            random_engine);
  }
  return transitions;
}

void DQN::LoadTrainedModel(const std::string& actor_model_bin,
                           const std::string& critic_model_bin) {
  actor_net_->CopyTrainedLayersFrom(actor_model_bin);
  critic_net_->CopyTrainedLayersFrom(critic_model_bin);
  CloneNet(critic_net_, critic_target_net_);
}

void DQN::RestoreSolver(const std::string& actor_solver_bin,
                        const std::string& critic_solver_bin) {
  actor_solver_->Restore(actor_solver_bin.c_str());
  critic_solver_->Restore(critic_solver_bin.c_str());
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
  // Initialize net and solver
  actor_solver_.reset(caffe::GetSolver<float>(actor_solver_param_));
  critic_solver_.reset(caffe::GetSolver<float>(critic_solver_param_));
  actor_net_ = actor_solver_->net();
  critic_net_ = critic_solver_->net();
  // Check that nets have the necessary layers in the required sizes
  HasBlobSize(*actor_net_, states_blob_name,
              {kMinibatchSize, kStateInputCount, kStateSize, 1});
  HasBlobSize(*actor_net_, actions_blob_name,
              {kMinibatchSize, kActionSize});
  HasBlobSize(*critic_net_, states_blob_name,
              {kMinibatchSize, kStateInputCount, kStateSize, 1});
  HasBlobSize(*critic_net_, actions_blob_name,
              {kMinibatchSize, 1, kActionSize, 1});
  HasBlobSize(*critic_net_, targets_blob_name,
              {kMinibatchSize, 1, kActionSize, 1});
  HasBlobSize(*critic_net_, q_values_blob_name,
              {kMinibatchSize, kActionSize});
  CloneNet(critic_net_, critic_target_net_);
}

float DQN::SelectAction(const InputStates& last_states, const double epsilon) {
  return SelectActions(std::vector<InputStates>{{last_states}}, epsilon)[0];
}

std::vector<float> DQN::SelectActions(const std::vector<InputStates>& states_batch,
                                      const double epsilon) {
  CHECK(epsilon >= 0.0 && epsilon <= 1.0);
  CHECK_LE(states_batch.size(), kMinibatchSize);
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < epsilon) {
    // Select randomly
    std::vector<float> actions(states_batch.size());
    for (int i=0; i<actions.size(); ++i) {
      // TODO: Generalize for multiple actions
      float kickangle = std::uniform_real_distribution<float>
          (-90.0, 90.0)(random_engine);
      actions[i] = kickangle;
    }
    return actions;
  } else {
    // Select greedily
    return SelectActionGreedily(*actor_net_, states_batch);
  }
}

float DQN::SelectActionGreedily(caffe::Net<float>& actor,
                                const InputStates& last_states) {
  return SelectActionGreedily(
      actor, std::vector<InputStates>{{last_states}}).front();
}

std::vector<float>
DQN::SelectActionGreedily(caffe::Net<float>& actor,
                          const std::vector<InputStates>& states_batch) {
  DLOG(INFO) << "  [Forward] Actor";
  CHECK(actor.has_blob(actions_blob_name));
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
  InputDataIntoLayers(actor, states_input.data(), NULL, NULL, NULL);
  actor.ForwardPrefilled(nullptr);
  std::vector<float> actions(states_batch.size());
  const auto actions_blob = actor.blob_by_name(actions_blob_name);
  for (int n = 0; n < states_batch.size(); ++n) {
    actions[n] = actions_blob->data_at(n,0,0,0);
  }
  return actions;
}

void DQN::AddTransition(const Transition& transition) {
  if (replay_memory_.size() == replay_memory_capacity_) {
    replay_memory_.pop_front();
  }
  replay_memory_.push_back(transition);
}

void DQN::UpdateCritic() {
  DLOG(INFO) << "[Update] Critic";
  CHECK(critic_net_->has_blob(states_blob_name));
  CHECK(critic_net_->has_blob(actions_blob_name));
  CHECK(critic_net_->has_blob(targets_blob_name));
  const auto states_blob = critic_net_->blob_by_name(states_blob_name);
  const auto action_blob = critic_net_->blob_by_name(actions_blob_name);
  const auto target_blob = critic_net_->blob_by_name(targets_blob_name);
  // Every clone_iters steps, update the clone_net_ to equal the primary net
  if (current_iteration() % clone_frequency_ == 0) {
    LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
    CloneNet(critic_net_, critic_target_net_);
  }
  // Collect a batch of next-states used to generate target_q_values
  std::vector<int> transitions = SampleReplayMemory(kMinibatchSize);
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
  std::vector<float> target_input(kTargetInputDataSize, 0.0f);
  auto target_value_idx = 0;
  for (int n = 0; n < kMinibatchSize; ++n) {
    const auto& transition = replay_memory_[transitions[n]];
    const float action = std::get<1>(transition);
    const float reward = std::get<2>(transition);
    CHECK(reward >= -1.0 && reward <= 1.0);
    const auto target = std::get<3>(transition) ?
        reward + gamma_ * target_q_values[target_value_idx++] : reward;
    CHECK(!std::isnan(target));
    target_input[target_blob->offset(n,0,0,0)] = target;
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = std::get<0>(transition)[c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + states_blob->offset(n,c,0,0));
    }
    for(int c = 0; c < kActionSize; c++) {
      action_input[action_blob->offset(n,c,0,0)] = action;
    }
  }
  InputDataIntoLayers(*critic_net_, states_input.data(), action_input.data(),
                      target_input.data(), NULL);
  DLOG(INFO) << " [Step] Critic";
  critic_solver_->Step(1);
}

void DQN::UpdateActor() {
  UpdateActor(*critic_net_);
}

void DQN::UpdateActor(caffe::Net<float>& critic) {
  DLOG(INFO) << "[Update] Actor";
  CHECK(critic.has_blob(q_values_blob_name));
  CHECK(critic.has_blob(states_blob_name));
  CHECK(critic.has_blob(actions_blob_name));
  CHECK(actor_net_->has_blob(actions_blob_name));
  CHECK(actor_net_->has_layer(q_values_layer_name));
  // Prevent accumulation of actor gradients
  ZeroGradParameters(*actor_net_);
  std::vector<int> transitions = SampleReplayMemory(kMinibatchSize);
  std::vector<InputStates> states_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    InputStates last_states;
    for (int i = 0; i < kStateInputCount; ++i) {
      last_states[i] = std::get<0>(transition)[i];
    }
    states_batch.push_back(last_states);
  }
  CriticForwardThroughActor(critic, states_batch);
  // Set the critic diff
  const auto q_values_blob = critic.blob_by_name(q_values_blob_name);
  float* q_values_diff = q_values_blob->mutable_cpu_diff();
  for (int n = 0; n < kMinibatchSize; n++) {
    q_values_diff[q_values_blob->offset(n,0,0,0)] = -1.0; // TODO: not -1?
  }
  // Run the critic backward
  DLOG(INFO) << " [Backwards] " << critic.name();
  int ip2_indx = GetLayerIndex(critic, q_values_layer_name);
  critic.BackwardFrom(ip2_indx);
  // Transfer actions-diff from Critic to Actor
  const auto& critic_actions_blob = critic.blob_by_name(actions_blob_name);
  const auto& actor_actions_blob = actor_net_->blob_by_name(actions_blob_name);
  actor_actions_blob->ShareDiff(*critic_actions_blob); // TODO: Correct?
  // Run actor backward and update
  DLOG(INFO) << " [Backwards] " << actor_net_->name();
  actor_net_->Backward();
  actor_solver_->ComputeUpdateValue();
  actor_solver_->set_iter(actor_solver_->iter() + 1);
  actor_net_->Update();
}

std::vector<float> DQN::CriticForwardThroughActor(caffe::Net<float>& critic,
                                                  std::vector<InputStates>& states_batch) {
  DLOG(INFO) << " [Forward] " << critic.name() << " Through " << actor_net_->name();
  const std::vector<float> action_batch =
      SelectActionGreedily(*actor_net_, states_batch);
  return CriticForward(critic, states_batch, action_batch);
}

std::vector<float> DQN::CriticForward(caffe::Net<float>& critic,
                                      std::vector<InputStates>& states_batch,
                                      const std::vector<float>& action_batch) {
  DLOG(INFO) << "  [Forward] " << critic.name();
  // TODO: action_batch needs to be updated for the case of multiple actions
  CHECK(critic.has_blob(states_blob_name));
  CHECK(critic.has_blob(actions_blob_name));
  CHECK(critic.has_blob(q_values_blob_name));
  CHECK_LE(states_batch.size(), kMinibatchSize);
  CHECK_EQ(states_batch.size(), action_batch.size());
  const auto states_blob = critic.blob_by_name(states_blob_name);
  const auto actions_blob = critic.blob_by_name(actions_blob_name);
  std::vector<float> states_input(kStateInputDataSize, 0.0f);
  std::vector<float> action_input(kActionInputDataSize, 0.0f);
  std::vector<float> target_input(kTargetInputDataSize, 0.0f);
  for (int n = 0; n < states_batch.size(); ++n) {
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = states_batch[n][c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + states_blob->offset(n,c,0,0));
    }
    for (int c = 0; c < kActionSize; ++c) {
      action_input[actions_blob->offset(n,c,0,0)] = action_batch[n];
    }
  }
  InputDataIntoLayers(critic, states_input.data(), action_input.data(),
                      target_input.data(), NULL);
  critic.ForwardPrefilled(nullptr);
  const auto q_values_blob = critic.blob_by_name(q_values_blob_name);
  std::vector<float> q_values(states_batch.size());
  for (int n = 0; n < states_batch.size(); ++n) {
    for (int c = 0; c < kActionSize; ++c) {
      q_values[n] = q_values_blob->data_at(n,c,0,0);
    }
  }
  return q_values;
}

void DQN::CloneNet(NetSp& net_from, NetSp& net_to) {
  caffe::NetParameter net_param;
  net_from->ToProto(&net_param);
  net_param.set_name(net_param.name() + "Clone");
  net_param.set_force_backward(true);
  // net_param.set_debug_info(true);
  if (!net_to) {
    net_to.reset(new caffe::Net<float>(net_param));
  } else {
    net_to->CopyTrainedLayersFrom(net_param);
  }
}

void DQN::InputDataIntoLayers(caffe::Net<float>& net,
                              float* states_input,
                              float* actions_input,
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
  // For the first transition, save the history of states
  const InputStates& first_transition = std::get<0>(replay_memory_[0]);
  for (int i = 0; i < kStateInputCount - 1; ++i) {
    const StateDataSp state = first_transition[i];
    out.write((char*)state->begin(), kStateSize * sizeof(float));
  }
  // For all other transitions, save only the current state
  for (const Transition& t : replay_memory_) {
    const InputStates& states = std::get<0>(t);
    const StateDataSp curr_state = states.back();
    out.write((char*)curr_state->begin(), kStateSize * sizeof(float));
    const float& action = std::get<1>(t);
    out.write((char*)&action, sizeof(float));
    const float& reward = std::get<2>(t);
    out.write((char*)&reward, sizeof(float));
  }
  LOG(INFO) << "Saved memory of size " << memory_size();
}

// TODO: Need to fix the bug in this method?
void DQN::LoadReplayMemory(const std::string& filename) {
  LOG(INFO) << "Loading memory from " << filename;
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
  // First read the state history
  for (int i = 0; i < kStateInputCount - 1; ++i) {
    StateDataSp state = std::make_shared<StateData>();
    in.read((char*)state->begin(), kStateSize * sizeof(float));
    past_states.push_back(state);
  }
  for (int i = 0; i < num_transitions; ++i) {
    Transition& t = replay_memory_[i];
    StateDataSp state = std::make_shared<StateData>();
    in.read((char*)state->begin(), kStateSize * sizeof(float));
    past_states.push_back(state);
    while (past_states.size() > kStateInputCount) {
      past_states.pop_front();
    }
    CHECK_EQ(past_states.size(), kStateInputCount);
    InputStates& states = std::get<0>(t);
    std::copy(past_states.begin(), past_states.end(), states.begin());
    in.read((char*)&std::get<1>(t), sizeof(float));
    in.read((char*)&std::get<2>(t), sizeof(float));
    std::get<3>(t) == boost::none;
    // Set the next state for the last transition
    if (i > 0) {
      std::get<3>(replay_memory_[i-1]) = state;
    }
  }
  LOG(INFO) << "replay_mem_size = " << memory_size();
}

}
