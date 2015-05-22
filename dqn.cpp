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
  CHECK_EQ(blob_shape.size(), expected_shape.size());
  CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
                   expected_shape.begin()));
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

void RemoveSnapshots(const std::string& snapshot_prefix, int min_iter) {
  std::string regexp(snapshot_prefix + "(_actor|_critic)?_iter_[0-9]+\\."
                     "(caffemodel|solverstate|replaymemory)");
  for (const std::string& f : FilesMatchingRegexp(regexp)) {
    int iter = ParseIterFromSnapshot(f);
    if (iter < min_iter) {
      LOG(INFO) << "Removing " << f;
      CHECK(boost::filesystem::is_regular_file(f));
      boost::filesystem::remove(f);
    }
  }
}

std::tuple<std::string, std::string, std::string> FindLatestSnapshot(
    const std::string& snapshot_prefix) {
  using namespace boost::filesystem;
  std::string regexp(snapshot_prefix + "_(actor|critic)_iter_[0-9]+\\.solverstate");
  std::vector<std::string> matching_files = FilesMatchingRegexp(regexp);
  int max_iter = -1;
  std::tuple<std::string, std::string, std::string> latest;
  for (const std::string& f : matching_files) {
    int iter = ParseIterFromSnapshot(f);
    if (iter > max_iter) {
      // Look for an associated caffemodel + replaymemory
      std::string it = std::to_string(iter);
      std::string actor_solver = snapshot_prefix + "_actor_iter_" + it + ".solverstate";
      std::string actor_model = snapshot_prefix + "_actor_iter_" + it + ".caffemodel";
      std::string critic_solver = snapshot_prefix + "_critic_iter_" + it + ".solverstate";
      std::string critic_model = snapshot_prefix + "_critic_iter_" + it + ".caffemodel";
      std::string replaymemory = snapshot_prefix + "_iter_" + it + ".replaymemory";
      if (is_regular_file(actor_solver) && is_regular_file(actor_model) &&
          is_regular_file(critic_solver) && is_regular_file(critic_model) &&
          is_regular_file(replaymemory)) {
        max_iter = iter;
        latest = std::make_tuple(actor_solver, critic_solver, replaymemory);
      }
    }
  }
  return latest;
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

void DQN::LoadTrainedModel(const std::string& actor_model_bin,
                           const std::string& critic_model_bin) {
  actor_net_->CopyTrainedLayersFrom(actor_model_bin);
  critic_net_->CopyTrainedLayersFrom(critic_model_bin);
}

void DQN::RestoreSolver(const std::string& actor_solver_bin,
                        const std::string& critic_solver_bin) {
  actor_solver_->Restore(actor_solver_bin.c_str());
  critic_solver_->Restore(critic_solver_bin.c_str());
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
  CHECK_EQ(actor_solver_->iter(), critic_solver_->iter());
  actor_solver_->Snapshot(snapshot_prefix + "_actor");
  critic_solver_->Snapshot(snapshot_prefix + "_critic");
  int snapshot_iter = current_iteration() + 1;
  std::string fname = snapshot_prefix + "_actor_iter_" + std::to_string(snapshot_iter);
  CHECK(is_regular_file(fname + ".caffemodel"));
  CHECK(is_regular_file(fname + ".solverstate"));
  fname = snapshot_prefix + "_critic_iter_" + std::to_string(snapshot_iter);
  CHECK(is_regular_file(fname + ".caffemodel"));
  CHECK(is_regular_file(fname + ".solverstate"));
  if (snapshot_memory) {
    fname = snapshot_prefix + "_iter_" + std::to_string(snapshot_iter);
    std::string mem_fname = fname + ".replaymemory";
    LOG(INFO) << "Snapshotting memory to " << mem_fname;
    SnapshotReplayMemory(mem_fname);
    CHECK(is_regular_file(mem_fname));
  }
  if (remove_old) {
    RemoveSnapshots(snapshot_prefix, snapshot_iter);
  }
}

void DQN::Initialize() {
  // Initialize net and solver
  actor_solver_.reset(caffe::GetSolver<float>(actor_solver_param_));
  critic_solver_.reset(caffe::GetSolver<float>(critic_solver_param_));
  actor_net_ = actor_solver_->net();
  critic_net_ = critic_solver_->net();
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);
  HasBlobSize(*actor_net_, "states",
              {kMinibatchSize,kStateInputCount,kStateDataSize,1});
  // HasBlobSize(*critic_net_, "states",
  //             {kMinibatchSize,kStateInputCount,kStateDataSize,1});
  HasBlobSize(*critic_net_, "target",
              {kMinibatchSize,kOutputCount,1,1});
  CloneNet(critic_net_, critic_target_net_);
}

float DQN::SelectAction(const ActorInputStates& last_states, const double epsilon) {
  return SelectActions(std::vector<ActorInputStates>{{last_states}}, epsilon)[0];
}

std::vector<float> DQN::SelectActions(const std::vector<ActorInputStates>& states_batch,
                                    const double epsilon) {
  assert(epsilon >= 0.0 && epsilon <= 1.0);
  assert(states_batch.size() <= kMinibatchSize);
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < epsilon) {
    // Select randomly
    std::vector<float> actions(states_batch.size());
    for (int i=0; i<actions.size(); ++i) {
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

float DQN::SelectActionGreedily(caffe::Net<float>& net,
                                      const ActorInputStates& last_states) {
  return SelectActionGreedily(
      net, std::vector<ActorInputStates>{{last_states}}).front();
}

std::vector<float> DQN::SelectActionGreedily(
    caffe::Net<float>& net,
    const std::vector<ActorInputStates>& last_states_batch) {
  assert(last_states_batch.size() <= kMinibatchSize);
  StateLayerInputData states_input;
  // Input states to the net and compute Q values for each legal actions
  for (auto i = 0; i < last_states_batch.size(); ++i) {
    for (auto j = 0; j < kStateInputCount; ++j) {
      const auto& state_data = last_states_batch[i][j];
      std::copy(state_data->begin(),
                state_data->end(),
                states_input.begin() + i * kActorInputDataSize +
                j * kStateDataSize);
    }
  }
  InputDataIntoLayers(net, states_input.data(), NULL, NULL);
  net.ForwardPrefilled(nullptr);
  // Collect the Results
  std::vector<float> results(last_states_batch.size());
  const auto kickangle_blob = net.blob_by_name("kickangle");
  for (auto i = 0; i < last_states_batch.size(); ++i) {
    // Get the kickangle from the net
    results[i] = kickangle_blob->data_at(i, 0, 0, 0);
  }
  return results;
}

void DQN::AddTransition(const Transition& transition) {
  if (replay_memory_.size() == replay_memory_capacity_) {
    replay_memory_.pop_front();
  }
  replay_memory_.push_back(transition);
}

void DQN::UpdateCritic() {
  // Every clone_iters steps, update the clone_net_ to equal the primary net
  if (current_iteration() % clone_frequency_ == 0) {
    LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
    CloneNet(critic_net_, critic_target_net_);
  }

  // Sample transitions from replay memory
  std::vector<int> transitions;
  transitions.reserve(kMinibatchSize);
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto random_transition_idx =
        std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
            random_engine);
    transitions.push_back(random_transition_idx);
  }
  // Compute target values: max_a Q(s',a)
  std::vector<ActorInputStates> target_last_states_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    if (!std::get<3>(transition)) {
      // This is a terminal state
      continue;
    }
    // Compute target ActorInputStates value
    ActorInputStates target_last_states;
    for (auto i = 0; i < kStateInputCount - 1; ++i) {
      target_last_states[i] = std::get<0>(transition)[i + 1];
    }
    target_last_states[kStateInputCount - 1] = std::get<3>(transition).get();
    target_last_states_batch.push_back(target_last_states);
  }
  // Get the actions value from the Actor network
  const std::vector<float> actions =
      SelectActionGreedily(*actor_net_, target_last_states_batch);
  // Get the Q-Values with respect to the actions value from Critic network
  const std::vector<float> q_values = GetQValue(*critic_target_net_,
                                                target_last_states_batch, actions);
  CriticStateLayerInputData states_input;
  TargetLayerInputData target_input;
  std::fill(states_input.begin(), states_input.end(), 0.0);
  std::fill(target_input.begin(), target_input.end(), 0.0);
  // Fill the StateInputLayer and the TargetInputLayer
  auto target_value_idx = 0;
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto& transition = replay_memory_[transitions[i]];
    const float action = std::get<1>(transition);
    const float reward = std::get<2>(transition);
    CHECK(reward >= -1.0 && reward <= 1.0);
    const auto target = std::get<3>(transition) ?
        reward + gamma_ * q_values[target_value_idx++] :
        reward;
    CHECK(!std::isnan(target));
    target_input[i * kOutputCount] = target;
    for (auto j = 0; j < kStateInputCount; ++j) {
      const auto& state_data = std::get<0>(transition)[j];
      std::copy(state_data->begin(), state_data->end(), states_input.begin() +
                i * kCriticInputDataSize + j * kStateDataSize);
    }
    for(int j = 0; j < kOutputCount; j++) {
      states_input[(i+1) * kCriticInputDataSize - kOutputCount + j] = action;
    }
  }
  InputDataIntoLayers(*critic_net_, states_input.data(), target_input.data(), NULL);
  critic_solver_->Step(1);
}

void DQN::UpdateActor() {
  // Sample transitions from replay memory
  std::vector<int> transitions;
  transitions.reserve(kMinibatchSize);
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto random_transition_idx =
        std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
            random_engine);
    transitions.push_back(random_transition_idx);
  }
  // Compute target values: max_a Q(s',a)
  std::vector<ActorInputStates> states_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    // Compute target value
    ActorInputStates last_states;
    for (auto i = 0; i < kStateInputCount; ++i) {
      last_states[i] = std::get<0>(transition)[i];
    }
    states_batch.push_back(last_states);
  }
  float start_q = 0;
  for(int k = 0; k<5; k++) {
    // Get the actions and q_values from the network
    const std::vector<float> actions =
        SelectActionGreedily(*actor_net_, states_batch);
    const std::vector<float> q_values = GetQValue(*critic_target_net_,
                                                  states_batch, actions);
    LOG(INFO) << "Iteration " << k <<" in updating the Actor network";
    LOG(INFO) << "q_value[0] = " << q_values[0] << " action[0] = " << actions[0];
    if (k == 0) {
      start_q = q_values[0];
    }
    // Set the q_value diff to be a positve num
    const auto q_values_blob = critic_target_net_->blob_by_name("q_values");
    float* q_values_diff = q_values_blob->mutable_cpu_diff();
    // TODO change the parameter value diff_num
    float diff_num = 10.0;
    for (int i = 0; i < kMinibatchSize; i++) {
      q_values_diff[q_values_blob->offset(i,0,0,0)] = diff_num;
    }
    // Run the network backwards to see the resulting actions diff at the input layer
    const std::vector<std::string> names = critic_target_net_->layer_names();
    int pos = std::distance(names.begin(),
                            std::find(names.begin(), names.end(), "ip2_layer"));
    critic_target_net_->BackwardFrom(pos);
    std::vector<float> data_diff(kMinibatchSize);
    // std::vector<float> data_all_states_diff(kCriticInputDataSize);
    const auto states_blob = critic_target_net_->blob_by_name("states");
    // Set the diff in the actions ouput in Actor network
    for (int i = 0; i < kMinibatchSize; i++) {
      float d = states_blob->diff_at(i, 0, kMinibatchSize - 1, 0);
      data_diff[i] = d;//q_values[i] > 0 ? d : -d;
    }
    LOG(INFO) << "data_diff[0] = " << data_diff[0];
    // for (int t = 0; t < kCriticInputDataSize ; t++) {
    //   data_all_states_diff[t] = states_blob->diff_at(0, 0, t, 0);
    //   // LOG(INFO) << "data_all_states_diff[" << t << "] = " << data_all_states_diff[t];
    // }
    const auto kickangle_blob = actor_net_->blob_by_name("kickangle");
    float* kickangle_diff = kickangle_blob->mutable_cpu_diff();
    for (int i = 0; i < kMinibatchSize; i++) {
      kickangle_diff[kickangle_blob->offset(i,0,0,0)] = data_diff[i];
    }
    // Run backwards to update the Actor network
    actor_net_->Backward();
    actor_solver_->ComputeUpdateValue();
    actor_net_->Update();
    LOG(INFO) << "QDiff: " << q_values[0] - start_q;
  }
  exit(0);
}


std::vector<float> DQN::GetQValue(
    caffe::Net<float>& net,
    std::vector<ActorInputStates>& last_actor_states_batch,
    const std::vector<float>& actions) {
  CHECK_LE(last_actor_states_batch.size(), kMinibatchSize);
  CHECK_EQ(last_actor_states_batch.size(), actions.size());
  // Tansform the Actor input states to Critic input states
  CriticStateLayerInputData states_input;
  for (auto i = 0; i < last_actor_states_batch.size(); ++i) {
    for (auto j = 0; j < kStateInputCount; ++j) {
      const auto& state_data = last_actor_states_batch[i][j];
      std::copy(state_data->begin(),
                state_data->end(),
                states_input.begin() + i * kCriticInputDataSize +
                j * kStateDataSize);
    }
    for(int j = 0; j < kOutputCount; j++) {
      states_input[(i+1) * kCriticInputDataSize - kOutputCount + j] = actions[i];
    }
  }
  // Target Layer is empty
  TargetLayerInputData target_input;
  std::fill(target_input.begin(), target_input.end(), 0.0);
  InputDataIntoLayers(net, states_input.data(), target_input.data(), NULL);
  net.ForwardPrefilled(nullptr);
  // Collect the Results
  std::vector<float> results(last_actor_states_batch.size());
  const auto q_values_blob = net.blob_by_name("q_values");
  for (auto i = 0; i < last_actor_states_batch.size(); ++i) {
    // Get the kickangle from the net
    results[i] = q_values_blob->data_at(i, 0, 0, 0);
  }
  return results;
}

void DQN::CloneNet(NetSp& net_from, NetSp& net_to) {
  caffe::NetParameter net_param;
  net_from->ToProto(&net_param);
  net_param.set_force_backward(true);
  if (!net_to) {
    net_to.reset(new caffe::Net<float>(net_param));
  } else {
    net_to->CopyTrainedLayersFrom(net_param);
  }
}

void DQN::InputDataIntoLayers(caffe::Net<float>& net,
                              float* states_input,
                              float* target_input,
                              float* filter_input) {
  if (states_input != NULL) {
    const auto state_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name("state_input_layer"));
    CHECK(state_input_layer);
    state_input_layer->Reset(states_input, states_input,
                             state_input_layer->batch_size());
  }
  if (target_input != NULL) {
    const auto target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name("target_input_layer"));
    CHECK(target_input_layer);
    target_input_layer->Reset(target_input, target_input,
                              target_input_layer->batch_size());
  }
  if (filter_input != NULL) {
    const auto filter_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name("filter_input_layer"));
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
  const ActorInputStates& first_transition = std::get<0>(replay_memory_[0]);
  for (int i = 0; i < kStateInputCount - 1; ++i) {
    const ActorStateDataSp state = first_transition[i];
    out.write((char*)state->begin(), kStateDataSize * sizeof(float));
  }
  // For all other transitions, save only the current state
  for (const Transition& t : replay_memory_) {
    const ActorInputStates& states = std::get<0>(t);
    const ActorStateDataSp curr_state = states.back();
    out.write((char*)curr_state->begin(), kStateDataSize * sizeof(float));
    const float& action = std::get<1>(t);
    out.write((char*)&action, sizeof(float));
    const float& reward = std::get<2>(t);
    out.write((char*)&reward, sizeof(float));
  }
  LOG(INFO) << "Saved memory of size " << memory_size();
}

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
  std::deque<dqn::ActorStateDataSp> past_states;
  // First read the state history
  for (int i = 0; i < kStateInputCount - 1; ++i) {
    ActorStateDataSp state = std::make_shared<ActorStateData>();
    in.read((char*)state->begin(), kStateDataSize * sizeof(float));
    past_states.push_back(state);
  }
  for (int i = 0; i < num_transitions; ++i) {
    Transition& t = replay_memory_[i];
    ActorStateDataSp state = std::make_shared<ActorStateData>();
    in.read((char*)state->begin(), kStateDataSize * sizeof(float));
    past_states.push_back(state);
    while (past_states.size() > kStateInputCount) {
      past_states.pop_front();
    }
    CHECK_EQ(past_states.size(), kStateInputCount);
    ActorInputStates& states = std::get<0>(t);
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
