#include "dqn.hpp"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

namespace dqn {

template <typename Dtype>
void HasBlobSize(caffe::Net<Dtype>& net,
                 const std::string& blob_name,
                 const std::vector<int> expected_shape) {
  net.has_blob(blob_name);
  const caffe::Blob<Dtype>& blob = *net.blob_by_name(blob_name);
  const std::vector<int>& blob_shape = blob.shape();
  CHECK_EQ(blob_shape.size(), expected_shape.size());
  CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
                   expected_shape.begin()));
}

void DQN::LoadTrainedModel(const std::string& model_bin) {
  net_->CopyTrainedLayersFrom(model_bin);
}

void DQN::RestoreSolver(const std::string& solver_bin) {
  solver_->Restore(solver_bin.c_str());
}

void DQN::Initialize() {
  // Initialize net and solver
  solver_.reset(caffe::GetSolver<float>(solver_param_));
  net_ = solver_->net();
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);
  HasBlobSize(*net_, "states", {kMinibatchSize,kInputCount,kStateDataSize,1});
  HasBlobSize(*net_, "target", {kMinibatchSize,kOutputCount,1,1});
  HasBlobSize(*net_, "filter", {kMinibatchSize,kOutputCount,1,1});
  ClonePrimaryNet();
}

int DQN::SelectAction(const InputStates& last_states, const double epsilon) {
  return SelectActions(std::vector<InputStates>{{last_states}}, epsilon)[0];
}

std::vector<int> DQN::SelectActions(const std::vector<InputStates>& states_batch,
                                    const double epsilon) {
  assert(epsilon >= 0.0 && epsilon <= 1.0);
  assert(states_batch.size() <= kMinibatchSize);
  std::vector<int> actions(states_batch.size());
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < epsilon) {
    // Select randomly
    for (int i=0; i<actions.size(); ++i) {
      const auto random_idx = std::uniform_int_distribution<int>
          (0, legal_actions_.size() - 1)(random_engine);
      actions[i] = legal_actions_[random_idx];
    }
  } else {
    // Select greedily
    std::vector<ActionValue> q = SelectActionGreedily(*net_, states_batch);
    assert(q.size() == actions.size());
    for (int i=0; i<actions.size(); ++i) {
      actions[i] = q[i].first;
    }
  }
  return actions;
}

ActionValue DQN::SelectActionGreedily(caffe::Net<float>& net,
                                      const InputStates& last_states) {
  return SelectActionGreedily(
      net, std::vector<InputStates>{{last_states}}).front();
}

std::vector<ActionValue> DQN::SelectActionGreedily(
    caffe::Net<float>& net,
    const std::vector<InputStates>& last_states_batch) {
  assert(last_states_batch.size() <= kMinibatchSize);
  std::array<float, kMinibatchDataSize> states_input;
  // Input states to the net and compute Q values for each legal actions
  for (auto i = 0; i < last_states_batch.size(); ++i) {
    for (auto j = 0; j < kInputCount; ++j) {
      const auto& state_data = last_states_batch[i][j];
      std::copy(state_data->begin(),
                state_data->end(),
                states_input.begin() + i * kInputDataSize +
                j * kStateDataSize);
    }
  }
  InputDataIntoLayers(net, states_input, dummy_input_data_, dummy_input_data_);
  net.ForwardPrefilled(nullptr);
  // Collect the Results
  std::vector<ActionValue> results;
  results.reserve(last_states_batch.size());
  const auto q_values_blob = net.blob_by_name("q_values");
  for (auto i = 0; i < last_states_batch.size(); ++i) {
    // Get the Q values from the net
    const auto action_evaluator = [&](int action) {
      const auto q = q_values_blob->data_at(i, static_cast<int>(action), 0, 0);
      assert(!std::isnan(q));
      return q;
    };
    std::vector<float> q_values(legal_actions_.size());
    std::transform(legal_actions_.begin(), legal_actions_.end(),
                   q_values.begin(), action_evaluator);
    // Select the action with the maximum Q value
    const auto max_idx = std::distance(
        q_values.begin(),
        std::max_element(q_values.begin(), q_values.end()));
    results.emplace_back(legal_actions_[max_idx], q_values[max_idx]);
  }
  return results;
}

void DQN::AddTransition(const Transition& transition) {
  if (replay_memory_.size() == replay_memory_capacity_) {
    replay_memory_.pop_front();
  }
  replay_memory_.push_back(transition);
}

void DQN::Update() {
  // Every clone_iters steps, update the clone_net_ to equal the primary net
  if (current_iteration() % clone_frequency_ == 0) {
    LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
    ClonePrimaryNet();
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
  std::vector<InputStates> target_last_states_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    if (!std::get<3>(transition)) {
      // This is a terminal state
      continue;
    }
    // Compute target value
    InputStates target_last_states;
    for (auto i = 0; i < kInputCount - 1; ++i) {
      target_last_states[i] = std::get<0>(transition)[i + 1];
    }
    target_last_states[kInputCount - 1] = std::get<3>(transition).get();
    target_last_states_batch.push_back(target_last_states);
  }
  // Get the update targets from the cloned network
  const auto actions_and_values =
      SelectActionGreedily(*clone_net_, target_last_states_batch);
  StateLayerInputData states_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  std::fill(target_input.begin(), target_input.end(), 0.0f);
  std::fill(filter_input.begin(), filter_input.end(), 0.0f);
  auto target_value_idx = 0;
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto& transition = replay_memory_[transitions[i]];
    const auto action = std::get<1>(transition);
    assert(static_cast<int>(action) < kOutputCount);
    const auto reward = std::get<2>(transition);
    assert(reward >= -1.0 && reward <= 1.0);
    const auto target = std::get<3>(transition) ?
          reward + gamma_ * actions_and_values[target_value_idx++].second :
          reward;
    assert(!std::isnan(target));
    target_input[i * kOutputCount + static_cast<int>(action)] = target;
    filter_input[i * kOutputCount + static_cast<int>(action)] = 1;
    for (auto j = 0; j < kInputCount; ++j) {
      const auto& frame_data = std::get<0>(transition)[j];
      std::copy(frame_data->begin(), frame_data->end(), states_input.begin() +
                i * kInputDataSize + j * kStateDataSize);
    }
  }
  InputDataIntoLayers(*net_, states_input, target_input, filter_input);
  solver_->Step(1);
}

void DQN::ClonePrimaryNet() {
  caffe::NetParameter net_param;
  net_->ToProto(&net_param);
  clone_net_.reset(new caffe::Net<float>(net_param));
}

void DQN::InputDataIntoLayers(caffe::Net<float>& net,
                              const StateLayerInputData& states_input,
                              const TargetLayerInputData& target_input,
                              const FilterLayerInputData& filter_input) {
  // Get the layers by name and cast them to memory layers
  const auto states_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net.layer_by_name("state_input_layer"));
  const auto target_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net.layer_by_name("target_input_layer"));
  const auto filter_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net.layer_by_name("filter_input_layer"));
  // Make sure they were found and correctly casted
  assert(states_input_layer);
  assert(target_input_layer);
  assert(filter_input_layer);
  // Input the data into the layers
  states_input_layer->Reset(const_cast<float*>(states_input.data()),
                            dummy_input_data_.data(), kMinibatchSize);
  target_input_layer->Reset(const_cast<float*>(target_input.data()),
                            dummy_input_data_.data(), kMinibatchSize);
  filter_input_layer->Reset(const_cast<float*>(filter_input.data()),
                            dummy_input_data_.data(), kMinibatchSize);
}
}
