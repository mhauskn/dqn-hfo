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
  HasBlobSize(*actor_net_, "states",
              {kMinibatchSize,kStateInputCount,kStateDataSize,1});
  // HasBlobSize(*critic_net_, "states",
  //             {kMinibatchSize,kStateInputCount,kStateDataSize,1});
  // HasBlobSize(*critic_net_, "target",
  //             {kMinibatchSize,kOutputCount,1,1});
  CloneNet(critic_net_, critic_target_net_);
}

Action DQN::SelectAction(const ActorInputStates& last_states, const double epsilon) {
  return SelectActions(std::vector<ActorInputStates>{{last_states}}, epsilon)[0];
}

std::vector<Action> DQN::SelectActions(const std::vector<ActorInputStates>&
                                       states_batch, const double epsilon) {
  assert(epsilon >= 0.0 && epsilon <= 1.0);
  assert(states_batch.size() <= kMinibatchSize);
  // if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < epsilon) {
  //   // Select randomly
  //   std::vector<float> actions(states_batch.size());
  //   for (int i=0; i<actions.size(); ++i) {
  //     float kickangle = std::uniform_real_distribution<float>
  //         (-90.0, 90.0)(random_engine);
  //     actions[i] = kickangle;
  //   }
  //   return actions;
  // } else {
    // Select greedily
  return SelectActionGreedily(*actor_net_, states_batch);
  // }
}

Action DQN::SelectActionGreedily(caffe::Net<float>& net,
                                 const ActorInputStates& last_states) {
  return SelectActionGreedily(
      net, std::vector<ActorInputStates>{{last_states}}).front();
}

std::vector<Action> DQN::SelectActionGreedily(
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
  ActionTargetLayerInputData target_action_choice_batch;
  ActionparaTargetLayerInputData target_actionpara_batch;
  FilterLayerInputData filter_batch;
  std::fill(target_action_choice_batch.begin(),
            target_action_choice_batch.end(), 0.0);
  std::fill(target_actionpara_batch.begin(), target_actionpara_batch.end(), 0.0);
  std::fill(filter_batch.begin(), filter_batch.end(), 0.0);
  InputDataIntoLayers(net, states_input.data(), target_action_choice_batch.data(),
                      target_actionpara_batch.data(), filter_batch.data());
  net.ForwardPrefilled(nullptr);
  // Collect the Results
  std::vector<int> action_results(last_states_batch.size());
  std::vector<Action> actions(last_states_batch.size());
  const auto action_blob = net.blob_by_name("action");
  const auto actionpara_blob = net.blob_by_name("actionpara");
  for (int i = 0; i < last_states_batch.size(); ++i) {
  // Get the action from the net
    action_results[i] = 0;
    float action_value = action_blob->data_at(i, 0, 0, 0);
    for (int j = 0; j < kActionCount; ++j)
      if (action_value < action_blob->data_at(i, j, 0, 0)) {
          action_value = action_blob->data_at(i, j, 0, 0);
          action_results[i] = j;
        }
    switch (action_results[i]) {
      case 0:
        actions[i].action = DASH;
        actions[i].arg1 = actionpara_blob->data_at(i, 0, 0, 0);
        actions[i].arg2 = actionpara_blob->data_at(i, 1, 0, 0);
        break;
      case 1:
        actions[i].action = TURN;
        actions[i].arg1 = actionpara_blob->data_at(i, 2, 0, 0);
        break;
      case 2:
        actions[i].action = TACKLE;
        actions[i].arg1 = actionpara_blob->data_at(i, 3, 0, 0);
        break;
      case 3:
        actions[i].action = KICK;
        actions[i].arg1 = actionpara_blob->data_at(i, 4, 0, 0);
        actions[i].arg2 = actionpara_blob->data_at(i, 5, 0, 0);
        break;
    }
  }
  // return results;
  return actions;
}

void DQN::AddTransition(const Transition& transition) {
  if (replay_memory_.size() == replay_memory_capacity_) {
    replay_memory_.pop_front();
  }
  replay_memory_.push_back(transition);
}

// void DQN::UpdateCritic() {
//   // Every clone_iters steps, update the clone_net_ to equal the primary net
//   if (current_iteration() % clone_frequency_ == 0) {
//     LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
//     CloneNet(critic_net_, critic_target_net_);
//   }

//   // Sample transitions from replay memory
//   std::vector<int> transitions;
//   transitions.reserve(kMinibatchSize);
//   for (auto i = 0; i < kMinibatchSize; ++i) {
//     const auto random_transition_idx =
//         std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
//             random_engine);
//     transitions.push_back(random_transition_idx);
//   }
//   // Compute target values: max_a Q(s',a)
//   std::vector<ActorInputStates> target_last_states_batch;
//   for (const auto idx : transitions) {
//     const auto& transition = replay_memory_[idx];
//     if (!std::get<3>(transition)) {
//       // This is a terminal state
//       continue;
//     }
//     // Compute target ActorInputStates value
//     ActorInputStates target_last_states;
//     for (auto i = 0; i < kStateInputCount - 1; ++i) {
//       target_last_states[i] = std::get<0>(transition)[i + 1];
//     }
//     target_last_states[kStateInputCount - 1] = std::get<3>(transition).get();
//     target_last_states_batch.push_back(target_last_states);
//   }
//   // Get the actions value from the Actor network
//   const std::vector<float> actions =
//       SelectActionGreedily(*actor_net_, target_last_states_batch);
//   // Get the Q-Values with respect to the actions value from Critic network
//   const std::vector<float> q_values = GetQValue(*critic_target_net_,
//                                                 target_last_states_batch, actions);
//   CriticStateLayerInputData states_input;
//   TargetLayerInputData target_input;
//   std::fill(states_input.begin(), states_input.end(), 0.0);
//   std::fill(target_input.begin(), target_input.end(), 0.0);
//   // Fill the StateInputLayer and the TargetInputLayer
//   auto target_value_idx = 0;
//   for (auto i = 0; i < kMinibatchSize; ++i) {
//     const auto& transition = replay_memory_[transitions[i]];
//     const float action = std::get<1>(transition);
//     const float reward = std::get<2>(transition);
//     CHECK(reward >= -1.0 && reward <= 1.0);
//     const auto target = std::get<3>(transition) ?
//         reward + gamma_ * q_values[target_value_idx++] :
//         reward;
//     CHECK(!std::isnan(target));
//     target_input[i * kOutputCount] = target;
//     for (auto j = 0; j < kStateInputCount; ++j) {
//       const auto& state_data = std::get<0>(transition)[j];
//       std::copy(state_data->begin(), state_data->end(), states_input.begin() +
//                 i * kCriticInputDataSize + j * kStateDataSize);
//     }
//     for(int j = 0; j < kOutputCount; j++) {
//       states_input[(i+1) * kCriticInputDataSize - kOutputCount + j] = action;
//     }
//   }
//   InputDataIntoLayers(*critic_net_, states_input.data(), target_input.data(), NULL);
//   critic_solver_->Step(1);
// }

std::pair<float,float> DQN::UpdateActor(int update_idx, bool update,
                                        std::vector<std::pair<int,int>>& accuracy,
                                        std::vector<float>& deviation) {
  StateLayerInputData past_states_batch;
  ActionTargetLayerInputData target_action_choice_batch;
  ActionparaTargetLayerInputData target_actionpara_batch;
  FilterLayerInputData filter_batch;
  std::fill(past_states_batch.begin(), past_states_batch.end(), 0.0);
  std::fill(target_action_choice_batch.begin(),
            target_action_choice_batch.end(), 0.0);
  std::fill(target_actionpara_batch.begin(), target_actionpara_batch.end(), 0.0);
  std::fill(filter_batch.begin(), filter_batch.end(), 0.0);
  std::vector<Action> target_action_batch(kMinibatchSize);
  for (auto i = 0; i < kMinibatchSize; ++i) {
    // Sample transitions from replay memory_size
    const auto& transition = replay_memory_[update_idx * kMinibatchSize + i];
    // fill state input
    for (auto j = 0; j < kStateInputCount; ++j) {
      const auto& state_data = std::get<0>(transition)[j];
      std::copy(state_data->begin(), state_data->end(), past_states_batch.begin() +
                i * kActorInputDataSize + j * kStateDataSize);
    }
    target_action_batch[i] = std::get<1>(transition);
    //fill action choice
    int target_action_choice = (int)target_action_batch[i].action;
    target_action_choice_batch[i] = (float)target_action_choice;
    //fill actionpara
    switch (target_action_choice) {
      case 0:
        target_actionpara_batch[kActionparaCount * i] = target_action_batch[i].arg1;
        target_actionpara_batch[kActionparaCount * i + 1] = target_action_batch[i].arg2;
        filter_batch[kActionparaCount * i] = 1;
        filter_batch[kActionparaCount * i + 1] = 1;
        break;
      case 1:
        target_actionpara_batch[kActionparaCount * i + 2] = target_action_batch[i].arg1;
        filter_batch[kActionparaCount * i + 2] = 1;
        break;
      case 2:
        target_actionpara_batch[kActionparaCount * i + 3] = target_action_batch[i].arg1;
        filter_batch[kActionparaCount * i + 3] = 1;
        break;
      case 3:
        target_actionpara_batch[kActionparaCount * i + 4] = target_action_batch[i].arg1;
        target_actionpara_batch[kActionparaCount * i + 5] = target_action_batch[i].arg2;
        filter_batch[kActionparaCount * i + 4] = 1;
        filter_batch[kActionparaCount * i + 5] = 1;
        break;
    }
  }
  InputDataIntoLayers(*actor_net_, past_states_batch.data(),
                      target_action_choice_batch.data(),
                      target_actionpara_batch.data(),
                      filter_batch.data());
  if (update == true) {
      actor_solver_->Step(1);
  }  else {
    actor_net_->ForwardPrefilled(nullptr);
  }

  const auto action_blob = actor_net_->blob_by_name("action");
  const auto actionpara_blob = actor_net_->blob_by_name("actionpara");
  for (int i = 0; i < kMinibatchSize; ++i) {
    int action_result = 0;
    float action_value = action_blob->data_at(i, 0, 0, 0);
    for (int j = 0; j < kActionCount; ++j)
      if (action_value < action_blob->data_at(i, j, 0, 0)) {
          action_value = action_blob->data_at(i, j, 0, 0);
          action_result = j;
        }
    accuracy[(int)target_action_choice_batch[i]].second++;
    if (action_result == (int)target_action_choice_batch[i]) {
      switch (action_result) {
        case 0:
          deviation[0] += std::abs(actionpara_blob->data_at(i, 0, 0, 0) -
                                 target_actionpara_batch[kActionparaCount * i + 0]);
          deviation[1] += std::abs(actionpara_blob->data_at(i, 1, 0, 0) -
                                 target_actionpara_batch[kActionparaCount * i + 1]);
          break;
        case 1:
          deviation[2] += std::abs(actionpara_blob->data_at(i, 2, 0, 0) -
                                 target_actionpara_batch[kActionparaCount * i + 2]);
          break;
        case 2:
          deviation[3] += std::abs(actionpara_blob->data_at(i, 3, 0, 0) -
                                 target_actionpara_batch[kActionparaCount * i + 3]);
          break;
        case 3:
          deviation[4] += std::abs(actionpara_blob->data_at(i, 4, 0, 0) -
                                 target_actionpara_batch[kActionparaCount * i + 4]);
          deviation[5] += std::abs(actionpara_blob->data_at(i, 5, 0, 0) -
                                 target_actionpara_batch[kActionparaCount * i + 5]);
          break;
      }
      accuracy[(int)target_action_choice_batch[i]].first++;
    }
  }

  const auto euclideanloss_blob = actor_net_->blob_by_name("euclideanloss");
  const auto softmaxloss_blob = actor_net_->blob_by_name("softmaxloss");
  float euclideanloss = euclideanloss_blob->data_at(0, 0, 0, 0);
  float softmaxloss = softmaxloss_blob->data_at(0, 0, 0, 0);
  return std::make_pair(euclideanloss, softmaxloss);


  // // Get the actions and q_values from the network
  // const std::vector<Action> actions_batch =
  //     SelectActionGreedily(*actor_net_, states_batch);
  // // const std::vector<float> q_values = GetQValue(
  // //     *critic_target_net_, states_batch, actions);
  // const auto q_values_blob = critic_target_net_->blob_by_name("q_values");
  // float* q_values_diff = q_values_blob->mutable_cpu_diff();
  // for (int i = 0; i < kMinibatchSize; i++) {
  //   q_values_diff[q_values_blob->offset(i,0,0,0)] = -1.0;
  // }
  // // Run the network backwards and get the action diff at the input layer
  // const std::vector<std::string>& names = critic_target_net_->layer_names();
  // int ip2_indx = std::distance(names.begin(),
  //                              std::find(names.begin(), names.end(), "ip2_layer"));
  // CHECK_LT(ip2_indx, names.size()) << "[Actor Update] Couldn't find ip2_layer";
  // critic_target_net_->BackwardFrom(ip2_indx);
  // CHECK(critic_target_net_->has_blob("states"));
  // const auto states_blob = critic_target_net_->blob_by_name("states");
  // CHECK(actor_net_->has_blob("kickangle"));
  // const auto kickangle_blob = actor_net_->blob_by_name("kickangle");
  // // Set the diff in the actions ouput in Actor network
  // float* kickangle_diff = kickangle_blob->mutable_cpu_diff();
  // for (int i = 0; i < kMinibatchSize; i++) {
  //   kickangle_diff[kickangle_blob->offset(i,0,0,0)] =
  //       states_blob->diff_at(i, 0, kCriticInputDataSize-1, 0);
  // }
  // // Run backwards to update the Actor network
  // actor_net_->Backward();
  // actor_solver_->ComputeUpdateValue();
  // actor_solver_->set_iter(actor_solver_->iter() + 1);
  // actor_net_->Update();
}


// std::vector<float> DQN::GetQValue(
//     caffe::Net<float>& net,
//     std::vector<ActorInputStates>& last_actor_states_batch,
//     const std::vector<float>& actions) {
//   CHECK_LE(last_actor_states_batch.size(), kMinibatchSize);
//   CHECK_EQ(last_actor_states_batch.size(), actions.size());
//   // Tansform the Actor input states to Critic input states
//   CriticStateLayerInputData states_input;
//   for (auto i = 0; i < last_actor_states_batch.size(); ++i) {
//     for (auto j = 0; j < kStateInputCount; ++j) {
//       const auto& state_data = last_actor_states_batch[i][j];
//       std::copy(state_data->begin(),
//                 state_data->end(),
//                 states_input.begin() + i * kCriticInputDataSize +
//                 j * kStateDataSize);
//     }
//     for(int j = 0; j < kOutputCount; j++) {
//       states_input[(i+1) * kCriticInputDataSize - kOutputCount + j] = actions[i];
//     }
//   }
//   // Target Layer is empty
//   TargetLayerInputData target_input;
//   std::fill(target_input.begin(), target_input.end(), 0.0);
//   InputDataIntoLayers(net, states_input.data(), target_input.data(), NULL);
//   net.ForwardPrefilled(nullptr);
//   // Collect the Results
//   std::vector<float> results(last_actor_states_batch.size());
//   const auto q_values_blob = net.blob_by_name("q_values");
//   for (auto i = 0; i < last_actor_states_batch.size(); ++i) {
//     // Get the kickangle from the net
//     results[i] = q_values_blob->data_at(i, 0, 0, 0);
//   }
//   return results;
// }

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
                              float* action_target_input,
                              float* actionpara_target_input,
                              float* filter_input) {
  if (states_input != NULL) {
    const auto state_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name("state_input_layer"));
    CHECK(state_input_layer);
    state_input_layer->Reset(states_input, states_input,
                             state_input_layer->batch_size());
  }
  if (action_target_input != NULL) {
    const auto action_target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name("action_target_input_layer"));
    CHECK(action_target_input_layer);
    action_target_input_layer->Reset(action_target_input, action_target_input,
                              action_target_input_layer->batch_size());
  }
  if (actionpara_target_input != NULL) {
    const auto actionpara_target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net.layer_by_name("actionpara_target_input_layer"));
    CHECK(actionpara_target_input_layer);
    actionpara_target_input_layer->Reset(actionpara_target_input,
                                         actionpara_target_input,
                                         actionpara_target_input_layer->batch_size());
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
    const Action& action = std::get<1>(t);
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

void DQN::LoadMimicData(const std::string& filename){
  LOG(INFO) << "Loading mimic data from /scratch/cluster/chen/mimic_data/" << filename;
  ClearReplayMemory();
  std::ifstream ifile(("/scratch/cluster/chen/mimic_data/" + filename).c_str(),std::ios_base::in);
  if (!ifile.is_open())
    LOG(INFO) << "Cannot open file /scratch/cluster/chen/mimic_data/" << filename;
  std::string temp_string;
  int temp_int;
  bool memory_full = false;
  while (!ifile.eof()) {
    ifile >> temp_int >> temp_int >> temp_string >> temp_string;
    //    LOG(INFO) << temp_string;
    if(temp_string == "player_agent.cpp:")
      std::getline(ifile, temp_string);
    else {
      CHECK(temp_string == "GameStatus") << "Not match.";
      std::getline(ifile, temp_string);
      break;
    }
  }
  while (!ifile.eof() && !memory_full) {
    std::deque<dqn::ActorStateDataSp> past_states;
    for (int i = 0; i < kStateInputCount - 1; ++i) {
      ifile >> temp_int >> temp_int >> temp_string >> temp_string;
      CHECK(temp_string == "StateFeatures") << "Not match," << i;
      ActorStateDataSp statesp = std::make_shared<ActorStateData>();
      for (int i = 0; i < kStateDataSize; ++i) {
        ifile >> (*statesp)[i];
      }
      past_states.push_back(statesp);
      std::getline(ifile, temp_string);
      std::getline(ifile, temp_string);
      std::getline(ifile, temp_string);
    }
    dqn::Transition t;
    int game_status = 0;
    while (!ifile.eof() && !memory_full) {
      ifile >> temp_int >> temp_int >> temp_string >> temp_string;
      if (temp_string == "GameStatus") {
        ifile >> game_status;
        // TODO find a way not to load data with ball catched
        if (game_status == 1 || game_status == 2 ||
            game_status == 3 || game_status == 4) {
          std::getline(ifile, temp_string);
          std::getline(ifile, temp_string);
          std::getline(ifile, temp_string);
          std::getline(ifile, temp_string);
          break;
        }
      }
      else if (temp_string == "StateFeatures") {
        ActorStateDataSp statesp = std::make_shared<ActorStateData>();
        // std::shared_ptr<ActorStateData> statesp(new dqn::ActorStateData);
        for (int i = 0; i < kStateDataSize; ++i) {
          ifile >> (*statesp)[i];
        }
        past_states.push_back(statesp);
        while (past_states.size() > kStateInputCount) {
          past_states.pop_front();
        }
        ActorInputStates& states = std::get<0>(t);
        std::copy(past_states.begin(), past_states.end(), states.begin());
      }
      else if (temp_string == "player_agent.cpp:") {
        ifile >> temp_string;
        std::vector<std::string> strs;
        boost::split(strs,temp_string,boost::is_any_of("(,)"));
        if (strs[0] == "Dash") {
          std::get<1>(t) = {DASH, std::stof(strs[1]), std::stof(strs[2])};
        }
        else if (strs[0] == "Turn") {
          std::get<1>(t) = {TURN, std::stof(strs[1])};
        }
        else if (strs[0] == "Tackle") {
          std::get<1>(t) = {TACKLE, std::stof(strs[1])};
        }
        else if (strs[0] == "Kick") {
          std::get<1>(t) = {KICK, std::stof(strs[1]), std::stof(strs[2])};
        }
        replay_memory_.push_back(t);
        if (replay_memory_.size() % 100000 == 0) {
          LOG(INFO) << "Parse " << replay_memory_.size() << " transitions.";
        }
        if (replay_memory_.size() == replay_memory_capacity_) {
          memory_full = true;
        }
      }
      //      LOG(INFO) << temp_string;
    }
    //   LOG(INFO) << replay_memory_.size();
  }

  //  LOG(INFO) << replay_memory_.size();

  // for (int i = 0; i < replay_memory_.size(); ++i) {
  //   LOG(INFO) << "\nTransition: " << i;
  //   LOG(INFO) << "PastStates:";
  //   std::string state = "";
  //   std::string action;
  //   for (int j = 0; j < kStateInputCount; ++j) {
  //     LOG(INFO) << "Frame: " << j;
  //     for (int k = 0; k < kStateDataSize; ++k) {
  //       state.append(std::to_string((*(std::get<0>(replay_memory_[i])[j]))[k]));
  //       state.append(" ");
  //     }
  //     LOG(INFO) << state;
  //     state = "";
  //     LOG(INFO) << "Action:";
  //     switch ( std::get<1>(replay_memory_[i]).action) {
  //       case DASH :
  //         LOG(INFO) << "DASH " << std::get<1>(replay_memory_[i]).arg1 << " "
  //                   << std::get<1>(replay_memory_[i]).arg2;
  //         break;
  //       case TURN :
  //         LOG(INFO) << "TURN " << std::get<1>(replay_memory_[i]).arg1;
  //         break;
  //       case TACKLE :
  //         LOG(INFO) << "TACKLE " << std::get<1>(replay_memory_[i]).arg1;
  //         break;
  //       case KICK :
  //         LOG(INFO) << "KICK " << std::get<1>(replay_memory_[i]).arg1 << " "
  //                   << std::get<1>(replay_memory_[i]).arg2;
  //         break;
  //     }
  //   }
  // }
}

}
