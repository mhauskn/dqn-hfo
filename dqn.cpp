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
#include <caffe/layers/memory_data_layer.hpp>

namespace dqn {

using namespace hfo;

DEFINE_int32(seed, 0, "Seed the RNG. Default: time");
DEFINE_double(tau, .001, "Step size for soft updates.");
DEFINE_double(gamma, .99, "Discount factor of future rewards (0,1]");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(memory_threshold, 10000, "Number of transitions required to start learning");
DEFINE_int32(loss_display_iter, 1000, "Frequency of loss display");
DEFINE_int32(snapshot_freq, 10000, "Frequency (steps) snapshots");
DEFINE_bool(remove_old_snapshots, true, "Remove old snapshots when writing more recent ones.");
DEFINE_bool(snapshot_memory, true, "Snapshot the replay memory along with the network.");
DEFINE_bool(advantage_update, false, "Use advantage learning.");
DEFINE_bool(max_advantage_update, false, "Take a max during advantage learning.");
DEFINE_double(alpha, .6, "Advantage learning gain");

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

Action GetAction(const ActorOutput& actor_output) {
  ActorOutput copy(actor_output); // TODO: Remove hack
  copy[TACKLE] = -99999; // Remove tackle action.
  Action action;
  action_t max_act = (action_t) std::distance(copy.begin(), std::max_element(
      copy.begin(), copy.begin() + kActionSize));
  action.action = max_act;
  int arg1_offset = GetParamOffset(max_act, 0); CHECK_GE(arg1_offset, 0);
  action.arg1 = actor_output[kActionSize + arg1_offset];
  int arg2_offset = GetParamOffset(max_act, 1);
  action.arg2 = arg2_offset < 0 ? 0 : actor_output[kActionSize + arg2_offset];
  return action;
}

std::string PrintActorOutput(const ActorOutput& actor_output) {
  return "Dash(" + std::to_string(actor_output[4]) + ", " + std::to_string(actor_output[5]) + ")="
      + std::to_string(actor_output[0]) + ", Turn(" + std::to_string(actor_output[6]) + ")="
      + std::to_string(actor_output[1]) + ", Tackle(" + std::to_string(actor_output[7]) + ")="
      + std::to_string(actor_output[2]) + ", Kick(" + std::to_string(actor_output[8])
      + ", " + std::to_string(actor_output[9]) + ")=" + std::to_string(actor_output[3]);
}
std::string PrintActorOutput(const float* actions, const float* params) {
  return "Dash(" + std::to_string(params[0]) + ", " + std::to_string(params[1]) + ")="
      + std::to_string(actions[0]) + ", Turn(" + std::to_string(params[2]) + ")="
      + std::to_string(actions[1]) + ", Tackle(" + std::to_string(params[3]) + ")="
      + std::to_string(actions[2]) + ", Kick(" + std::to_string(params[4])
      + ", " + std::to_string(params[5]) + ")=" + std::to_string(actions[3]);
}

void PopulateLayer(caffe::LayerParameter& layer,
                   const std::string& name, const std::string& type,
                   const std::vector<std::string>& bottoms,
                   const std::vector<std::string>& tops,
                   const boost::optional<caffe::Phase>& include_phase) {
  layer.set_name(name);
  layer.set_type(type);
  for (auto& bottom : bottoms) {
    layer.add_bottom(bottom);
  }
  for (auto& top : tops) {
    layer.add_top(top);
  }
  // PopulateLayer(layer, name, type, bottoms, tops);
  if (include_phase) {
    layer.add_include()->set_phase(*include_phase);
  }
}
void ConcatLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase,
                 const int& axis) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Concat", bottoms, tops, include_phase);
  caffe::ConcatParameter* concat_param = layer.mutable_concat_param();
  concat_param->set_axis(axis);
}
void SliceLayer(caffe::NetParameter& net_param,
                const std::string& name,
                const std::vector<std::string>& bottoms,
                const std::vector<std::string>& tops,
                const boost::optional<caffe::Phase>& include_phase,
                const int axis,
                const std::vector<int>& slice_points) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Slice", bottoms, tops, include_phase);
  caffe::SliceParameter* slice_param = layer.mutable_slice_param();
  slice_param->set_axis(axis);
  for (auto& p : slice_points) {
    slice_param->add_slice_point(p);
  }
}
void MemoryDataLayer(caffe::NetParameter& net_param,
                     const std::string& name,
                     const std::vector<std::string>& tops,
                     const boost::optional<caffe::Phase>& include_phase,
                     const std::vector<int>& shape) {
  caffe::LayerParameter& memory_layer = *net_param.add_layer();
  PopulateLayer(memory_layer, name, "MemoryData", {}, tops, include_phase);
  CHECK_EQ(shape.size(), 4);
  caffe::MemoryDataParameter* memory_data_param =
      memory_layer.mutable_memory_data_param();
  memory_data_param->set_batch_size(shape[0]);
  memory_data_param->set_channels(shape[1]);
  memory_data_param->set_height(shape[2]);
  memory_data_param->set_width(shape[3]);
}
void SilenceLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Silence", bottoms, tops, include_phase);
}
void ReluLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "ReLU", bottoms, tops, include_phase);
  caffe::ReLUParameter* relu_param = layer.mutable_relu_param();
  relu_param->set_negative_slope(0.01);
}
void TanhLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "TanH", bottoms, tops, include_phase);
}
void EltwiseLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const caffe::EltwiseParameter::EltwiseOp& op) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Eltwise", bottoms, tops, include_phase);
  caffe::EltwiseParameter* eltwise_param = layer.mutable_eltwise_param();
  eltwise_param->set_operation(op);
}
void DummyDataLayer(caffe::NetParameter& net_param,
                    const std::string& name,
                    const std::vector<std::string>& tops,
                    const boost::optional<caffe::Phase>& include_phase,
                    const std::vector<std::vector<float> > shapes,
                    const std::vector<float> values) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "DummyData", {}, tops, include_phase);
  caffe::DummyDataParameter* param = layer.mutable_dummy_data_param();
  for (int i=0; i<values.size(); ++i) {
    caffe::BlobShape* shape = param->add_shape();
    for (int j=0; j<shapes[i].size(); ++j) {
      shape->add_dim(shapes[i][j]);
    }
    caffe::FillerParameter* filler = param->add_data_filler();
    filler->set_type("constant");
    filler->set_value(values[i]);
  }
}
void IPLayer(caffe::NetParameter& net_param,
             const std::string& name,
             const std::vector<std::string>& bottoms,
             const std::vector<std::string>& tops,
             const boost::optional<caffe::Phase>& include_phase,
             const int num_output) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "InnerProduct", bottoms, tops, include_phase);
  caffe::InnerProductParameter* ip_param = layer.mutable_inner_product_param();
  ip_param->set_num_output(num_output);
  caffe::FillerParameter* weight_filler = ip_param->mutable_weight_filler();
  weight_filler->set_type("gaussian");
  weight_filler->set_std(0.01);
  // caffe::FillerParameter* bias_filler = ip_param->mutable_bias_filler();
  // bias_filler->set_type("constant");
  // bias_filler->set_value(1);
}
void EuclideanLossLayer(caffe::NetParameter& net_param,
                        const std::string& name,
                        const std::vector<std::string>& bottoms,
                        const std::vector<std::string>& tops,
                        const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "EuclideanLoss", bottoms, tops, include_phase);
}
caffe::NetParameter CreateActorNet(int state_size) {
  caffe::NetParameter np;
  np.set_name("Actor");
  np.set_force_backward(true);
  MemoryDataLayer(np, state_input_layer_name, {states_blob_name,"dummy1"},
                  boost::none, {kMinibatchSize, kStateInputCount, state_size, 1});
  SilenceLayer(np, "silence", {"dummy1"}, {}, boost::none);
  IPLayer(np, "ip1_layer", {states_blob_name}, {"ip1"}, boost::none, 1024);
  ReluLayer(np, "ip1_relu_layer", {"ip1"}, {"ip1"}, boost::none);
  IPLayer(np, "ip2_layer", {"ip1"}, {"ip2"}, boost::none, 512);
  ReluLayer(np, "ip2_relu_layer", {"ip2"}, {"ip2"}, boost::none);
  IPLayer(np, "ip3_layer", {"ip2"}, {"ip3"}, boost::none, 256);
  ReluLayer(np, "ip3_relu_layer", {"ip3"}, {"ip3"}, boost::none);
  IPLayer(np, "ip4_layer", {"ip3"}, {"ip4"}, boost::none, 128);
  ReluLayer(np, "ip4_relu_layer", {"ip4"}, {"ip4"}, boost::none);
  IPLayer(np, "action_layer", {"ip4"}, {"actions"}, boost::none, 4);
  IPLayer(np, "actionpara_layer", {"ip4"}, {"action_params"}, boost::none, 6);
  return np;
}
caffe::NetParameter CreateCriticNet(int state_size) {
  caffe::NetParameter np;
  np.set_name("Critic");
  np.set_force_backward(true);
  MemoryDataLayer(np, state_input_layer_name, {states_blob_name,"dummy1"},
                  boost::none, {kMinibatchSize, kStateInputCount, state_size, 1});
  MemoryDataLayer(np, action_input_layer_name,
                  {actions_blob_name,"dummy2"},
                  boost::none, {kMinibatchSize, kStateInputCount, kActionSize, 1});
  MemoryDataLayer(np, action_params_input_layer_name,
                  {action_params_blob_name,"dummy3"},
                  boost::none, {kMinibatchSize, kStateInputCount, kActionParamSize, 1});
  MemoryDataLayer(np, target_input_layer_name, {targets_blob_name,"dummy4"},
                  boost::none, {kMinibatchSize, 1, 1, 1});
  SilenceLayer(np, "silence", {"dummy1","dummy2","dummy3","dummy4"}, {}, boost::none);
  ConcatLayer(np, "concat",
              {states_blob_name,actions_blob_name,action_params_blob_name},
              {"state_actions"}, boost::none, 2);
  IPLayer(np, "ip1_layer", {"state_actions"}, {"ip1"}, boost::none, 1024);
  ReluLayer(np, "ip1_relu_layer", {"ip1"}, {"ip1"}, boost::none);
  IPLayer(np, "ip2_layer", {"ip1"}, {"ip2"}, boost::none, 512);
  ReluLayer(np, "ip2_relu_layer", {"ip2"}, {"ip2"}, boost::none);
  IPLayer(np, "ip3_layer", {"ip2"}, {"ip3"}, boost::none, 256);
  ReluLayer(np, "ip3_relu_layer", {"ip3"}, {"ip3"}, boost::none);
  IPLayer(np, "ip4_layer", {"ip3"}, {"ip4"}, boost::none, 128);
  ReluLayer(np, "ip4_relu_layer", {"ip4"}, {"ip4"}, boost::none);
  IPLayer(np, q_values_layer_name, {"ip4"}, {q_values_blob_name}, boost::none, 1);
  EuclideanLossLayer(np, "loss", {q_values_blob_name, targets_blob_name},
                     {loss_blob_name}, boost::none);
  return np;
}


DQN::DQN(caffe::SolverParameter& actor_solver_param,
         caffe::SolverParameter& critic_solver_param,
         std::string save_path, int state_size) :
        actor_solver_param_(actor_solver_param),
        critic_solver_param_(critic_solver_param),
        replay_memory_capacity_(FLAGS_memory),
        gamma_(FLAGS_gamma),
        random_engine(),
        smoothed_critic_loss_(0),
        smoothed_actor_loss_(0),
        last_snapshot_iter_(0),
        save_path_(save_path),
        state_size_(state_size),
  state_input_data_size_(kMinibatchSize * state_size * kStateInputCount) {
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
  caffe::Timer dual_timer;
  dual_timer.Start();
  for (int i=0; i< iterations; ++i) {
    UpdateActorCritic();
  }
  dual_timer.Stop();
  LOG(INFO) << "Average Update: "
            << dual_timer.MilliSeconds()/iterations << " ms.";
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
  CloneNet(actor_net_, actor_target_net_);
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
  CloneNet(actor_net_, actor_target_net_);
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

void DQN::Snapshot() {
  Snapshot(save_path_, FLAGS_remove_old_snapshots, FLAGS_snapshot_memory);
}

void DQN::Snapshot(const std::string& snapshot_prefix,
                   bool remove_old, bool snapshot_memory) {
  using namespace boost::filesystem;
  actor_solver_->Snapshot();
  critic_solver_->Snapshot();
  int actor_iter = actor_solver_->iter();
  std::string actor_fname = save_path_+"_actor_iter_"+std::to_string(actor_iter);
  CHECK(is_regular_file(actor_fname + ".caffemodel"));
  CHECK(is_regular_file(actor_fname + ".solverstate"));
  std::string target_actor_fname = snapshot_prefix+"_actor_iter_"+std::to_string(actor_iter);
  rename(actor_fname + ".caffemodel", target_actor_fname + ".caffemodel");
  rename(actor_fname + ".solverstate", target_actor_fname + ".solverstate");
  int critic_iter = critic_solver_->iter();
  std::string critic_fname = save_path_+"_critic_iter_"+std::to_string(critic_iter);
  CHECK(is_regular_file(critic_fname + ".caffemodel"));
  CHECK(is_regular_file(critic_fname + ".solverstate"));
  std::string target_critic_fname = snapshot_prefix+"_critic_iter_"+std::to_string(critic_iter);
  rename(critic_fname + ".caffemodel", target_critic_fname + ".caffemodel");
  rename(critic_fname + ".solverstate", target_critic_fname + ".solverstate");
  if (snapshot_memory) {
    std::string mem_fname = snapshot_prefix + "_iter_" +
        std::to_string(max_iter()) + ".replaymemory";
    LOG(INFO) << "Snapshotting memory to " << mem_fname;
    SnapshotReplayMemory(mem_fname);
    CHECK(is_regular_file(mem_fname));
  }
  if (remove_old) {
    RemoveSnapshots(snapshot_prefix + "_actor_iter_[0-9]+"
                    "\\.(caffemodel|solverstate)", actor_iter - 1);
    RemoveSnapshots(snapshot_prefix + "_critic_iter_[0-9]+"
                    "\\.(caffemodel|solverstate)", critic_iter - 1);
    RemoveSnapshots(snapshot_prefix + "_iter_[0-9]+\\.replaymemory", critic_iter - 1);
  }
  LOG(INFO) << "Snapshotting Finished!";
}

void DQN::Initialize() {
#ifndef NDEBUG
  actor_solver_param_.set_debug_info(true);
  critic_solver_param_.set_debug_info(true);
#endif
  // Initialize net and solver
  actor_solver_.reset(caffe::SolverRegistry<float>::CreateSolver(actor_solver_param_));
  critic_solver_.reset(caffe::SolverRegistry<float>::CreateSolver(critic_solver_param_));
  actor_net_ = actor_solver_->net();
  critic_net_ = critic_solver_->net();
#ifndef NDEBUG
  actor_net_->set_debug_info(true);
  critic_net_->set_debug_info(true);
#endif
  // Check that nets have the necessary layers and blobs
  HasBlobSize(*actor_net_, states_blob_name,
              {kMinibatchSize, kStateInputCount, state_size_, 1});
  HasBlobSize(*actor_net_, actions_blob_name,
              {kMinibatchSize, kActionSize});
  HasBlobSize(*actor_net_, action_params_blob_name,
              {kMinibatchSize, kActionParamSize});
  HasBlobSize(*critic_net_, states_blob_name,
              {kMinibatchSize, kStateInputCount, state_size_, 1});
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
  CloneNet(actor_net_, actor_target_net_);
}

ActorOutput DQN::GetRandomActorOutput() {
  ActorOutput actor_output;
  for (int i = 0; i < kActionSize; ++i) {
    actor_output[i] = std::uniform_real_distribution<float>(-1.0,1.0)(random_engine);
  }
  actor_output[kActionSize + 0] = // Dash Power
      std::uniform_real_distribution<float>(-100.0, 100.0)(random_engine);
  actor_output[kActionSize + 1] = // Dash Angle
      std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
  actor_output[kActionSize + 2] = // Turn Angle
      std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
  actor_output[kActionSize + 3] = // Tackle Angle
      std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
  actor_output[kActionSize + 4] = // Kick Power
      std::uniform_real_distribution<float>(0.0, 100.0)(random_engine);
  actor_output[kActionSize + 5] = // Kick Angle
      std::uniform_real_distribution<float>(-180.0, 180.0)(random_engine);
  return actor_output;
}

ActorOutput DQN::SelectAction(const InputStates& last_states, const double epsilon) {
  return SelectActions(std::vector<InputStates>{{last_states}}, epsilon)[0];
}

float DQN::EvaluateAction(const InputStates& input_states,
                          const ActorOutput& actor_output) {
  return CriticForward(*critic_net_,
                       std::vector<InputStates>{{input_states}},
                       std::vector<ActorOutput>{{actor_output}})[0];
}

std::vector<ActorOutput>
DQN::SelectActions(const std::vector<InputStates>& states_batch,
                   const double epsilon) {
  CHECK(epsilon >= 0.0 && epsilon <= 1.0);
  CHECK_LE(states_batch.size(), kMinibatchSize);
  if (std::uniform_real_distribution<double>(0.0, 1.0)(random_engine) < epsilon) {
    // Select randomly
    std::vector<ActorOutput> actor_outputs(states_batch.size());
    for (int i = 0; i < actor_outputs.size(); ++i) {
      actor_outputs[i] = GetRandomActorOutput();
    }
    return actor_outputs;
  } else {
    // Select greedily
    return SelectActionGreedily(*actor_net_, states_batch);
  }
}

ActorOutput DQN::SelectActionGreedily(caffe::Net<float>& actor,
                                      const InputStates& last_states) {
  return SelectActionGreedily(
      actor, std::vector<InputStates>{{last_states}}).front();
}

std::vector<ActorOutput>
DQN::SelectActionGreedily(caffe::Net<float>& actor,
                          const std::vector<InputStates>& states_batch) {
  DLOG(INFO) << "  [Forward] Actor";
  CHECK(actor.has_blob(actions_blob_name));
  CHECK(actor.has_blob(action_params_blob_name));
  CHECK_LE(states_batch.size(), kMinibatchSize);
  std::vector<float> states_input(state_input_data_size_, 0.0f);
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
  std::vector<ActorOutput> actor_outputs(states_batch.size());
  const auto actions_blob = actor.blob_by_name(actions_blob_name);
  const auto action_params_blob = actor.blob_by_name(action_params_blob_name);
  for (int n = 0; n < states_batch.size(); ++n) {
    ActorOutput actor_output;
    // TODO: Optimize these copies
    for (int c = 0; c < kActionSize; ++c) {
      actor_output[c] = actions_blob->data_at(n,c,0,0);
    }
    for (int c = 0; c < kActionParamSize; ++c) {
      actor_output[kActionSize + c] = action_params_blob->data_at(n,c,0,0);
    }
    actor_outputs[n] = actor_output;
  }
  return actor_outputs;
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
  std::pair<float,float> res = UpdateActorCritic();
  float critic_loss = res.first;
  float avg_q = res.second;
  if (critic_iter() % FLAGS_loss_display_iter == 0) {
    LOG(INFO) << "Critic Iteration " << critic_iter()
              << ", loss = " << smoothed_critic_loss_;
    smoothed_critic_loss_ = 0;
  }
  smoothed_critic_loss_ += critic_loss / float(FLAGS_loss_display_iter);
  if (actor_iter() % FLAGS_loss_display_iter == 0) {
    LOG(INFO) << "Actor Iteration " << actor_iter()
              << ", avg_q_value = " << smoothed_actor_loss_;
    smoothed_actor_loss_ = 0;
  }
  smoothed_actor_loss_ += avg_q / float(FLAGS_loss_display_iter);
  bool critic_needs_snapshot =
      critic_iter() >= last_snapshot_iter_ + FLAGS_snapshot_freq;
  bool actor_needs_snapshot =
      actor_iter() >= last_snapshot_iter_ + FLAGS_snapshot_freq;
  if (critic_needs_snapshot || actor_needs_snapshot) {
    Snapshot();
    last_snapshot_iter_ = max_iter();
  }
}

std::pair<float,float> DQN::UpdateActorCritic() {
  CHECK(critic_net_->has_blob(states_blob_name));
  CHECK(critic_net_->has_blob(actions_blob_name));
  CHECK(critic_net_->has_blob(action_params_blob_name));
  CHECK(critic_net_->has_blob(targets_blob_name));
  CHECK(critic_net_->has_blob(loss_blob_name));
  const auto actor_states_blob = actor_net_->blob_by_name(states_blob_name);
  const auto actor_actions_blob = actor_net_->blob_by_name(actions_blob_name);
  const auto actor_action_params_blob = actor_net_->
      blob_by_name(action_params_blob_name);
  const auto critic_states_blob = critic_net_->blob_by_name(states_blob_name);
  const auto critic_action_blob = critic_net_->blob_by_name(actions_blob_name);
  const auto critic_action_params_blob =
      critic_net_->blob_by_name(action_params_blob_name);
  const auto target_blob = critic_net_->blob_by_name(targets_blob_name);
  const auto q_values_blob = critic_net_->blob_by_name(q_values_blob_name);
  const auto loss_blob = critic_net_->blob_by_name(loss_blob_name);
  // Collect a batch of next-states used to generate target_q_values
  std::vector<int> transitions = SampleTransitionsFromMemory(kMinibatchSize);
  std::vector<InputStates> states_batch(kMinibatchSize);
  std::vector<ActorOutput> actions_batch(kMinibatchSize);
  std::vector<float> rewards_batch(kMinibatchSize);
  std::vector<bool> terminal(kMinibatchSize);
  std::vector<InputStates> next_states_batch;
  next_states_batch.reserve(kMinibatchSize);
  // Raw data used for input to networks
  std::vector<float> states_input(state_input_data_size_, 0.0f);
  std::vector<float> action_input(kActionInputDataSize, 0.0f);
  std::vector<float> action_params_input(kActionParamsInputDataSize, 0.0f);
  std::vector<float> target_input(kTargetInputDataSize, 0.0f);
  for (int n = 0; n < kMinibatchSize; ++n) {
    const auto& transition = replay_memory_[transitions[n]];
    InputStates last_states;
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = std::get<0>(transition)[c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + critic_states_blob->offset(n,c,0,0));
      last_states[c] = state_data;
    }
    states_batch[n] = last_states;
    const ActorOutput& actor_output = std::get<1>(transition);
    std::copy(actor_output.begin(), actor_output.begin() + kActionSize,
              action_input.begin() + critic_action_blob->offset(n,0,0,0));
    std::copy(actor_output.begin() + kActionSize, actor_output.end(),
              action_params_input.begin() + critic_action_params_blob->offset(n,0,0,0));
    actions_batch[n] = actor_output;
    const float reward = std::get<2>(transition);
    rewards_batch[n] = reward;
    terminal[n] = !std::get<3>(transition);
    if (!terminal[n]) {
      InputStates next_states;
      for (int i = 0; i < kStateInputCount - 1; ++i) {
        next_states[i] = std::get<0>(transition)[i + 1];
      }
      next_states[kStateInputCount-1] = std::get<3>(transition).get();
      next_states_batch.push_back(next_states);
    }
  }
  std::vector<float> advantage(kMinibatchSize, 0.0f);
  if (FLAGS_advantage_update) {
    // Q(s,a) - Q-Values for the actions taken in the reply memory
    const std::vector<float> q_s_a =
        CriticForward(*critic_target_net_, states_batch, actions_batch);
    // Q(s,a*) - Q-values for the actions the actor would currently take
    const std::vector<float> q_s_aPrime =
        CriticForwardThroughActor(
            *critic_target_net_, *actor_target_net_, states_batch);
    for (int n = 0; n < kMinibatchSize; ++n) {
      if (FLAGS_max_advantage_update) {
        advantage[n] = std::max(0.f, q_s_a[n] - q_s_aPrime[n]);
      } else {
        advantage[n] = q_s_a[n] - q_s_aPrime[n];
      }
    }
  }
  // Generate targets using the target nets
  const std::vector<float> target_q_values =
      CriticForwardThroughActor(
          *critic_target_net_, *actor_target_net_, next_states_batch);
  int target_value_idx = 0;
  for (int n = 0; n < kMinibatchSize; ++n) {
    float target = terminal[n] ? rewards_batch[n] :
        rewards_batch[n] + gamma_ * target_q_values[target_value_idx++];
    if (FLAGS_advantage_update) {
      target -= FLAGS_alpha * advantage[n];
    }
    CHECK(std::isfinite(target)) << "Target not finite!";
    target_input[target_blob->offset(n,0,0,0)] = target;
  }
  InputDataIntoLayers(*critic_net_, states_input.data(), action_input.data(),
                      action_params_input.data(), target_input.data(), NULL);
  DLOG(INFO) << " [Step] Critic";
  critic_solver_->Step(1);
  float critic_loss = loss_blob->data_at(0,0,0,0);
  CHECK(std::isfinite(critic_loss)) << "Critic loss not finite!";
  // Update the actor
  ZeroGradParameters(*critic_net_);
  ZeroGradParameters(*actor_net_);
  std::vector<ActorOutput> actor_output_batch =
      SelectActionGreedily(*actor_net_, states_batch);
  DLOG(INFO) << "ActorOutput:  " << PrintActorOutput(actor_output_batch[0]);
  std::vector<float> q_values =
      CriticForward(*critic_net_, states_batch, actor_output_batch);
  float avg_q = std::accumulate(q_values.begin(), q_values.end(), 0.0) /
      float(q_values.size());
  // Set the critic diff and run backward
  float* q_values_diff = q_values_blob->mutable_cpu_diff();
  for (int n = 0; n < kMinibatchSize; n++) {
    q_values_diff[q_values_blob->offset(n,0,0,0)] = -1.0;
  }
  DLOG(INFO) << " [Backwards] " << critic_net_->name();
  critic_net_->BackwardFrom(GetLayerIndex(*critic_net_, q_values_layer_name));
  float* action_diff = critic_action_blob->mutable_cpu_diff();
  float* param_diff = critic_action_params_blob->mutable_cpu_diff();
  DLOG(INFO) << "Diff: " << PrintActorOutput(action_diff, param_diff);
  for (int n = 0; n < kMinibatchSize; ++n) {
    for (int h = 0; h < kActionSize; ++h) {
      int offset = critic_action_blob->offset(n,0,h,0);
      float diff = action_diff[offset];
      float output = actor_output_batch[n][h];
      float min = -1.0; float max = 1.0;
      if (diff < 0) {
        diff *= (max - output) / (max - min);
      } else if (diff > 0) {
        diff *= (output - min) / (max - min);
      }
      action_diff[offset] = diff;
    }
    for (int h = 0; h < kActionParamSize; ++h) {
      int offset = critic_action_params_blob->offset(n,0,h,0);
      float diff = param_diff[offset];
      float output = actor_output_batch[n][h+kActionSize];
      float min, max;
      if (h == 0 || h == 4) {
        min = 0; max = 100;
      } else if (h == 1 || h == 2 || h == 3 || h == 5) {
        min = -180; max = 180;
      }
      if (diff < 0) {
        diff *= (max - output) / (max - min);
      } else if (diff > 0) {
        diff *= (output - min) / (max - min);
      }
      param_diff[offset] = diff;
    }
  }
  DLOG(INFO) << "Diff2 " << PrintActorOutput(action_diff, param_diff);
  // Transfer input-level diffs from Critic to Actor
  actor_actions_blob->ShareDiff(*critic_action_blob);
  actor_action_params_blob->ShareDiff(*critic_action_params_blob);
  DLOG(INFO) << " [Backwards] " << actor_net_->name();
  actor_net_->BackwardFrom(GetLayerIndex(*actor_net_, "actionpara_layer"));
  actor_solver_->ApplyUpdate();
  actor_solver_->set_iter(actor_solver_->iter() + 1);
  // Soft update the target networks
  SoftUpdateNet(critic_net_, critic_target_net_, FLAGS_tau);
  SoftUpdateNet(actor_net_, actor_target_net_, FLAGS_tau);
  return std::make_pair(critic_loss, avg_q);
}

std::vector<float> DQN::CriticForwardThroughActor(
    caffe::Net<float>& critic, caffe::Net<float>& actor,
    const std::vector<InputStates>& states_batch) {
  DLOG(INFO) << " [Forward] " << critic.name() << " Through " << actor_net_->name();
  return CriticForward(critic, states_batch,
                       SelectActionGreedily(actor, states_batch));
}

std::vector<float> DQN::CriticForward(caffe::Net<float>& critic,
                                      const std::vector<InputStates>& states_batch,
                                      const std::vector<ActorOutput>& action_batch) {
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
  std::vector<float> states_input(state_input_data_size_, 0.0f);
  std::vector<float> action_input(kActionInputDataSize, 0.0f);
  std::vector<float> action_params_input(kActionParamsInputDataSize, 0.0f);
  std::vector<float> target_input(kTargetInputDataSize, 0.0f);
  for (int n = 0; n < states_batch.size(); ++n) {
    for (int c = 0; c < kStateInputCount; ++c) {
      const auto& state_data = states_batch[n][c];
      std::copy(state_data->begin(), state_data->end(),
                states_input.begin() + states_blob->offset(n,c,0,0));
    }
    const ActorOutput& actor_output = action_batch[n];
    std::copy(actor_output.begin(), actor_output.begin() + kActionSize,
              action_input.begin() + actions_blob->offset(n,0,0,0));
    std::copy(actor_output.begin() + kActionSize, actor_output.end(),
              action_params_input.begin() + action_params_blob->offset(n,0,0,0));
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

void DQN::SoftUpdateNet(NetSp& net_from, NetSp& net_to, float tau) {
  const auto& from_params = net_from->params();
  const auto& to_params = net_to->params();
  CHECK_EQ(from_params.size(), to_params.size());
  for (int i = 0; i < from_params.size(); ++i) {
    auto& from_blob = from_params[i];
    auto& to_blob = to_params[i];
    caffe::caffe_cpu_axpby(from_blob->count(), tau, from_blob->cpu_data(),
                           (1-tau), to_blob->mutable_cpu_data());
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
        out.write((char*)state->data(), state_size_ * sizeof(float));
      }
    }
    const StateDataSp curr_state = states.back();
    out.write((char*)curr_state->data(), state_size_ * sizeof(float));
    const ActorOutput& actor_output = std::get<1>(t);
    out.write((char*)&actor_output, sizeof(ActorOutput));
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
        StateDataSp state = std::make_shared<StateData>(state_size_);
        in.read((char*)state->data(), state_size_ * sizeof(float));
        past_states.push_back(state);
      }
    }
    StateDataSp state = std::make_shared<StateData>(state_size_);
    in.read((char*)state->data(), state_size_ * sizeof(float));
    past_states.push_back(state);
    while (past_states.size() > kStateInputCount) {
      past_states.pop_front();
    }
    CHECK_EQ(past_states.size(), kStateInputCount);
    InputStates& states = std::get<0>(t);
    std::copy(past_states.begin(), past_states.end(), states.begin());
    in.read((char*)&std::get<1>(t), sizeof(ActorOutput));
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

} // namespace dqn
