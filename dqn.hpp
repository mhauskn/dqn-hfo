#ifndef DQN_HPP_
#define DQN_HPP_

#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <ale_interface.hpp>
#include <caffe/caffe.hpp>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>

namespace dqn {

constexpr auto kRawFrameHeight = 210;
constexpr auto kRawFrameWidth = 160;
constexpr auto kCroppedFrameSize = 84;
constexpr auto kCroppedFrameDataSize = kCroppedFrameSize * kCroppedFrameSize;
constexpr auto kInputFrameCount = 2;
constexpr auto kInputDataSize = kCroppedFrameDataSize * kInputFrameCount;
constexpr auto kMinibatchSize = 32;
constexpr auto kMinibatchDataSize = kInputDataSize * kMinibatchSize;
constexpr auto kOutputCount = 18;

using FrameData = std::array<uint8_t, kCroppedFrameDataSize>;
using FrameDataSp = std::shared_ptr<FrameData>;
using InputFrames = std::array<FrameDataSp, 4>;
using Transition = std::tuple<InputFrames, Action,
                              float, boost::optional<FrameDataSp>>;

using FramesLayerInputData = std::array<float, kMinibatchDataSize>;
using TargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
using FilterLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;

using ActionValue = std::pair<Action, float>;
using SolverSp = std::shared_ptr<caffe::Solver<float>>;
using NetSp = boost::shared_ptr<caffe::Net<float>>;

/**
 * Deep Q-Network
 */
class DQN {
public:
  DQN(const ActionVect& legal_actions,
      const caffe::SolverParameter& solver_param,
      const int replay_memory_capacity,
      const double gamma,
      const int clone_frequency) :
        legal_actions_(legal_actions),
        solver_param_(solver_param),
        replay_memory_capacity_(replay_memory_capacity),
        gamma_(gamma),
        clone_frequency_(clone_frequency),
        random_engine(0) {}

  // Initialize DQN. Must be called before calling any other method.
  void Initialize();

  // Load a trained model from a file.
  void LoadTrainedModel(const std::string& model_file);

  // Restore solving from a solver file.
  void RestoreSolver(const std::string& solver_file);

  // Snapshot the current model
  void Snapshot() { solver_->Snapshot(); }

  // Select an action by epsilon-greedy.
  Action SelectAction(const InputFrames& input_frames, double epsilon);

  // Select a batch of actions by epsilon-greedy.
  ActionVect SelectActions(const std::vector<InputFrames>& frames_batch,
                           double epsilon);

  // Add a transition to replay memory
  void AddTransition(const Transition& transition);

  // Update DQN using one minibatch
  void Update();

  // Clear the replay memory
  void ClearReplayMemory() { replay_memory_.clear(); }

  // Get the current size of the replay memory
  int memory_size() const { return replay_memory_.size(); }

  // Return the current iteration of the solver
  int current_iteration() const { return solver_->iter(); }

protected:
  // Clone the Primary network and store the result in clone_net_
  void ClonePrimaryNet();

  // Given a set of input frames and a network, select an
  // action. Returns the action and the estimated Q-Value.
  ActionValue SelectActionGreedily(caffe::Net<float>& net,
                                   const InputFrames& last_frames);

  // Given a batch of input frames, return a batch of selected actions + values.
  std::vector<ActionValue> SelectActionGreedily(
      caffe::Net<float>& net,
      const std::vector<InputFrames>& last_frames);

  // Input data into the Frames/Target/Filter layers of the given
  // net. This must be done before forward is called.
  void InputDataIntoLayers(caffe::Net<float>& net,
                           const FramesLayerInputData& frames_data,
                           const TargetLayerInputData& target_data,
                           const FilterLayerInputData& filter_data);

protected:
  const ActionVect legal_actions_;
  const caffe::SolverParameter solver_param_;
  const int replay_memory_capacity_;
  const double gamma_;
  const int clone_frequency_; // How often (steps) the clone_net is updated
  std::deque<Transition> replay_memory_;
  SolverSp solver_;
  NetSp net_; // The primary network used for action selection.
  NetSp clone_net_; // Clone of primary net. Used to generate targets.
  TargetLayerInputData dummy_input_data_;
  std::mt19937 random_engine;
};

/**
 * Preprocess an ALE screen (downsampling & grayscaling)
 */
FrameDataSp PreprocessScreen(const ALEScreen& raw_screen);

}

#endif /* DQN_HPP_ */
