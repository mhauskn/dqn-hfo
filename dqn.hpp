#ifndef DQN_HPP_
#define DQN_HPP_

#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <HFO.hpp>
#include <caffe/caffe.hpp>
#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>

namespace dqn {

constexpr auto kStateInputCount = 2;
constexpr auto kMinibatchSize = 32;
constexpr auto kOutputCount = 1;

constexpr auto kActorStateDataSize = 58;
constexpr auto kActorInputDataSize = kActorStateDataSize * kStateInputCount;
constexpr auto kActorMinibatchDataSize = kActorInputDataSize * kMinibatchSize;

using ActorStateData = std::array<float, kActorStateDataSize>;
using ActorStateDataSp = std::shared_ptr<ActorStateData>;
using ActorInputStates = std::array<ActorStateDataSp, kStateInputCount>;
using Transition = std::tuple<ActorInputStates, float,
                              float, boost::optional<ActorStateDataSp>>;

using StateLayerInputData = std::array<float, kActorMinibatchDataSize>;
using TargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
using FilterLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;

constexpr auto kCriticStateDataSize = kActorStateDataSize + kOutputCount;
constexpr auto kCriticInputDataSize = kCriticStateDataSize * kStateInputCount;
constexpr auto kCriticMinibatchDataSize = kCriticInputDataSize * kMinibatchSize;

using CriticStateData = std::array<float, kCriticInputDataSize>;
using CriticStateDataSp = std::shared_ptr<CriticStateData>;
using CriticInputStates = std::array<CriticStateDataSp, kStateInputCount>;

using CriticStateLayerInputData = std::array<float, kCriticMinibatchDataSize>;
using CriticTargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;

using ActionValue = std::pair<float, float>;
using SolverSp = std::shared_ptr<caffe::Solver<float>>;
using NetSp = boost::shared_ptr<caffe::Net<float>>;

/**
 * Deep Q-Network
 */
class DQN {
public:
  DQN(const std::vector<int>& legal_actions,
      const caffe::SolverParameter& actor_solver_param,
      const caffe::SolverParameter& critic_solver_param,
      const int replay_memory_capacity,
      const double gamma,
      const int clone_frequency) :
        legal_actions_(legal_actions),
        actor_solver_param_(actor_solver_param),
        critic_solver_param_(critic_solver_param),
        replay_memory_capacity_(replay_memory_capacity),
        gamma_(gamma),
        clone_frequency_(clone_frequency),
        random_engine(0) {}

  // Initialize DQN. Must be called before calling any other method.
  void Initialize();

  // Load a trained model from a file.
  void LoadTrainedModel(const std::string& actor_model_file,
                        const std::string& critic_model_file);

  // Restore solving from a solver file.
  void RestoreSolver(const std::string& actor_solver_file,
                     const std::string& critic_solver_bin);

  // Snapshot the current model
  void Snapshot() { actor_solver_->Snapshot(); critic_solver_->Snapshot(); }

  // Select an action by epsilon-greedy.
  float SelectAction(const ActorInputStates& input_states, double epsilon);

  // Select a batch of actions by epsilon-greedy.
  std::vector<float> SelectActions(const std::vector<ActorInputStates>& states_batch,
                                 double epsilon);

  // Add a transition to replay memory
  void AddTransition(const Transition& transition);

  // Update DQN using one minibatch
  void UpdateCritic();

  // Clear the replay memory
  void ClearReplayMemory() { replay_memory_.clear(); }

  // Get the current size of the replay memory
  int memory_size() const { return replay_memory_.size(); }

  // Return the current iteration of the solver
  int current_iteration() const { return actor_solver_->iter(); }

protected:
  // Clone the network and store the result in clone_net_
  void CloneNet(NetSp& net_from, NetSp& net_to);

  // Given a set of input states and a network, select an
  // action. Returns the action and the estimated Q-Value.
  float SelectActionGreedily(caffe::Net<float>& net,
                             const ActorInputStates& last_states);

  // Given a batch of input states, return a batch of selected actions + values.
  std::vector<float> SelectActionGreedily(
      caffe::Net<float>& net,
      const std::vector<ActorInputStates>& last_states);

  // get the Q-value from the critic network
  std::vector<float> GetQValue(caffe::Net<float> net,
                               std::vector<CriticInputStates> last_critic_states_batch);

  // Input data into the State/Target/Filter layers of the given
  // net. This must be done before forward is called.
  void InputDataIntoLayers(caffe::Net<float>& net,
                           float* states_input,
                           float* target_input,
                           float* filter_input);


protected:
  const std::vector<int> legal_actions_;
  const caffe::SolverParameter actor_solver_param_;
  const caffe::SolverParameter critic_solver_param_;
  const int replay_memory_capacity_;
  const double gamma_;
  const int clone_frequency_; // How often (steps) the clone_net is updated
  std::deque<Transition> replay_memory_;
  SolverSp actor_solver_;
  NetSp actor_net_; // The actor network used for continuous action evaluation.
  SolverSp critic_solver_;
  NetSp critic_net_;  // The critic network used for giving q-value of a continuous action;
  NetSp critic_target_net_; // Clone of critic net. Used to generate targets.
  TargetLayerInputData dummy_input_data_;
  std::mt19937 random_engine;
};

} // namespace dqn

#endif /* DQN_HPP_ */
