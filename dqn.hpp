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
#include <boost/algorithm/string.hpp>

namespace dqn {

constexpr auto kStateInputCount = 1;
constexpr auto kMinibatchSize = 32;
constexpr auto kActionCount = 4;
constexpr auto kActionparaCount = 6;
constexpr auto kStateDataSize = 58;

constexpr auto kActorInputDataSize = kStateDataSize * kStateInputCount;
constexpr auto kActorMinibatchDataSize = kActorInputDataSize * kMinibatchSize;

using ActorStateData = std::array<float, kStateDataSize>;
using ActorStateDataSp = std::shared_ptr<ActorStateData>;
using ActorInputStates = std::array<ActorStateDataSp, kStateInputCount>;
using Transition = std::tuple<ActorInputStates, Action,
                              float, boost::optional<ActorStateDataSp>>;

using StateLayerInputData = std::array<float, kActorMinibatchDataSize>;
using ActionTargetLayerInputData = std::array<float, kMinibatchSize>;
using ActionparaTargetLayerInputData = std::array<float, kMinibatchSize * kActionparaCount>;
using FilterLayerInputData = std::array<float, kMinibatchSize * kActionparaCount>;

constexpr auto kCriticInputDataSize = kStateDataSize * kStateInputCount
    + kActionCount;
constexpr auto kCriticMinibatchDataSize = kCriticInputDataSize * kMinibatchSize;

// using CriticStateData = std::array<float, kCriticInputDataSize>;
// using CriticStateDataSp = std::shared_ptr<CriticStateData>;
// using CriticInputStates = std::array<CriticStateDataSp, kStateInputCount>;
using CriticStateLayerInputData = std::array<float, kCriticMinibatchDataSize>;

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

  // Snapshot the model/solver/replay memory. Produces files:
  // snapshot_prefix_iter_N.[caffemodel|solverstate|replaymem]. Optionally
  // removes snapshots that share the same prefix but have a lower
  // iteration number.
  void Snapshot(const std::string& snapshot_prefix, bool remove_old=false,
                bool snapshot_memory=true);

  // Select an action by epsilon-greedy.
  Action SelectAction(const ActorInputStates& input_states, double epsilon);

  // Select a batch of actions by epsilon-greedy.
  std::vector<Action> SelectActions(const std::vector<ActorInputStates>& states_batch,
                                 double epsilon);

  // Add a transition to replay memory
  void AddTransition(const Transition& transition);

  // // Update DQN using one minibatch
  // void UpdateCritic();

  // update the actor network
  std::pair<float,float> UpdateActor(int update_idx, bool update,
                                     std::vector<std::pair<int, int>>& accuracy,
                                     std::vector<float>& deviation);

  // evaluate the actor network performance by loss
  void EvaluateActor(int evaluate_idx);

  // Clear the replay memory
  void ClearReplayMemory() { replay_memory_.clear(); }

  // Save the replay memory to a gzipped compressed file
  void SnapshotReplayMemory(const std::string& filename);

  // Load the replay memory from a gzipped compressed file
  void LoadReplayMemory(const std::string& filename);

  // Get the current size of the replay memory
  int memory_size() const { return replay_memory_.size(); }

  // Load mimicing data form a log file
  void LoadMimicData(const std::string& filename);

  // Return the current iteration of the solver
  int current_iteration() const { return critic_solver_->iter(); }

protected:
  // Clone the network and store the result in clone_net_
  void CloneNet(NetSp& net_from, NetSp& net_to);

  // Given a set of input states and a network, select an
  // action. Returns the action and the estimated Q-Value.
  Action SelectActionGreedily(caffe::Net<float>& net,
                             const ActorInputStates& last_states);

  // Given a batch of input states, return a batch of selected actions + values.
  std::vector<Action> SelectActionGreedily(
      caffe::Net<float>& net,
      const std::vector<ActorInputStates>& last_states);

  // // get the Q-Value with the actor parameter
  // std::vector<float> GetQValue(
  //     caffe::Net<float>& net,
  //     std::vector<ActorInputStates>& last_actor_states_batch,
  //     const std::vector<float>& actions);

  // Input data into the State/Target/Filter layers of the given
  // net. This must be done before forward is called.
  void InputDataIntoLayers(caffe::Net<float>& net,
                           float* states_input,
                           float* action_target_input,
                           float* actionpara_target_input,
                           float* filter_input);


protected:
  const std::vector<int> legal_actions_; // TODO : not used
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
  std::mt19937 random_engine;
};

/**
 * Returns a vector of filenames matching a given regular expression.
 */
std::vector<std::string> FilesMatchingRegexp(const std::string& regexp);

/**
 * Removes snapshots matching regexp that have an iteration less than
 * min_iter.
 */
void RemoveSnapshots(const std::string& regexp, int min_iter);

/**
 * Look for the latest snapshot to resume from. Returns a string
 * containing the path to the .solverstate. Returns empty string if
 * none is found. Will only return if the snapshot contains all of:
 * .solverstate,.caffemodel,.replaymemory
 */
void FindLatestSnapshot(const std::string& snapshot_prefix,
                        std::string& actor_snapshot,
                        std::string& critic_snapshot,
                        std::string& memory_snapshot);

/**
 * Look for the best HiScore matching the given snapshot prefix
 */
int FindHiScore(const std::string& snapshot_prefix);

} // namespace dqn

#endif /* DQN_HPP_ */
