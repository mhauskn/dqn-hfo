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

constexpr auto kStateInputCount = 1;
constexpr auto kMinibatchSize = 32;
constexpr auto kActionSize = 4;
constexpr auto kActionParamSize = 6;
constexpr auto kStateSize = 58;

constexpr auto kStateInputDataSize = kMinibatchSize * kStateSize * kStateInputCount;
constexpr auto kActionInputDataSize = kMinibatchSize * kActionSize;
constexpr auto kActionParamsInputDataSize = kMinibatchSize * kActionParamSize;
constexpr auto kTargetInputDataSize = kMinibatchSize * kActionSize;
constexpr auto kFilterInputDataSize = kMinibatchSize * kActionSize;

using ActorOutput = std::array<float, kActionSize + kActionParamSize>;
using StateData   = std::array<float, kStateSize>;
using StateDataSp = std::shared_ptr<StateData>;
using InputStates = std::array<StateDataSp, kStateInputCount>;
using Transition  = std::tuple<InputStates, ActorOutput,
                               float, boost::optional<StateDataSp>>;
using SolverSp    = std::shared_ptr<caffe::Solver<float>>;
using NetSp       = boost::shared_ptr<caffe::Net<float>>;

// Layer Names
constexpr auto state_input_layer_name         = "state_input_layer";
constexpr auto action_input_layer_name        = "action_input_layer";
constexpr auto action_params_input_layer_name = "action_params_input_layer";
constexpr auto target_input_layer_name        = "target_input_layer";
constexpr auto filter_input_layer_name        = "filter_input_layer";
constexpr auto q_values_layer_name            = "q_values_layer";
// Blob names
constexpr auto states_blob_name        = "states";
constexpr auto actions_blob_name       = "actions";
constexpr auto action_params_blob_name = "action_params";
constexpr auto targets_blob_name       = "target";
constexpr auto filter_blob_name        = "filter";
constexpr auto q_values_blob_name      = "q_values";
constexpr auto loss_blob_name          = "loss";

/**
 * Deep Q-Network
 */
class DQN {
public:
  DQN(caffe::SolverParameter& actor_solver_param,
      caffe::SolverParameter& critic_solver_param);

  // Benchmark the speed of updates
  void Benchmark(int iterations=1000);

  // Loading methods
  void RestoreActorSolver(const std::string& actor_solver);
  void RestoreCriticSolver(const std::string& critic_solver);
  void LoadActorWeights(const std::string& actor_model_file);
  void LoadCriticWeights(const std::string& critic_weights);
  void LoadReplayMemory(const std::string& filename);

  // Snapshot the model/solver/replay memory. Produces files:
  // snapshot_prefix_iter_N.[caffemodel|solverstate|replaymem]. Optionally
  // removes snapshots with same prefix but lower iteration.
  void Snapshot(const std::string& snapshot_prefix, bool remove_old=false,
                bool snapshot_memory=true);

  // Generates a HFO action uniformly at random
  hfo::Action GetRandomHFOAction();
  ActorOutput GetRandomActorOutput();

  // Select an action using epsilon-greedy action selection.
  ActorOutput SelectAction(const InputStates& input_states, double epsilon);

  // Select a batch of actions using epsilon-greedy action selection.
  std::vector<ActorOutput> SelectActions(const std::vector<InputStates>& states_batch,
                                         double epsilon);

  // Warps an action according to the gradients provided by the
  // critic. The degree of warping is determined by a gain sampled
  // uniformly from the range [min_gain, max_gain].
  ActorOutput WarpAction(const InputStates& input_states, const ActorOutput& action,
                         float min_gain=0, float max_gain=100);

  // Evaluate a state-action, returning the q-value.
  float EvaluateAction(const InputStates& input_states, const ActorOutput& action);

  // Assess the critic's (ignorant) optimism. Returns the probability
  // that a random action is better than the actor's suggested action
  // for the provided states
  float AssessOptimism(int n_states=32, int n_samples_per_state=32);
  float AssessOptimism(const InputStates& input_states, int n_samples=32);
  std::vector<float> AssessOptimism(const std::vector<InputStates>& states_batch,
                                    int n_samples=32);

  // Add a transition to replay memory
  void AddTransition(const Transition& transition);

  // Update the model(s)
  void Update();

  // Update DQN using one minibatch. Returns the loss.
  float UpdateCritic();

  // Updates the actor against the critic_net_. Returns diff from critic.
  float UpdateActor();

  // Compare the q_values in slow and fast dash
  void EvaluateCritic();

  // Clear the replay memory
  void ClearReplayMemory() { replay_memory_.clear(); }

  // Save the replay memory to a gzipped compressed file
  void SnapshotReplayMemory(const std::string& filename);

  // Get the current size of the replay memory
  int memory_size() const { return replay_memory_.size(); }

  // Return the current iteration of the solvers
  int max_iter() const { return std::max(actor_iter(), critic_iter()); }
  int critic_iter() const { return critic_solver_->iter(); }
  int actor_iter() const { return actor_solver_->iter(); }

protected:
  // Initialize DQN. Called by the constructor
  void Initialize();

  // Update the actor network from the gradients provided by the critic
  float UpdateActor(caffe::Net<float>& critic);

  // Randomly sample the replay memory n-times, returning transition indexes
  std::vector<int> SampleTransitionsFromMemory(int n, int idx_l, int idx_h);
  // Randomly sample the replay memory n-times returning input_states
  std::vector<InputStates> SampleStatesFromMemory(int n);

  // Clone the network and store the result in clone_net_
  void CloneNet(NetSp& net_from, NetSp& net_to);

  // Given input states, use the actor network to select an action.
  ActorOutput SelectActionGreedily(caffe::Net<float>& actor,
                                   const InputStates& last_states);

  // Given a batch of input states, return a batch of selected actions.
  std::vector<ActorOutput> SelectActionGreedily(
      caffe::Net<float>& actor,
      const std::vector<InputStates>& states_batch);

  // Runs forward on critic to produce q-values. Actions inferred by actor.
  std::vector<float> CriticForwardThroughActor(
      caffe::Net<float>& critic, const std::vector<InputStates>& states_batch);

  // Runs forward on critic to produce q-values.
  std::vector<float> CriticForward(caffe::Net<float>& critic,
                                   const std::vector<InputStates>& states_batch,
                                   const std::vector<ActorOutput>& action_batch);

  // Input data into the State/Target/Filter layers of the given
  // net. This must be done before forward is called.
  void InputDataIntoLayers(caffe::Net<float>& net,
                           float* states_input,
                           float* actions_input,
                           float* action_params_input,
                           float* target_input,
                           float* filter_input);

 protected:
  caffe::SolverParameter actor_solver_param_;
  caffe::SolverParameter critic_solver_param_;
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
  float smoothed_critic_loss_, smoothed_actor_loss_;
};

/**
 * Converts an ActorOutput into an action
 */
hfo::Action GetAction(const ActorOutput& actor_output);

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
