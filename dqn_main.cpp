#include <cmath>
#include <iostream>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "prettyprint.hpp"
#include "dqn.hpp"
#include <boost/filesystem.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <limits>

using namespace boost::filesystem;

// DQN Parameters
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(save, "", "Prefix for saving snapshots");
DEFINE_string(rom, "", "Atari 2600 ROM file to play");
DEFINE_int32(memory, 400000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, .1, "Value of epsilon after explore iterations.");
DEFINE_double(gamma, .99, "Discount factor of future rewards (0,1]");
DEFINE_int32(clone_freq, 10000, "Frequency (steps) of cloning the target network.");
DEFINE_int32(memory_threshold, 50000, "Number of transitions to start learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_string(save_screen, "", "File prefix in to save frames");
DEFINE_string(save_binary_screen, "", "File prefix in to save binary frames");
DEFINE_string(weights, "", "The pretrained weights load (*.caffemodel).");
DEFINE_string(snapshot, "", "The solver state to load (*.solverstate).");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, .05, "Epsilon value to be used in evaluation mode");
DEFINE_int32(evaluate_freq, 250000, "Frequency (steps) between evaluations");
DEFINE_int32(repeat_games, 32, "Number of games played in evaluation mode");
DEFINE_string(solver, "dqn_solver.prototxt", "Solver parameter file (*.prototxt)");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - (1.0 - FLAGS_epsilon) * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return FLAGS_epsilon;
  }
}

void SaveScreen(const ALEScreen& screen, const ALEInterface& ale,
                const string filename) {
  IntMatrix screen_matrix;
  for (auto row = 0; row < screen.height(); row++) {
    IntVect row_vec;
    for (auto col = 0; col < screen.width(); col++) {
      int pixel = screen.get(row, col);
      row_vec.emplace_back(pixel);
    }
    screen_matrix.emplace_back(row_vec);
  }
  ale.theOSystem->p_export_screen->save_png(&screen_matrix, filename);
}

void SaveInputFrames(const dqn::InputFrames& frames, const string filename) {
  std::ofstream ofs;
  ofs.open(filename, ios::out | ios::binary);
  for (int i = 0; i < dqn::kInputFrameCount; ++i) {
    const dqn::FrameData& frame = *frames[i];
    for (int j = 0; j < dqn::kCroppedFrameDataSize; ++j) {
      ofs.write((char*) &frame[j], sizeof(uint8_t));
    }
  }
  ofs.close();
}

void InitializeALE(ALEInterface& ale, bool display_screen, std::string& rom) {
  ale.set("display_screen", display_screen);
  ale.set("disable_color_averaging", true);
  ale.loadROM(rom);
}

std::mutex mtx;
ActionVect act_to_take;
std::vector<dqn::InputFrames> frames_batch;
std::vector<float> rewards;
std::vector<bool> thread_ready;
std::vector<bool> thread_done;
std::vector<bool> action_ready;
std::vector<double> thread_scores;

/**
 * Main method used by threads. Plays a single game.
 */
void ThreadEvaluate(int id) {
  mtx.lock();
  ALEInterface ale;
  InitializeALE(ale, false, FLAGS_rom);
  mtx.unlock();
  std::deque<dqn::FrameDataSp> past_frames;
  auto total_score = 0;
  auto reward = 0;
  while (!ale.game_over()) {
    const ALEScreen& screen = ale.getScreen();
    const auto current_frame = dqn::PreprocessScreen(screen);
    past_frames.push_back(current_frame);
    if (past_frames.size() < dqn::kInputFrameCount) {
      for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
        total_score += ale.act(PLAYER_A_NOOP);
      }
      continue;
    }
    while (past_frames.size() > dqn::kInputFrameCount) {
      past_frames.pop_front();
    }
    assert(past_frames.size() == dqn::kInputFrameCount);
    assert(frames_batch.size() >= id);
    dqn::InputFrames input_frames;
    std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
    mtx.lock();
    frames_batch[id] = input_frames;
    thread_ready[id] = true;
    rewards[id] = reward;
    mtx.unlock();
    while (!action_ready[id]) {
      std::this_thread::yield();
    }
    auto immediate_score = 0.0;
    for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
      immediate_score += ale.act(act_to_take[id]);
    }
    total_score += immediate_score;
    reward = immediate_score == 0 ? 0 : immediate_score /
        std::abs(immediate_score);
    assert(reward <= 1 && reward >= -1);
    action_ready[id] = false;
  }
  LOG(INFO) << "Thread " << id << " Score " << total_score;
  mtx.lock();
  thread_done[id] = true;
  thread_ready[id] = true;
  thread_scores[id] = total_score;
  mtx.unlock();
}

/**
 * Plays kMinibatchSize episodes in parallel using threads. Returns a
 * vector of scores for each thread.
 */
std::vector<double> PlayParallelEpisodes(dqn::DQN& dqn, double epsilon,
                                         bool update) {
  assert(FLAGS_repeat_games <= dqn::kMinibatchSize);
  int num_threads = FLAGS_repeat_games;
  frames_batch.resize(num_threads);
  rewards.resize(num_threads);
  act_to_take.resize(num_threads);
  thread_ready.resize(num_threads);
  thread_done.resize(num_threads);
  action_ready.resize(num_threads);
  thread_scores.resize(num_threads);

  std::fill(act_to_take.begin(), act_to_take.end(), PLAYER_A_NOOP);
  std::fill(thread_ready.begin(), thread_ready.end(), false);
  std::fill(thread_done.begin(), thread_done.end(), false);
  std::fill(action_ready.begin(), action_ready.end(), false);
  std::fill(thread_scores.begin(), thread_scores.end(), 0.0);

  std::thread threads[num_threads];
  std::vector<dqn::Transition> games_in_progress[num_threads];
  std::vector<dqn::InputFrames> past_frames_batch;
  for (int i=0; i<num_threads; ++i) {
    threads[i] = std::thread(ThreadEvaluate, i);
  }
  while (std::any_of(thread_done.begin(), thread_done.end(),
                     [](bool done){return !done;})) {
    if (std::all_of(thread_ready.begin(), thread_ready.end(),
                    [](bool ready){return ready;})) {
      if (update) {
        if (past_frames_batch.empty()) {
          past_frames_batch.resize(num_threads);
        } else {
          for (int i=0; i<num_threads; ++i) {
            if (!thread_done[i]) {
              const dqn::FrameDataSp& next_frame =
                  frames_batch[i][dqn::kInputFrameCount-1];
              const auto transition = dqn::Transition(
                  past_frames_batch[i], act_to_take[i], rewards[i], next_frame);
              dqn.AddTransition(transition);
              if (dqn.memory_size() > FLAGS_memory_threshold) {
                dqn.Update();
              }
            }
          }
        }
      }
      ActionVect av = dqn.SelectActions(frames_batch, epsilon);
      assert(av.size() == num_threads);
      for (int i=0; i<num_threads; ++i) {
        act_to_take[i] = av[i];
        if (!thread_done[i]) {
          thread_ready[i] = false;
        }
      }
      if (update) {
        // Swap the past frames with the current frames
        past_frames_batch.swap(frames_batch);
      }
      std::fill(action_ready.begin(), action_ready.end(), true);
    } else {
      std::this_thread::yield();
    }
  }
  for (auto& th: threads) {
    th.join();
  }
  if (update) {
    for (int i=0; i<num_threads; ++i) {
      const auto transition = dqn::Transition(
          frames_batch[i], act_to_take[i], rewards[i], boost::none);
      dqn.AddTransition(transition);
      if (dqn.memory_size() > FLAGS_memory_threshold) {
        dqn.Update();
      }
    }
  }
  return thread_scores;
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(ALEInterface& ale, dqn::DQN& dqn, const double epsilon,
                      const bool update) {
  CHECK(!ale.game_over());
  std::deque<dqn::FrameDataSp> past_frames;
  auto total_score = 0.0;
  for (auto frame = 0; !ale.game_over(); ++frame) {
    const ALEScreen& screen = ale.getScreen();
    if (!FLAGS_save_screen.empty()) {
      std::stringstream ss;
      ss << FLAGS_save_screen << setfill('0') << setw(5) <<
          std::to_string(frame) << ".png";
      SaveScreen(screen, ale, ss.str());
    }
    const auto current_frame = dqn::PreprocessScreen(screen);
    past_frames.push_back(current_frame);
    if (past_frames.size() < dqn::kInputFrameCount) {
      // If there are not past frames enough for DQN input, just select NOOP
      for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
        total_score += ale.act(PLAYER_A_NOOP);
      }
    } else {
      while (past_frames.size() > dqn::kInputFrameCount) {
        past_frames.pop_front();
      }
      dqn::InputFrames input_frames;
      std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
      if (!FLAGS_save_binary_screen.empty()) {
        static int binary_save_num = 0;
        string fname = FLAGS_save_binary_screen +
            std::to_string(binary_save_num++) + ".bin";
        SaveInputFrames(input_frames, fname);
      }
      const auto action = dqn.SelectAction(input_frames, epsilon);
      auto immediate_score = 0.0;
      for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
        immediate_score += ale.act(action);
      }
      total_score += immediate_score;
      // Rewards for DQN are normalized as follows:
      // 1 for any positive score, -1 for any negative score, otherwise 0
      const auto reward = immediate_score == 0 ? 0 : immediate_score /
          std::abs(immediate_score);
      assert(reward <= 1 && reward >= -1);
      if (update) {
        // Add the current transition to replay memory
        const auto transition = ale.game_over() ?
            dqn::Transition(input_frames, action, reward, boost::none) :
            dqn::Transition(input_frames, action, reward,
                            dqn::PreprocessScreen(ale.getScreen()));
        dqn.AddTransition(transition);
        // If the size of replay memory is large enough, update DQN
        if (dqn.memory_size() > FLAGS_memory_threshold) {
          dqn.Update();
        }
      }
    }
  }
  ale.reset_game();
  return total_score;
}

/**
 * Evaluate the current player
 */
double Evaluate(dqn::DQN& dqn) {
  std::vector<double> scores = PlayParallelEpisodes(
      dqn, FLAGS_evaluate_with_epsilon, false);
  double total_score = 0.0;
  for (auto score : scores) {
    total_score += score;
  }
  const auto avg_score = total_score / static_cast<double>(scores.size());
  double stddev = 0.0; // Compute the sample standard deviation
  for (auto i=0; i<scores.size(); ++i) {
    stddev += (scores[i] - avg_score) * (scores[i] - avg_score);
  }
  stddev = sqrt(stddev / static_cast<double>(FLAGS_repeat_games - 1));
  LOG(INFO) << "Evaluation avg_score = " << avg_score << " std = " << stddev;
  return avg_score;
}

int main(int argc, char** argv) {
  std::string usage(argv[0]);
  usage.append(" -rom rom -[evaluate|save path]");
  gflags::SetUsageMessage(usage);
  gflags::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  // google::LogToStderr();

  if (FLAGS_rom.empty()) {
    LOG(ERROR) << "Rom file required but not set.";
    LOG(ERROR) << "Usage: " << gflags::ProgramUsage();
    exit(1);
  }
  path rom_file(FLAGS_rom);
  if (!is_regular_file(rom_file)) {
    LOG(ERROR) << "Invalid ROM file: " << FLAGS_rom;
    exit(1);
  }
  if (!is_regular_file(FLAGS_solver)) {
    LOG(ERROR) << "Invalid solver: " << FLAGS_solver;
    exit(1);
  }
  if (FLAGS_save.empty() && !FLAGS_evaluate) {
    LOG(ERROR) << "Save path (or evaluate) required but not set.";
    LOG(ERROR) << "Usage: " << gflags::ProgramUsage();
    exit(1);
  }
  path save_path(FLAGS_save);
  path snapshot_dir(current_path());
  if (is_directory(save_path)) {
    snapshot_dir = save_path;
    save_path /= rom_file.stem();
  } else {
    if (save_path.has_parent_path()) {
      snapshot_dir = save_path.parent_path();
    }
    save_path += "_";
    save_path += rom_file.stem();
  }
  // Check for files that may be overwritten
  assert(is_directory(snapshot_dir));
  LOG(INFO) << "Snapshots Prefix: " << save_path;
  directory_iterator end;
  for(directory_iterator it(snapshot_dir); it!=end; ++it) {
    if(boost::filesystem::is_regular_file(it->status())) {
      std::string save_path_str = save_path.stem().native();
      std::string other_str = it->path().filename().native();
      auto res = std::mismatch(save_path_str.begin(),
                               save_path_str.end(),
                               other_str.begin());
      if (res.first == save_path_str.end()) {
        LOG(ERROR) << "Existing file " << it->path()
                   << " conflicts with save path " << save_path;
        LOG(ERROR) << "Please remove this file or specify another save path.";
        exit(1);
      }
    }
  }
  // Set the logging destinations
  google::SetLogDestination(google::GLOG_INFO,
                            (save_path.native() + "_INFO_").c_str());
  google::SetLogDestination(google::GLOG_WARNING,
                            (save_path.native() + "_WARNING_").c_str());
  google::SetLogDestination(google::GLOG_ERROR,
                            (save_path.native() + "_ERROR_").c_str());
  google::SetLogDestination(google::GLOG_FATAL,
                            (save_path.native() + "_FATAL_").c_str());

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  ALEInterface ale;
  InitializeALE(ale, FLAGS_gui, FLAGS_rom);

  // Get the vector of legal actions
  const auto legal_actions = ale.getMinimalActionSet();

  CHECK(FLAGS_snapshot.empty() || FLAGS_weights.empty())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  // Construct the solver
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  solver_param.set_snapshot_prefix(save_path.c_str());

  dqn::DQN dqn(legal_actions, solver_param, FLAGS_memory, FLAGS_gamma,
               FLAGS_clone_freq);
  dqn.Initialize();

  if (!FLAGS_save_screen.empty()) {
    LOG(INFO) << "Saving screens to: " << FLAGS_save_screen;
  }

  if (!FLAGS_snapshot.empty()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    dqn.RestoreSolver(FLAGS_snapshot);
  } else if (!FLAGS_weights.empty()) {
    LOG(INFO) << "Finetuning from " << FLAGS_weights;
    dqn.LoadTrainedModel(FLAGS_weights);
  }

  if (FLAGS_evaluate) {
    if (FLAGS_gui) {
      auto score = PlayOneEpisode(ale, dqn, FLAGS_evaluate_with_epsilon, false);
      LOG(INFO) << "Score " << score;
    } else {
      Evaluate(dqn);
    }
    return 0;
  }

  int last_eval_iter = 0;
  int play_batch = 0;
  double best_score = std::numeric_limits<double>::min();
  while (dqn.current_iteration() < solver_param.max_iter()) {
    double epsilon = CalculateEpsilon(dqn.current_iteration());
    double score = PlayOneEpisode(ale, dqn, epsilon, true);
    std::vector<double> scores;
    scores.push_back(score);
    // std::vector<double> scores = PlayParallelEpisodes(dqn, epsilon, true);
    double total_score = 0.0;
    for (auto score : scores) {
      total_score += score;
    }
    const auto avg_score = total_score / static_cast<double>(scores.size());
    LOG(INFO) << "PlayBatch " << play_batch << " avg_score = " << avg_score
              << ", epsilon = " << epsilon
              << ", iter = " << dqn.current_iteration()
              << ", replay_mem_size = " << dqn.memory_size();
    play_batch++;

    if (dqn.current_iteration() >= last_eval_iter + FLAGS_evaluate_freq) {
      double avg_score = Evaluate(dqn);
      if (avg_score > best_score) {
        LOG(INFO) << "iter " << dqn.current_iteration()
                  << " New High Score: " << avg_score;
        best_score = avg_score;
        dqn.Snapshot();
      }
      last_eval_iter = dqn.current_iteration();
    }
  }
  if (dqn.current_iteration() >= last_eval_iter) {
    Evaluate(dqn);
  }
};
