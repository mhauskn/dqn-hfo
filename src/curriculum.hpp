#ifndef CURRICULUM_HPP_
#define CURRICULUM_HPP_

#include "tasks/task.hpp"
#include <mutex>
#include <deque>

class Curriculum {
 public:
  Curriculum(int num_agents);
  ~Curriculum();

  // Returns a curriculum by name
  static Curriculum* getCurriculum(std::string name, int num_agents, unsigned seed);

  // Adds a task by pointer
  void addTask(Task* task);

  // Adds a task by name
  Task& addTask(std::string task_name, int server_port,
                int offense_agents, int defense_agents);

  // Delete a task from the curriculum
  void removeTask(std::string task_name);

  Task& getTask(int tid);

  // Returns a task by name
  Task& getTask(std::string task_name);

  inline std::vector<Task*>& getTasks() { return all_tasks_; }

 protected:
  virtual void queueTasks() = 0;

  std::vector<Task*> all_tasks_;
  std::vector<Task*> current_tasks_;
  std::mutex mutex_;
};

/**
 * RandomCurriculum selects a random task at each step to perform.
 */
class RandomCurriculum : public Curriculum {
 public:
  RandomCurriculum(int num_agents, unsigned seed);

 protected:
  virtual void queueTasks();

  std::mt19937 random_engine_;
};

/**
 * SequentialCurriculum performs one task until mastery before
 * switching to the next.
 */
class SequentialCurriculum : public Curriculum {
 public:
  SequentialCurriculum(int num_agents);

 protected:
  virtual void queueTasks();
  int curr_task_indx_;
  std::deque<float> task_perf_queue_; // Used to average performance of task
  const static int kMaxQueueSize = 1000; // Size of task_perf_queue
  float perf_sum_;
};

#endif
