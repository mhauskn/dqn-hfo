#ifndef CURRICULUM_HPP_
#define CURRICULUM_HPP_

#include "tasks/task.hpp"
#include <mutex>

class Curriculum {
 public:
  Curriculum(int num_agents);
  ~Curriculum();

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

  inline std::vector<Task*>& getTasks() { return tasks_; }

 protected:
  virtual void queueTasks() = 0;

  std::vector<Task*> tasks_;
  std::vector<Task*> current_;
  std::mutex mutex_;
};

class RandomCurriculum : public Curriculum {
 public:
  RandomCurriculum(int num_agents, unsigned seed);

 protected:
  virtual void queueTasks();

  std::mt19937 random_engine_;
};

#endif
