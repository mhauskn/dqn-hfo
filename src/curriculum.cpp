#include "curriculum.hpp"

using namespace std;
using namespace hfo;

Curriculum::Curriculum(int num_agents) :
    current_(num_agents, NULL)
{
}

Curriculum::~Curriculum() {
  for (Task* t : tasks_) {
    delete t;
  }
}

Task& Curriculum::getTask(std::string task_name) {
  for (Task* t : tasks_) {
    if (t->getName().compare(task_name) == 0) {
      return *t;
    }
  }
  LOG(FATAL) << "Task " << task_name << " not found!";
}

Task& Curriculum::getTask(int tid) {
  CHECK_GT(current_.size(), tid);
  mutex_.lock();
  if (std::all_of(current_.cbegin(), current_.cend(),
                  [](Task* t){ return t == NULL; })) {
    queueTasks();
  }
  Task* task_ptr = current_[tid];
  CHECK_NOTNULL(task_ptr);
  current_[tid] = NULL;
  mutex_.unlock();
  return *task_ptr;
}

void Curriculum::addTask(Task* task) {
  tasks_.push_back(task);
}

Task& Curriculum::addTask(std::string task_name, int server_port,
                          int offense_agents, int defense_agents) {
  Task* task = NULL;
  if (task_name.compare(MoveToBall::taskName()) == 0) {
    task = new MoveToBall(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(KickToGoal::taskName()) == 0) {
    task = new KickToGoal(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Dribble::taskName()) == 0) {
    task = new Dribble(server_port, offense_agents, defense_agents);
  } else {
    LOG(FATAL) << "Task " << task_name << " is not a recognized task!";
  }
  tasks_.push_back(task);
  return *task;
}

void Curriculum::removeTask(std::string task_name) {
  for (vector<Task*>::iterator it = tasks_.begin() ; it != tasks_.end(); ++it) {
    if ((*it)->getName().compare(task_name) == 0) {
      delete *it;
      tasks_.erase(it);
      return;
    }
  }
}

RandomCurriculum::RandomCurriculum(int num_agents, unsigned seed) :
    Curriculum(num_agents),
    random_engine_()
{
  random_engine_.seed(seed);
}

void RandomCurriculum::queueTasks() {
  int indx = std::uniform_int_distribution<int>(0, tasks_.size()-1)(random_engine_);
  for (int i=0; i<current_.size(); ++i) {
    current_[i] = tasks_[indx];
  }
}
