#include "curriculum.hpp"

using namespace std;
using namespace hfo;

DEFINE_double(performance_threshold, .8, "Threshold for Sequential Curriculum");

Curriculum* Curriculum::getCurriculum(std::string name, int num_agents, unsigned seed) {
  if (name.compare("random") == 0) {
    return new RandomCurriculum(num_agents, seed);
  } else if (name.compare("sequential") == 0) {
    return new SequentialCurriculum(num_agents);
  } else {
    LOG(FATAL) << "Unrecognized Curriculum Requested: " << name;
  }
}

Curriculum::Curriculum(int num_agents) :
    current_tasks_(num_agents, NULL)
{
}

Curriculum::~Curriculum() {
  for (Task* t : all_tasks_) {
    delete t;
  }
}

Task& Curriculum::getTask(std::string task_name) {
  for (Task* t : all_tasks_) {
    if (t->getName().compare(task_name) == 0) {
      return *t;
    }
  }
  LOG(FATAL) << "Task " << task_name << " not found!";
}

Task& Curriculum::getTask(int tid) {
  CHECK_GT(current_tasks_.size(), tid);
  mutex_.lock();
  if (std::all_of(current_tasks_.cbegin(), current_tasks_.cend(),
                  [](Task* t){ return t == NULL; })) {
    queueTasks();
  }
  Task* task_ptr = current_tasks_[tid];
  CHECK_NOTNULL(task_ptr);
  current_tasks_[tid] = NULL;
  mutex_.unlock();
  return *task_ptr;
}

void Curriculum::addTask(Task* task) {
  int task_id = all_tasks_.size();
  task->setID(task_id);
  all_tasks_.push_back(task);
}

Task& Curriculum::addTask(std::string task_name, int server_port,
                          int offense_agents, int defense_agents) {
  Task* task = NULL;
  if (task_name.compare(MoveToBall::taskName()) == 0) {
    task = new MoveToBall(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(MoveAwayFromBall::taskName()) == 0) {
    task = new MoveAwayFromBall(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(KickToGoal::taskName()) == 0) {
    task = new KickToGoal(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Soccer::taskName()) == 0) {
    task = new Soccer(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Soccer1v1::taskName()) == 0) {
    task = new Soccer1v1(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Soccer2v1::taskName()) == 0) {
    task = new Soccer2v1(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(SoccerEasy::taskName()) == 0) {
    task = new SoccerEasy(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Dribble::taskName()) == 0) {
    task = new Dribble(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Pass::taskName()) == 0) {
    task = new Pass(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(KickToTeammate::taskName()) == 0) {
    task = new KickToTeammate(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Cross::taskName()) == 0) {
    task = new Cross(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(MirrorActions::taskName()) == 0) {
    task = new MirrorActions(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(SayMyTid::taskName()) == 0) {
    task = new SayMyTid(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(Keepaway::taskName()) == 0) {
    task = new Keepaway(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(BlindSoccer::taskName()) == 0) {
    task = new BlindSoccer(server_port, offense_agents, defense_agents);
  } else if (task_name.compare(BlindMoveToBall::taskName()) == 0) {
    task = new BlindMoveToBall(server_port, offense_agents, defense_agents);
  } else {
    LOG(FATAL) << "Task " << task_name << " is not a recognized task!";
  }
  int task_id = all_tasks_.size();
  task->setID(task_id);
  all_tasks_.push_back(task);
  return *task;
}

void Curriculum::removeTask(std::string task_name) {
  for (vector<Task*>::iterator it = all_tasks_.begin() ; it != all_tasks_.end(); ++it) {
    if ((*it)->getName().compare(task_name) == 0) {
      delete *it;
      all_tasks_.erase(it);
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
  int indx = std::uniform_int_distribution<int>(0, all_tasks_.size()-1)(random_engine_);
  for (int i=0; i<current_tasks_.size(); ++i) {
    current_tasks_[i] = all_tasks_[indx];
  }
}

SequentialCurriculum::SequentialCurriculum(int num_agents) :
    Curriculum(num_agents),
    curr_task_indx_(0),
    task_eval_perf_(num_agents)
{
}

void SequentialCurriculum::queueTasks() {
  curr_task_indx_ = all_tasks_.size()-1;
  for (int t = 0; t < all_tasks_.size(); ++t) {
    float avg_perf = 0.;
    for (int a = 0; a < task_eval_perf_.size(); ++a) {
      if (task_eval_perf_[a].size() < all_tasks_.size()) {
        task_eval_perf_[a].resize(all_tasks_.size(), 0.);
      }
      avg_perf += task_eval_perf_[a][t];
    }
    avg_perf /= float(task_eval_perf_.size());
    if (avg_perf < FLAGS_performance_threshold) {
      curr_task_indx_ = t;
      break;
    }
  }
  for (int i=0; i<current_tasks_.size(); ++i) {
    current_tasks_[i] = all_tasks_[curr_task_indx_];
  }
}

void SequentialCurriculum::addEvalPerf(const Task& task, float perf, int tid) {
  CHECK_LE(tid, task_eval_perf_.size());
  CHECK_EQ(task_eval_perf_[tid].size(), all_tasks_.size());
  for (int i=0; i<all_tasks_.size(); ++i) {
    if (&task == all_tasks_[i]) {
      task_eval_perf_[tid][i] = perf;
    }
  }
}
