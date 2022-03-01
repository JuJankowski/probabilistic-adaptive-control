#pragma once

#include <string>
#include <vector>
#include <fstream> // ifstream

#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <probabilistic_trajectory_tracking/probabilistic_trajectory_tracking.h>
#include <probabilistic_trajectory_tracking/reference_handler.h>
#include <probabilistic_trajectory_tracking/utils.h>
#include <probabilistic_trajectory_tracking/Task.h>
#include <probabilistic_trajectory_tracking/ContextArray.h>

namespace probabilistic_trajectory_tracking {

struct TPMode {
  std::string frame_id;
  probabilistic_trajectory_tracking::ProTrajTracker tracker;
  Eigen::Quaterniond mu_quat;
  Eigen::Matrix3d P_dtheta;
};

struct PACTask {
  unsigned char id{0};
  std::vector<unsigned char> predecessor_id; // list containing typical predecessor tasks
  unsigned int goal{0}; // 0 -> pure motion, 1 -> pick, 2 -> place
  bool remove_after_success{false};
  bool active{false};
  bool invalid_references{false};
  ros::Time registration_time;
  
  TPMode mode; // If there are multiple KF TPModes, this contains the fused mode in the world frame
  std::vector<TPMode> relative_modes;
  std::vector<unsigned char> context_id; // context selection
};

class PAC {
 public:
  PAC(const double belief_threshold, const unsigned int belief_stride, const Eigen::MatrixXd& sigma_q, const Eigen::MatrixXd& sigma_dq, const Eigen::Matrix3d& sigma_orientation);
  void addTask(const TaskConstPtr& msg);
  void updateEnvironmentInference(const ContextArrayConstPtr& msg); // called with ~ 30 Hz
  void updateTaskTimeInference(const State& y, const bool gripper_open); // called with ~ 30 Hz
  bool control(const State& y, const Eigen::Quaterniond& quat, DynamicReference& ref, std::vector<Eigen::Vector3d>& ori_refs, unsigned int& gripper_action); // called with ~ 1 kHz
  double getCurrentPhase();
  bool isActive();

 private:
  unsigned int task_ndof_{3};
  std::vector<PACTask> tasks_;
  std::shared_ptr<std::mutex> task_mutex_;
  ReferenceHandler ref_handler_;
  
  DynamicReference last_reference_;
  double belief_threshold_;
  unsigned int belief_stride_;
  int finished_task_id_{-1}; // different from -1 if a task has just finished
  
  Eigen::MatrixXd sigma_q_, sigma_dq_; // system uncertainty
  Eigen::Matrix3d sigma_orientation_;
};

}  // namespace procontrol
