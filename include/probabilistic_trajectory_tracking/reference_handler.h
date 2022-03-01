#pragma once

#include <string>
#include <thread>
#include <fstream> // ifstream

#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <geometry_msgs/Vector3Stamped.h>
#include <tf/transform_listener.h>

#include <probabilistic_trajectory_tracking/utils.h>

namespace probabilistic_trajectory_tracking {

struct ReferenceFrame {
  ros::Time update_time;
  std::string frame_id;
  Eigen::Vector3d t;
  Eigen::Quaterniond quat;
  Eigen::Matrix3d R;
};

class ReferenceHandler {
 public:
  ReferenceHandler(unsigned int rate=30);
  
  void addFrame(const std::string frame_id);
  bool isValid(const std::string frame_id);
  State transformState(const State& y_in, const std::string frame_id);
  DynamicReference transformReference(const DynamicReference& ref_in, const std::string frame_id);
  Eigen::Vector3d getRefPosition(const std::string frame_id);
  Eigen::Quaterniond getRefQuaternion(const std::string frame_id);
  Eigen::Matrix3d getRefRotation(const std::string frame_id);

 private:
  std::string base_frame_{"panda_link0"};
  bool isValid_(const std::string frame_id);
  ros::Rate rate_{30};
  std::shared_ptr<std::mutex> data_mutex_;
  std::shared_ptr<std::thread> listener_thread_;
  void ListenerLoop();
  tf::TransformListener tf_listener_;
  std::vector<ReferenceFrame> reference_frames_;
};

}  // namespace procontrol
