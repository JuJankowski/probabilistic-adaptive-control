#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <probabilistic_trajectory_tracking/probabilistic_adaptive_control.h>

using namespace probabilistic_trajectory_tracking;

#define I3 Eigen::MatrixXd::Identity(3, 3)

class PACNode
{
public:
  PACNode(double belief_threshold, unsigned int belief_stride, double std_q, double std_dq, double std_ori) {
    pac = std::make_unique<PAC>(
      belief_threshold, belief_stride, std_q * I3, 
      std_dq * I3, std_ori * I3);
  }
  
  void TaskDataCallback(const TaskConstPtr& msg) { pac->addTask(msg); }
  void ContextCallback(const ContextArrayConstPtr& msg) { pac->updateEnvironmentInference(msg); }
  void update(std::vector<State>& y_list, std::vector<Eigen::Quaterniond>& quat_list);
  
  std::unique_ptr<probabilistic_trajectory_tracking::PAC> pac;
  
private:
  tf::TransformListener tf_listener_;
  std::string base_frame_{"panda_link0"}, EE_frame_{"panda_K"};
};

void PACNode::update(std::vector<State>& y_list, std::vector<Eigen::Quaterniond>& quat_list)
{
  y_list.clear();
  quat_list.clear();
  // Read current end-effector pose from tf
  tf::StampedTransform transform;
  try{
    tf_listener_.lookupTransform(base_frame_, EE_frame_,  
                             ros::Time(0), transform);
  }
  catch (tf::TransformException ex){
    ROS_WARN("%s",ex.what());
    return;
  }
  
  State y;
  y.q = Eigen::Vector3d::Zero();
  y.dq = Eigen::Vector3d::Zero();
  y.q << transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z();
  
  Eigen::Quaterniond quat;
  quat.w() = transform.getRotation().w();
  quat.x() = transform.getRotation().x();
  quat.y() = transform.getRotation().y();
  quat.z() = transform.getRotation().z();
  
  Eigen::Quaterniond quat_ref;
  quat_ref.w() = 0.0;
  quat_ref.x() = 1.0;
  quat_ref.y() = 0.0;
  quat_ref.z() = 0.0;
  
  // Update task selection plus current phase
  pac->updateTaskTimeInference(y, true); // assume gripper is open
  
  // Roll out references to obtain nominal end-effector trajectory
  while(pac->getCurrentPhase() >= 0.0 && pac->getCurrentPhase() < 1.0 && pac->isActive()) {
    y_list.push_back(y);
    quat = quaternionClip(quat, quat_ref);
    quat_list.push_back(quat);
    DynamicReference ref;
    std::vector<Eigen::Vector3d> orientation_references;
    unsigned int gripper_action; // 0 -> no gripper action, 1 -> close gripper, 2 -> open gripper
    Eigen::Vector3d ddq = Eigen::Vector3d::Zero();
    if(pac->control(y_list.back(), quat_list.back(), ref, orientation_references, gripper_action)) {
      ddq = - 625.0 * (y.q - ref.q) - 25.0 * (y.dq - ref.dq) + ref.ddq;
      if(!orientation_references.empty()) {
        quat = quaternionExpMap(0.01 * orientation_references.back()) * quat;
      }
    }
    y.dq += 0.001 * ddq;
    y.q += 0.001 * y.dq;
  }
  
  if(!y_list.empty()) {
    y_list.push_back(y);
    quat_list.push_back(quat);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pac_node");
  ros::NodeHandle n("~");
  
  double belief_threshold = 1.0;
  unsigned int belief_stride = 500;
  double std_q = 2e-5, std_dq = 5e-4, std_ori = 1e-3;
  PACNode pac(belief_threshold, belief_stride, std_q, std_dq, std_ori);
  
  ros::Subscriber task_data_subs = n.subscribe<Task>("task", 4, &PACNode::TaskDataCallback, &pac);
  ros::Subscriber context_data_sub = n.subscribe<ContextArray>("context", 1, &PACNode::ContextCallback, &pac);
  
  ros::Publisher output_pose_pub = n.advertise<geometry_msgs::PoseStamped>("final_pose", 4);
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("trajectory_marker", 4);
  
  geometry_msgs::PoseStamped output_pose;
  output_pose.header.frame_id = "panda_link0";
  
  visualization_msgs::Marker vis_marker;
  vis_marker.id = 0;
  vis_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  vis_marker.action = visualization_msgs::Marker::ADD;
  vis_marker.color.r = 0;
  vis_marker.color.g = 0;
  vis_marker.color.b = 1.0;
  vis_marker.color.a = 1.0;
  vis_marker.pose.orientation.w = 1.0;
  vis_marker.pose.orientation.x = 0.0;
  vis_marker.pose.orientation.y = 0.0;
  vis_marker.pose.orientation.z = 0.0;
  vis_marker.scale.x = 0.01;
  vis_marker.scale.y = 0.01;
  vis_marker.scale.z = 0.01;
  vis_marker.lifetime = ros::Duration(1.0);

  double rate = 30;
  ros::Rate loop_rate(rate);

  unsigned int pose_count = 0;
  
  std::vector<State> y_list;
  std::vector<Eigen::Quaterniond> quat_list;
  
  while (ros::ok()) {
    loop_rate.sleep();
    
    pac.update(y_list, quat_list);
    if(y_list.empty()) {
      ros::spinOnce();
      continue;
    }
  
    output_pose.header.seq = pose_count;
    output_pose.header.stamp = ros::Time::now();
    
    output_pose.pose.position.x = y_list.back().q[0];
    output_pose.pose.position.y = y_list.back().q[1];
    output_pose.pose.position.z = y_list.back().q[2];
    
    output_pose.pose.orientation.w = quat_list.back().w();
    output_pose.pose.orientation.x = quat_list.back().x();
    output_pose.pose.orientation.y = quat_list.back().y();
    output_pose.pose.orientation.z = quat_list.back().z();

    vis_marker.header = output_pose.header;
    vis_marker.points.clear();
    
    unsigned int I = (unsigned int)(y_list.size() / 200.0);
    for(unsigned int i = 0; i < I; i++) {
      geometry_msgs::Point p;
      p.x = y_list[i*200].q[0]; p.y = y_list[i*200].q[1]; p.z = y_list[i*200].q[2];
      vis_marker.points.push_back(p);
    }
    
    output_pose_pub.publish(output_pose);
    marker_pub.publish(vis_marker);

    ros::spinOnce();
    
    pose_count++;
  }

  return 0;
}
