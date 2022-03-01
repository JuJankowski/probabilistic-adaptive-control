#include <probabilistic_trajectory_tracking/reference_handler.h>

using namespace probabilistic_trajectory_tracking;

ReferenceHandler::ReferenceHandler(unsigned int rate) 
{
  // create thread which reads the tf transforms with <rate> Hz
  rate_ = ros::Rate(rate);
  data_mutex_ = std::make_shared<std::mutex>();
  listener_thread_ = std::make_shared<std::thread>(&ReferenceHandler::ListenerLoop, this);
}

void ReferenceHandler::addFrame(const std::string frame_id)
{
  data_mutex_->lock();
  for(auto &ref : reference_frames_) {
    if(ref.frame_id == frame_id) {
      data_mutex_->unlock();
      return;
    }
  }
  ReferenceFrame ref;
  ref.frame_id = frame_id;
  ref.update_time = ros::Time::now() - ros::Duration(1.0);
  reference_frames_.push_back(ref);
  data_mutex_->unlock();
}

bool ReferenceHandler::isValid(const std::string frame_id)
{
  data_mutex_->lock();
  bool valid = isValid_(frame_id);
  data_mutex_->unlock();
  return valid;
}
bool ReferenceHandler::isValid_(const std::string frame_id)
{
  ros::Time time = ros::Time::now();
  for(auto &ref : reference_frames_) {
    if(ref.frame_id == frame_id) {
      if((time - ref.update_time).toSec() < 10.0) {
        return true;
      }
      std::cout << (time - ref.update_time).toSec() << std::endl;
      return false;
    }
  }
  return false;
}

State ReferenceHandler::transformState(const State& y_in, const std::string frame_id)
{
  data_mutex_->lock();
  if(!isValid_(frame_id)) {
    ROS_WARN_STREAM_THROTTLE(1, "ReferenceHandler: Frame ID " << frame_id << " is not valid.");
    data_mutex_->unlock();
    return State();
  }

  State y_out;
  
  for(auto &ref : reference_frames_) {
    if(ref.frame_id == frame_id) {
      y_out.q = ref.R.transpose() * (y_in.q - ref.t);
      y_out.dq = ref.R.transpose() * y_in.dq;
      data_mutex_->unlock();
      return y_out; 
    }
  }
  data_mutex_->unlock();
  return State(); // will never be reached
}

DynamicReference ReferenceHandler::transformReference(const DynamicReference& ref_in, const std::string frame_id)
{
  data_mutex_->lock();
  if(!isValid_(frame_id)) {
    ROS_WARN_STREAM_THROTTLE(1, "ReferenceHandler: Frame ID " << frame_id << " is not valid.");
    data_mutex_->unlock();
    return DynamicReference();
  }
  
  DynamicReference ref_out;
  
  for(auto &ref : reference_frames_) {
    if(ref.frame_id == frame_id) {
      ref_out.q = ref.R * ref_in.q + ref.t;
      ref_out.dq = ref.R * ref_in.dq;
      ref_out.ddq = ref.R * ref_in.ddq;
      data_mutex_->unlock();
      return ref_out; 
    }
  }
  data_mutex_->unlock();
  return DynamicReference(); // will never be reached
}

Eigen::Vector3d ReferenceHandler::getRefPosition(const std::string frame_id)
{
  data_mutex_->lock();
  if(!isValid_(frame_id)) {
    ROS_WARN_STREAM_THROTTLE(1, "ReferenceHandler: Frame ID " << frame_id << " is not valid.");
    data_mutex_->unlock();
    return Eigen::Vector3d();
  }
  for(auto &ref : reference_frames_) {
    if(ref.frame_id == frame_id) {
      data_mutex_->unlock();
      return ref.t;
    }
  }
  data_mutex_->unlock();
  return Eigen::Vector3d(); // will never be reached
}

Eigen::Quaterniond ReferenceHandler::getRefQuaternion(const std::string frame_id)
{
  data_mutex_->lock();
  if(!isValid_(frame_id)) {
    ROS_WARN_STREAM_THROTTLE(1, "ReferenceHandler: Frame ID " << frame_id << " is not valid.");
    data_mutex_->unlock();
    return Eigen::Quaterniond();
  }
  for(auto &ref : reference_frames_) {
    if(ref.frame_id == frame_id) {
      data_mutex_->unlock();
      return ref.quat;
    }
  }
  data_mutex_->unlock();
  return Eigen::Quaterniond(); // will never be reached
}

Eigen::Matrix3d ReferenceHandler::getRefRotation(const std::string frame_id)
{
  data_mutex_->lock();
  if(!isValid_(frame_id)) {
    ROS_WARN_STREAM_THROTTLE(1, "ReferenceHandler: Frame ID " << frame_id << " is not valid.");
    data_mutex_->unlock();
    return Eigen::Matrix3d();
  }
  for(auto &ref : reference_frames_) {
    if(ref.frame_id == frame_id) {
      data_mutex_->unlock();
      return ref.R;
    }
  }
  data_mutex_->unlock();
  return Eigen::Matrix3d(); // will never be reached
}

void ReferenceHandler::ListenerLoop()
{
  while(ros::ok()) {
    data_mutex_->lock();
    std::vector<ReferenceFrame> reference_frames_tmp = reference_frames_;
    data_mutex_->unlock();
    for(auto &ref : reference_frames_tmp) {
      if(ref.frame_id == base_frame_) {
        ref.update_time = ros::Time::now();
        ref.t = Eigen::Vector3d::Zero();
        ref.quat.w() = 1.0;
        ref.quat.x() = 0.0;
        ref.quat.y() = 0.0;
        ref.quat.z() = 0.0;
        ref.R = ref.quat.toRotationMatrix();
      } else {
        tf::StampedTransform transform;
        try{
          tf_listener_.lookupTransform(base_frame_, ref.frame_id,  
                                        ros::Time(0), transform);
        }
        catch (tf::TransformException ex){
          ROS_WARN_THROTTLE(1, "%s", ex.what());
          continue;
        }

        ref.update_time = transform.stamp_;
        ref.t = Eigen::Vector3d(transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z());
        ref.quat.w() = transform.getRotation().w();
        ref.quat.x() = transform.getRotation().x();
        ref.quat.y() = transform.getRotation().y();
        ref.quat.z() = transform.getRotation().z();
        ref.R = ref.quat.toRotationMatrix();
      }
    }
    data_mutex_->lock();
    reference_frames_ = reference_frames_tmp;
    data_mutex_->unlock();
    rate_.sleep();
  }
}
