#include<chrono>
#include<thread>
#include <probabilistic_trajectory_tracking/probabilistic_adaptive_control.h>

#define OBF

using namespace probabilistic_trajectory_tracking;

PAC::PAC(const double belief_threshold, const unsigned int belief_stride, const Eigen::MatrixXd& sigma_q, const Eigen::MatrixXd& sigma_dq, const Eigen::Matrix3d& sigma_orientation)
{
  tasks_.clear();
  task_mutex_ = std::make_shared<std::mutex>();
  belief_threshold_ = belief_threshold;
  belief_stride_ = belief_stride;
  sigma_q_ = sigma_q;
  sigma_dq_ = sigma_dq;
  sigma_orientation_ = sigma_orientation;
}

bool PAC::isActive()
{
  for(auto &task : tasks_) {
    if(task.active) {
      return true;
    }
  }
  return false;
}

void PAC::addTask(const TaskConstPtr& msg)
{
  TPMode mode;
  mode.frame_id = msg->reference_frame;
  ref_handler_.addFrame(msg->reference_frame);
  mode.tracker = probabilistic_trajectory_tracking::ProTrajTracker(3);
  mode.tracker.setup(*msg, sigma_q_, sigma_dq_);
  mode.mu_quat = Eigen::Quaterniond(msg->quaternion.w, msg->quaternion.x, msg->quaternion.y, msg->quaternion.z);
  Eigen::Matrix3d sigma_dtheta_D_inv = invertPD(msgToEigen(msg->sigma_dtheta, 3, 3));
  mode.P_dtheta = invertPD(sigma_dtheta_D_inv + invertPD(sigma_orientation_)) * sigma_dtheta_D_inv;

  // Check if ID is new
  for(auto &task : tasks_) {
    if(task.id == msg->id) {
      for(auto &mode_ : task.relative_modes) {
        if(mode.frame_id == mode_.frame_id) {
          task_mutex_->lock();
          mode_ = mode;
          task_mutex_->unlock();
          ROS_INFO_STREAM("PAC: Replaced mode for task with ID " << (unsigned int)task.id << ".");
          return;
        }
      }
      task_mutex_->lock();
      task.invalid_references = true;
      task.relative_modes.push_back(mode);
      task_mutex_->unlock();
      ROS_INFO_STREAM("PAC: New mode for task with ID " << (unsigned int)task.id << " registered.");
      return;
    }
  }
  
  if(msg->reference_frame != "panda_link0") {
    ROS_ERROR("PAC: Task rejected. Please add the mode with panda_link0 as reference first.");
    return;
  }
  
  // New task
  PACTask task;
  task.id = msg->id;
  task.predecessor_id = msg->predecessor_id;
  task.goal = msg->goal;
  task.remove_after_success = msg->remove_after_success;
  task.context_id = msg->context_id;
  task.relative_modes.push_back(mode);
  task.registration_time = ros::Time::now();
  
  task.mode.frame_id = msg->reference_frame;
  task.mode.tracker = ProTrajTracker(3);
  task.mode.tracker.setup(*msg, sigma_q_, sigma_dq_);
    
  ROS_INFO_STREAM("PAC: New task with ID " << (unsigned int)task.id << " registered.");
  
  task_mutex_->lock();
  tasks_.push_back(task);
  task_mutex_->unlock();
}

void PAC::updateEnvironmentInference(const ContextArrayConstPtr& msg)
{
  for(auto &task : tasks_) {
    // Update context -> compute conditional distribution
    Gaussian context;
    for(auto &id : task.context_id) {
      for(auto &context_msg : msg->context) {
        if(id == context_msg.id) {
          unsigned int dim_1 = context.mu.size();
          unsigned int dim_2 = context_msg.s.size();
          Eigen::VectorXd vec_joined(dim_1 + dim_2);
          Eigen::MatrixXd mat_joined = Eigen::MatrixXd::Zero(dim_1 + dim_2, dim_1 + dim_2);
          vec_joined << context.mu, msgToEigen(context_msg.s, dim_2, 1);
          mat_joined.topLeftCorner(dim_1, dim_1) = context.sigma;
          mat_joined.bottomRightCorner(dim_2, dim_2) = msgToEigen(context_msg.sigma, dim_2, dim_2);
          context.mu = vec_joined;
          context.sigma = mat_joined;
          break;
        }
      }
    }
    context.invertSigma();

#ifndef OBF
    // Arbitrary basis function support
    // Update reference frames -> product of Gaussians in base frame
    // 1. Construct full rank square transformation matrix
    unsigned int ndof = task.mode.tracker.getNumDoF();
    unsigned int dimw = task.mode.tracker.getDimW();
    
    unsigned int num_samples = dimw / ndof; // The result is an integer if the task is defined properly
    Eigen::MatrixXd Phi_tilde = Eigen::MatrixXd::Zero(dimw, dimw);
    
    unsigned int K = task.mode.tracker.getMaxPhase();
    double factor = (double)(K-1) / (double)(num_samples-1);
    for(unsigned int i = 0; i < num_samples; i++) {
      unsigned int k = (unsigned int)(factor * i);
      Phi_tilde.block(i*ndof, 0, ndof, dimw) = task.mode.tracker.getPhi(k);
    }
    Eigen::MatrixXd Phi_tilde_inv = Phi_tilde.partialPivLu().inverse();
    
    std::vector<Eigen::VectorXd> mu_w_O;
    std::vector<Eigen::MatrixXd> lambda_w_O;
    mu_w_O.clear();
    lambda_w_O.clear();
    task.invalid_references = false;
    for(auto &mode : task.relative_modes) {
      mode.tracker.updateContext(context);
      auto traj_dist_frame = mode.tracker.getTrajDist();
      
      if(mode.frame_id == "panda_link0") {
        mu_w_O.push_back(traj_dist_frame.mu);
        lambda_w_O.push_back(traj_dist_frame.sigma_inv);
        continue;
      }
      
      if(!ref_handler_.isValid(mode.frame_id)) {
        task.invalid_references = true;
        break;
      }
      
      Eigen::VectorXd t_O = Eigen::VectorXd::Zero(ndof*num_samples);
      Eigen::MatrixXd R_O = Eigen::MatrixXd::Identity(ndof*num_samples, ndof*num_samples);
      Eigen::Vector3d t = ref_handler_.getRefPosition(mode.frame_id);
      Eigen::Matrix3d R = ref_handler_.getRefRotation(mode.frame_id);
      for(unsigned int i = 0; i < num_samples; i++) {
        t_O.segment(i * ndof, ndof) = t;
        R_O.block(i * ndof, i * ndof, ndof, ndof) = R;
      }
      Eigen::MatrixXd A_O = Phi_tilde_inv * R_O * Phi_tilde;
      
      mu_w_O.push_back(A_O * traj_dist_frame.mu + Phi_tilde_inv * t_O);
      lambda_w_O.push_back(invertPD(A_O * traj_dist_frame.sigma * A_O.transpose()));
    }
#endif
    
#ifdef OBF
    // OBFs
    // Update reference frames -> product of Gaussians in base frame
    std::vector<Eigen::VectorXd> mu_w_O;
    std::vector<Eigen::MatrixXd> lambda_w_O;
    mu_w_O.clear();
    lambda_w_O.clear();
    task.invalid_references = false;
    for(auto &mode : task.relative_modes) {
      mode.tracker.updateContext(context);
      auto traj_dist_frame = mode.tracker.getTrajDist();
      
      if(mode.frame_id == "panda_link0") {
        mu_w_O.push_back(traj_dist_frame.mu);
        lambda_w_O.push_back(traj_dist_frame.sigma_inv);
        continue;
      }
      
      if(!ref_handler_.isValid(mode.frame_id)) {
        task.invalid_references = true;
        break;
      }
      
      unsigned int dimw = traj_dist_frame.mu.size();
      unsigned int num_via = dimw / 3;
      Eigen::VectorXd t_O = Eigen::VectorXd::Zero(dimw);
      Eigen::MatrixXd R_O = Eigen::MatrixXd::Identity(dimw, dimw);
      Eigen::Vector3d t = ref_handler_.getRefPosition(mode.frame_id);
      Eigen::Matrix3d R = ref_handler_.getRefRotation(mode.frame_id);
      for(unsigned int i = 0; i < num_via; i++) {
        if(i < num_via-2) t_O.segment(i * 3, 3) = t;
        R_O.block(i * 3, i * 3, 3, 3) = R;
      }
      mu_w_O.push_back(R_O * traj_dist_frame.mu + t_O);
      lambda_w_O.push_back(invertPD(R_O * traj_dist_frame.sigma * R_O.transpose()));
    }
#endif

    
    // Valid for both again
    if(!task.invalid_references) {
      Eigen::MatrixXd lambda_w = Eigen::MatrixXd::Zero(mu_w_O[0].size(), mu_w_O[0].size());
      Eigen::VectorXd mu_w_tmp = Eigen::VectorXd::Zero(mu_w_O[0].size());
      for(unsigned int i = 0; i < task.relative_modes.size(); i++) {
        lambda_w += lambda_w_O[i];
        mu_w_tmp += lambda_w_O[i] * mu_w_O[i];
      }
      Eigen::MatrixXd sigma_w = invertPD(lambda_w);
      double scaling = (double)task.relative_modes.size();
      
      Gaussian traj_dist;
      traj_dist.sigma = sigma_w * scaling;
      traj_dist.mu = sigma_w * mu_w_tmp;
      traj_dist.sigma_inv = lambda_w / scaling;
      traj_dist.sigma_inverted = true;
      
      task_mutex_->lock();
      task.mode.tracker.setTrajDist(traj_dist);
      task_mutex_->unlock();
    }
  }
}

void PAC::updateTaskTimeInference(const State& y, const bool gripper_open)
{
  ros::Time time = ros::Time::now();
  int id_active = -1;
  double max_belief = belief_threshold_;
  for(auto &task : tasks_) {
    if((time - task.registration_time).toSec() < 3.0 || 
       (task.goal == 1 && !gripper_open) || 
       (task.goal == 2 && gripper_open) ||
       task.invalid_references) {
      std::cout << (unsigned int)task.id << ": " << ((time - task.registration_time).toSec() < 3.0) << " " 
                << (task.goal == 1 && !gripper_open) << " " << (task.goal == 2 && gripper_open) << " " << task.invalid_references << std::endl;
      continue;
    }
    
    if(finished_task_id_ != -1) {
      if(std::find(task.predecessor_id.begin(), task.predecessor_id.end(), finished_task_id_) == task.predecessor_id.end()) {
        continue;
      }
    }
    
    if(!task.active) {
      task.mode.tracker.resetPhase(0);
    }
    double belief = task.relative_modes[0].tracker.getContextBelief() * task.mode.tracker.getCurrentBelief(y);
      
    
    //unsigned int k;
    //double belief = task.relative_modes[0].tracker.getContextBelief() * task.mode.tracker.getBelief(y, k, belief_stride_, !task.active);
    //task.mode.tracker.resetPhase(k);
    /*if(!task.active) {
      task.mode.tracker.resetPhase(k);
      belief /= 10.0;
    }*/
    std::cout << (unsigned int)task.id << ": " << belief /*<< ", " << k*/ << std::endl;
      
    if(belief > max_belief) {
      max_belief = belief;
      id_active = task.id;
    }
  }
  task_mutex_->lock();
  for(auto &task : tasks_) {
    task.active = (task.id == id_active);
  }
  
  if(finished_task_id_ != -1) {
    for(unsigned int k=0; k<tasks_.size(); k++) {
      if(tasks_[k].id == finished_task_id_) {
        if(tasks_[k].remove_after_success) {
          ROS_INFO_STREAM("PAC: Removed task with ID " << (unsigned int)tasks_[k].id << ".");
          tasks_.erase(tasks_.begin()+k);
        }
        break;
      }
    }
  }
  task_mutex_->unlock();
  finished_task_id_ = -1;
}

double PAC::getCurrentPhase()
{
  for(auto &task : tasks_) {
    if(task.active) {
      return task.mode.tracker.getRelativePhase();
    }
  }
  return -1.0;
}

// Clip a quaternion to the unit hemisphere around the initial quaternion
inline Eigen::Quaterniond quaternionClip(const Eigen::Quaterniond& quat, const Eigen::Quaterniond& quat_ref)
{
  Eigen::Vector4d q1(quat.w(), quat.x(), quat.y(), quat.z());
  Eigen::Vector4d q2(quat_ref.w(), quat_ref.x(), quat_ref.y(), quat_ref.z());
  
  Eigen::Vector4d e = q1 - q2;
  if(e.squaredNorm() > 2) {
    return Eigen::Quaterniond(-q1);
  }
  return quat;
}

bool PAC::control(const State& y, const Eigen::Quaterniond& quat, DynamicReference& ref, std::vector<Eigen::Vector3d>& ori_refs, unsigned int& gripper_action)
{
  gripper_action = 0; // 0 -> no gripper action, 1 -> close gripper, 2 -> open gripper
  ori_refs.clear();
  ref.q = y.q;
  ref.dq = y.dq;
  ref.ddq = Eigen::VectorXd::Zero(y.q.size());
  
  if(!task_mutex_->try_lock()) {
    for(auto &task : tasks_) {
      if(task.active) {
        ref = last_reference_;
        return true;
      }
    }
    return false;
  }
  
  bool active = false;
  for(auto &task : tasks_) {
    if(!task.active) {
      continue;
    }
    // TODO: check first part of if
    if(task.relative_modes[0].tracker.getContextBelief() * task.mode.tracker.getCurrentBelief(y) < belief_threshold_) {
      task.active = false;
      break;
    }
    active = true;
    ref = task.mode.tracker.getReference(y);
    last_reference_ = ref;
    for(auto &mode : task.relative_modes) {
      Eigen::Quaterniond mu_quat = ref_handler_.getRefQuaternion(mode.frame_id) * mode.mu_quat.conjugate();
      Eigen::Matrix3d R = ref_handler_.getRefRotation(mode.frame_id);
      ori_refs.push_back(R * mode.P_dtheta * R.transpose() * quaternionLogMap(mu_quat * quat.conjugate()));
    }
    task.mode.tracker.forwardPhase();
    
    if(task.mode.tracker.getRelativePhase() == 1.0) {
      gripper_action = task.goal;
      finished_task_id_ = task.id;
    }
  }
  
  task_mutex_->unlock();
  return active;
}
