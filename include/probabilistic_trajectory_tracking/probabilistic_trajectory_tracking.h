#pragma once

#include <mutex>
#include <vector>
#include <algorithm>

#include <eigen3/Eigen/Dense>

#include <probabilistic_trajectory_tracking/utils.h>
#include <probabilistic_trajectory_tracking/Task.h>

namespace probabilistic_trajectory_tracking
{

/**
 * Generic representation of discrete trajectory distributions given in a latent space.
 */
class ProTrajTracker {

public:
  ProTrajTracker(unsigned int ndof=3) {
    ndof_ = ndof;
    traj_dist_mutex_ = std::make_shared<std::mutex>();
  }
  
  void setup(const Task& task, const Eigen::MatrixXd& sigma_q, const Eigen::MatrixXd& sigma_dq);
  unsigned int getNumDoF() { return ndof_; }
  unsigned int getDimW() { return dimw_; }
  unsigned int getDimS() { return dims_; }
  
  DynamicReference getReference(const State& x);
  
  double getCurrentBelief(const State& x);
  double getBelief(const State& x, unsigned int& k_best, const unsigned int stride, const bool closest);

  void resetPhase(const unsigned int k=0) { k_ = k; }
  void forwardPhase() { if(k_ < K_-1) k_++; }
  unsigned int getPhase() { return k_; }
  double getRelativePhase() { 
    if(K_ == 1) return 1.0;
    return (double)k_/(double)(K_-1);
  }
  unsigned int getMaxPhase() { return K_; }
  
  bool isSetup() { return is_setup_; }
  void enable() { is_setup_ = true; }
  void disable() { is_setup_ = false; }
  
  void setTrajDist(const Gaussian& traj_dist); // only call this from outside if you know what you are doing
  const Gaussian& getTrajDist() { return traj_dist_; }
  void updateContext(const Gaussian& context); // Might be called from another thread than getReference
  double getContextBelief() { return context_probability_; }
  
  const Eigen::MatrixXd& getPhi(const unsigned int k) { return basis_.Phi_q[k]; }
  const Eigen::MatrixXd& getdPhi(const unsigned int k) { return basis_.Phi_dq[k]; }
  const Eigen::MatrixXd& getddPhi(const unsigned int k) { return basis_.Phi_ddq[k]; }
  
private:
  std::shared_ptr<std::mutex> traj_dist_mutex_;
  double last_belief_{0.0};

  double pdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& cov);
  double pdf_from_inv(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& cov_inv);

  Basis basis_;
  Gaussian traj_dist_;
  ContextualGaussian contextual_gaussian_;
  double context_probability_{0.0};
  double getCurrentBelief(const State& x, const Eigen::VectorXd& mu_w, const Eigen::MatrixXd& sigma_w);

  Eigen::MatrixXd sigma_q_, sigma_q_inv_, sigma_dq_, sigma_dq_inv_;
  
  unsigned int k_{0}, K_{0}; // Phase parameters
  unsigned int ndof_{0}, dimw_{0}, dims_{0};
  
  bool is_setup_{false};
};

} // namespace
