#include <probabilistic_trajectory_tracking/probabilistic_trajectory_tracking.h>

#include <cmath>
#include <iostream>

using namespace probabilistic_trajectory_tracking;

// only call this from outside if you know what you are doing
void ProTrajTracker::setTrajDist(const Gaussian& traj_dist)
{
  // Only copy operations while locking the guard
  traj_dist_mutex_->lock();
  traj_dist_ = traj_dist;
  if(!traj_dist_.sigma_inverted) {
    traj_dist_.invertSigma();
  }
  traj_dist_mutex_->unlock();
}

void ProTrajTracker::setup(const Task& task, const Eigen::MatrixXd& sigma_q, const Eigen::MatrixXd& sigma_dq)
{
  // Read contextualized Gaussian
  dimw_ = task.mu_w.size();
  dims_ = task.mu_s.size();
  
  contextual_gaussian_.mu_w = msgToEigen(task.mu_w, dimw_, 1);
  contextual_gaussian_.sigma_w = msgToEigen(task.sigma_w, dimw_, dimw_);
  
  if(dims_ > 0) {
    contextual_gaussian_.mu_s = msgToEigen(task.mu_s, dims_, 1);
    contextual_gaussian_.sigma_s = msgToEigen(task.sigma_s, dims_, dims_);
    contextual_gaussian_.sigma_ws = msgToEigen(task.sigma_ws, dimw_, dims_);
    
    Eigen::MatrixXd sigma_xi = Eigen::MatrixXd::Zero(dimw_+dims_, dimw_+dims_);
    sigma_xi.topLeftCorner(dimw_, dimw_) = contextual_gaussian_.sigma_w;
    sigma_xi.bottomRightCorner(dims_, dims_) = contextual_gaussian_.sigma_s;
    sigma_xi.topRightCorner(dimw_, dims_) = contextual_gaussian_.sigma_ws;
    sigma_xi.bottomLeftCorner(dims_, dimw_) = contextual_gaussian_.sigma_ws.transpose();
    contextual_gaussian_.sigma_xi_inv = sigma_xi.inverse();
  } else {
    context_probability_ = 1.0;
  }
  
  // Read basis
  unsigned int num_rows_phi = task.Phi_q.size() / dimw_;
  K_ = num_rows_phi / ndof_;
  
  Eigen::MatrixXd Phi_q = msgToEigen(task.Phi_q, num_rows_phi, dimw_);
  Eigen::MatrixXd Phi_dq = msgToEigen(task.Phi_dq, num_rows_phi, dimw_);
  Eigen::MatrixXd Phi_ddq = msgToEigen(task.Phi_ddq, num_rows_phi, dimw_);
  
  basis_.Phi_q.clear();
  basis_.Phi_dq.clear();
  basis_.Phi_ddq.clear();
  
  for(unsigned int k = 0; k < K_; k++) {
    Eigen::MatrixXd phi_q = Phi_q.block(k*ndof_, 0,  ndof_, dimw_);
    basis_.Phi_q.push_back(phi_q);
    Eigen::MatrixXd phi_dq = Phi_dq.block(k*ndof_, 0,  ndof_, dimw_);
    basis_.Phi_dq.push_back(phi_dq);
    Eigen::MatrixXd phi_ddq = Phi_ddq.block(k*ndof_, 0,  ndof_, dimw_);
    basis_.Phi_ddq.push_back(phi_ddq);
  }
  basis_.Phi_ddq[K_-1] = Eigen::MatrixXd::Zero(ndof_, dimw_);
  
  // Read system uncertainty
  sigma_q_ = sigma_q;
  sigma_dq_ = sigma_dq;
  
  sigma_q_inv_ = sigma_q_.inverse();
  sigma_dq_inv_ = sigma_dq_.inverse();
  
  traj_dist_.mu = contextual_gaussian_.mu_w;
  traj_dist_.sigma = contextual_gaussian_.sigma_w;
  traj_dist_.invertSigma();
  
  is_setup_ = true;
}

void ProTrajTracker::updateContext(const Gaussian& context)
{
  if(!is_setup_ || dims_ == 0) {
    return;
  }
  
  Eigen::MatrixXd sigma_xi_hat_inv = contextual_gaussian_.sigma_xi_inv;
  sigma_xi_hat_inv.bottomRightCorner(dims_, dims_) += context.sigma_inv;
  Eigen::MatrixXd sigma_xi_hat = invertPD(sigma_xi_hat_inv);
  
  Eigen::MatrixXd sigma_ws_hat = sigma_xi_hat.topRightCorner(dimw_, dims_);
  
  Gaussian traj_dist_tmp;
  traj_dist_tmp.mu = contextual_gaussian_.mu_w + sigma_ws_hat * context.sigma_inv * (context.mu - contextual_gaussian_.mu_s);
  traj_dist_tmp.sigma = sigma_xi_hat.topLeftCorner(dimw_, dimw_);
  traj_dist_tmp.invertSigma();
  
  setTrajDist(traj_dist_tmp);
  
  context_probability_ = pdf(context.mu, contextual_gaussian_.mu_s, contextual_gaussian_.sigma_s + context.sigma);
}

DynamicReference ProTrajTracker::getReference(const State& x)
{
  DynamicReference reference;
  reference.q = x.q;
  reference.dq = x.dq;
  reference.ddq = Eigen::VectorXd::Zero(ndof_);

  if(!is_setup_) {
    return reference;
  }
  
  Eigen::VectorXd mu_w;
  Eigen::MatrixXd sigma_w_inv;
  if(traj_dist_mutex_->try_lock()) {
    mu_w = traj_dist_.mu;
    sigma_w_inv = traj_dist_.sigma_inv;
    traj_dist_mutex_->unlock();
  } else {
    return reference;
  }
  
  // Read basis functions for k_
  Eigen::MatrixXd phi_q = basis_.Phi_q[k_];
  Eigen::MatrixXd phi_dq = basis_.Phi_dq[k_];
  Eigen::MatrixXd phi_ddq = basis_.Phi_ddq[k_];
  
  Eigen::MatrixXd l_q = phi_q.transpose() * sigma_q_inv_;
  Eigen::MatrixXd l_dq = phi_dq.transpose() * sigma_dq_inv_;
  
  // Compute state-conditional reference trajectory
  Eigen::MatrixXd sigma_w_x = invertPD(sigma_w_inv + l_q * phi_q + l_dq * phi_dq);
  Eigen::VectorXd mu_w_x = mu_w + sigma_w_x * ( l_q * (x.q - phi_q * mu_w) + l_dq * (x.dq - phi_dq * mu_w) );
  
  reference.q = phi_q * mu_w_x;
  reference.dq = phi_dq * mu_w_x;
  reference.ddq = phi_ddq * mu_w_x;
  
  return reference;
}

double ProTrajTracker::getCurrentBelief(const State& x)
{
  if(!is_setup_) {
    return 0.0;
  }
  
  Eigen::VectorXd mu_w;
  Eigen::MatrixXd sigma_w;
  if(traj_dist_mutex_->try_lock()) {
    mu_w = traj_dist_.mu;
    sigma_w = traj_dist_.sigma;
    traj_dist_mutex_->unlock();
  } else {
    return last_belief_;
  }
  
  return getCurrentBelief(x, mu_w, sigma_w);
}

double ProTrajTracker::getCurrentBelief(const State& x, const Eigen::VectorXd& mu_w, const Eigen::MatrixXd& sigma_w)
{
  if(!is_setup_) {
    return 0.0;
  }

  // Read basis functions for k_
  Eigen::MatrixXd phi_q = basis_.Phi_q[k_];
  Eigen::MatrixXd phi_dq = basis_.Phi_dq[k_];
  
  // Stack variables for state representation
  Eigen::VectorXd x_vec(2*ndof_);
  x_vec << x.q, x.dq;
  
  Eigen::MatrixXd phi_x(2*ndof_, dimw_);
  phi_x << phi_q,
           phi_dq;
  
  Eigen::VectorXd mu_x = phi_x * mu_w;
  Eigen::MatrixXd sigma_x = phi_x * sigma_w * phi_x.transpose();
  
  sigma_x.block(0, 0, ndof_, ndof_) += sigma_q_;
  sigma_x.block(ndof_, ndof_, ndof_, ndof_) += sigma_dq_;  
  
  last_belief_ = pdf(x_vec, mu_x, sigma_x);
  
  return last_belief_;
}

double ProTrajTracker::getBelief(const State& x, unsigned int& k_best, const unsigned int stride, const bool active)
{
  if(!is_setup_) {
    return 0.0;
  }
  
  Eigen::VectorXd mu_w;
  Eigen::MatrixXd sigma_w;
  if(traj_dist_mutex_->try_lock()) {
    mu_w = traj_dist_.mu;
    sigma_w = traj_dist_.sigma;
    traj_dist_mutex_->unlock();
  } else {
    return last_belief_;
  }
  
  unsigned int k_center;
  if(!active) {
    // find closest euclidean distance in position along horizon (excluding end part)
    double d_min = 1000.0;
    for(unsigned int k = 0; k < (unsigned int)(0.75 * K_); k++) {
      Eigen::VectorXd e = x.q - basis_.Phi_q[k] * mu_w;
      double d = e.dot(e);
      if(d < d_min) {
        d_min = d;
        k_center = k;
      }
    }
  } else {
    k_center = k_;
  }
  
  unsigned int k_low = std::max((unsigned int)30, k_center - stride);
  unsigned int k_high = std::min(K_, k_center + stride);
  
  unsigned int k_internal = k_;
  double b_max = 0.0, b;
  for(unsigned int k = k_low; k < k_high; k++) {
    k_ = k;
    b = getCurrentBelief(x, mu_w, sigma_w);
    if(b > b_max) {
      b_max = b;
      k_best = k;
    }
  }

  if(!active) {
    k_ = 0;
    b = getCurrentBelief(x, mu_w, sigma_w);
    if(b > b_max) {
      b_max = b;
      k_best = 0;
    }
  }
  
  k_ = k_internal;
  
  return b_max;
}

double ProTrajTracker::pdf(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& cov)
{
  double norm = sqrt(pow(2 * M_PI, mu.size()) * cov.determinant());
  Eigen::VectorXd dist = x - mu;
  double proj_dist = dist.transpose() * cov.llt().solve(dist);
  
  return exp(- 0.5 * proj_dist) / norm;
}

double ProTrajTracker::pdf_from_inv(const Eigen::VectorXd& x, const Eigen::VectorXd& mu, const Eigen::MatrixXd& cov_inv)
{
  double norm = sqrt(pow(2 * M_PI, mu.size()) / cov_inv.determinant());
  Eigen::VectorXd dist = x - mu;
  double proj_dist = dist.transpose() * cov_inv * dist;
  
  return exp(- 0.5 * proj_dist) / norm;
}
