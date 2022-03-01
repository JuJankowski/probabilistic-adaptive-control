#pragma once

#include <mutex>
#include <vector>

#include <eigen3/Eigen/Dense>

namespace probabilistic_trajectory_tracking
{

inline Eigen::MatrixXd msgToEigen(std::vector<double> data, unsigned int rows, unsigned int cols) {
  return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data.data(), rows, cols);
}

inline Eigen::MatrixXd invertPD(const Eigen::MatrixXd& A) {
  unsigned int dim = A.cols();
  return A.llt().solve(Eigen::MatrixXd::Identity(dim, dim));
}

// Clip a quaternion to the unit hemisphere around a reference quaternion
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

// Compute the log map that converts a quaternion rotation into a rotation vector
inline Eigen::Vector3d quaternionLogMap(const Eigen::Quaterniond& quat)
{
  double qv_norm = quat.vec().norm();
  if(qv_norm < 1e-6) {
    return Eigen::Vector3d::Zero();
  }
  double phi = 2.0 * atan2(qv_norm, quat.w());
  if(abs(phi) < 3.141) {
    return quat.vec() * phi / qv_norm;
  }
  if(phi > 0.0) {
    return quat.vec() * (phi - 2*3.141) / qv_norm;
  }
  return quat.vec() * (phi + 2*3.141) / qv_norm;
}

// Compute the exp map that converts a rotation vector into a relative quaternion
inline Eigen::Quaterniond quaternionExpMap(const Eigen::Vector3d& phi)
{
  double phi_norm = phi.norm();
  if(phi_norm < 1e-6) {
    return Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  }
  Eigen::Vector3d q_v = sin(phi_norm*0.5) * phi / phi_norm;
  return Eigen::Quaterniond(cos(phi_norm*0.5), q_v[0], q_v[1], q_v[2]);
}

struct Gaussian {
  Eigen::VectorXd mu;
  Eigen::MatrixXd sigma, sigma_inv;
  bool sigma_inverted{false};
  void invertSigma() {
    sigma_inv = invertPD(sigma);
    sigma_inverted = true;
  }
};

struct ContextualGaussian {
  Eigen::VectorXd mu_w, mu_s;
  Eigen::MatrixXd sigma_w, sigma_ws, sigma_s;
  Eigen::MatrixXd sigma_xi_inv; // xi = (w,s)
  double pi{0.0};
};

struct Basis {
  std::vector<Eigen::MatrixXd> Phi_q;
  std::vector<Eigen::MatrixXd> Phi_dq;
  std::vector<Eigen::MatrixXd> Phi_ddq;
};

struct DynamicReference {
  Eigen::VectorXd q, dq, ddq;
};

struct State {
  Eigen::VectorXd q, dq;
};

} // namespace
