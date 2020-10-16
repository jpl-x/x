/*
 * Copyright 2020 California  Institute  of Technology (“Caltech”)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <x/vision/triangulation.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

using namespace x;

Triangulation::Triangulation(const x::QuaternionArray& quat_l,
                             const x::Vector3Array& pos_l,
                             const unsigned int max_iter,
                             const double term)
  : n_poses_( quat_l.size() )
  , positions_( pos_l )
  , max_iter_(max_iter)
  , term_(term)
{
  // Initialize pose vectors
  rotations_ = std::vector<Eigen::Matrix3d>(n_poses_, Eigen::Matrix3d());
  projs_     = std::vector<cv::Mat>(n_poses_, cv::Mat());

  // Attitudes and translations must have the same length
  // TODO(jeff): Create pose class and remove this
  assert(pos_l.size() == n_poses_);

  for (size_t i = 0; i < n_poses_; ++i)
  {
    // Compute rotation matrix
    rotations_[i] = quat_l[i].toRotationMatrix().transpose();

    // Compute projection matrix
    pose2proj(rotations_[i], positions_[i], projs_[i]); 
  }
}

Triangulation::Triangulation(const x::AttitudeList& attitudes,
                             const x::TranslationList& translations,
                             const unsigned int max_iter,
                             const double term)
  : n_poses_( attitudes.size() )
  , max_iter_(max_iter)
  , term_(term)
{
  // Initialize pose vectors
  rotations_ = std::vector<Eigen::Matrix3d>(n_poses_, Eigen::Matrix3d());
  positions_ = std::vector<Eigen::Vector3d>(n_poses_, Eigen::Vector3d());
  projs_     = std::vector<cv::Mat>(n_poses_, cv::Mat());

  // Attitudes and translations must have the same length
  // TODO(jeff): Create pose class and remove this
  assert(translations.size() == n_poses_);

  for (size_t i = 0; i < n_poses_; ++i)
  {
    // Convert xVIO types to Eigen
    // TODO(jeff): sync all types so this can be removed
    x::Quaternion quat;
    quat.x() = attitudes[i].ax;
    quat.y() = attitudes[i].ay;
    quat.z() = attitudes[i].az;
    quat.w() = attitudes[i].aw;
    quat = quat.normalized();
    rotations_[i] = quat.toRotationMatrix().transpose();
    
    Eigen::Vector3d pos(translations[i].tx,
                        translations[i].ty,
                        translations[i].tz);
    positions_[i] = pos;

    // Compute projection matrices
    pose2proj(rotations_[i], positions_[i], projs_[i]); 
  }
}

void Triangulation::triangulateDlt(const x::Feature& obs1,
                                   const x::Feature& obs2,
                                   const int i1,
                                   const int i2,
                                   Eigen::Vector3d& pt_xyz) const
{
  // Feature image coordinates
  cv::Mat obs1_cv = (cv::Mat_<double>(2, 1) << obs1.getX(), obs1.getY());
  cv::Mat obs2_cv = (cv::Mat_<double>(2, 1) << obs2.getX(), obs2.getY());

  // Feature triangulation with OpenCV Direct Linear Transform
  cv::Mat pt_h; // homogeneous coordinates
  cv::triangulatePoints(projs_[i1], projs_[i2], obs1_cv, obs2_cv, pt_h);

  // Feature cartesian coordinates (from OpenCV's homogeneous output)
  const double x = pt_h.at<double>(0) / pt_h.at<double>(3);
  const double y = pt_h.at<double>(1) / pt_h.at<double>(3);
  const double z = pt_h.at<double>(2) / pt_h.at<double>(3);
  pt_xyz = Eigen::Vector3d(x,y,z);
}

void Triangulation::triangulateGN(const x::Track& track,
                                  Eigen::Vector3d& pt_ivd) const
{
  /********************** Linear triangulation *********************/

  // Indices of the first and last observations of that feature in
  // the pose vectors
  const int n_obs = track.size();
  const int i2 = n_poses_ - 1;
  const int i1 = i2 - n_obs + 1;
  
  // First and last feature observations
  const x::Feature& obs1 = track[i1];
  const x::Feature& obs2 = track[i2];

  // Initial 2-view triangulation
  Eigen::Vector3d pt_xyz;
  triangulateDlt(obs1, obs2, i1, i2, pt_xyz);

  /*************************** Initialization ***********************/

  // Homogenous coordinates in world frame
  const cv::Mat pt_w_h =
    (cv::Mat_<double>(4, 1) << pt_xyz(0),
                               pt_xyz(1),
                               pt_xyz(2),
                               1);

  // Homogenous coordinates in last camera frame
  const cv::Mat pt_c2 = projs_[i2] * pt_w_h;
 
  // Inverse-depth coordinates initialization
  double alpha = pt_c2.at<double>(0) / pt_c2.at<double>(2);
  double beta  = pt_c2.at<double>(1) / pt_c2.at<double>(2);
  double rho   = 1.0 / pt_c2.at<double>(2);
  
  // Anchor frame for inverse-depth: last frame
  const Eigen::Matrix3d rot_a = rotations_[i2];
  const Eigen::VectorXd p_a   = positions_[i2];

  // Iteration variables
  const size_t n_meas = 2 * n_poses_;
  double r_norm_last = 1000.0;
  double r_norm = 100.0;
  unsigned int iter = 0;
  const Eigen::Vector3d u_x(1.0, 0.0, 0.0); // x axis
  const Eigen::Vector3d u_y(0.0, 1.0, 0.0); // y axis
  
  /************************* Iterations ****************************/

  // While the termination criterion is not reached
  while (r_norm_last - r_norm > term_)
  {
    // Check nb of iterations <= max
    iter++;
    if (iter > max_iter_)
      break;

    Eigen::VectorXd r = Eigen::VectorXd::Zero(n_meas);
    Eigen::MatrixXd j = Eigen::MatrixXd::Zero(n_meas, 3);

    // Loop over camera frames
    for (int i = i1; i <= i2; i++)
    {
      const Eigen::Matrix3d rot = rotations_[i];
      const Eigen::Matrix3d delta_rot_i2 = rot * rot_a.transpose();

      const Eigen::VectorXd p = positions_[i];
      const Eigen::Vector3d delta_pos_i2 = rot * p_a - rot * p;

      // Measurement
      const int i_trk = i - i1;
      Eigen::Vector2d h_meas(track[i_trk].getX(),
                             track[i_trk].getY());

      // Predicted measurement
      Eigen::Vector3d h_i = 
        delta_rot_i2 * Eigen::Vector3d(alpha, beta, 1)
        + rho * delta_pos_i2;
      Eigen::Vector2d h( h_i(0)/h_i(2), h_i(1)/h_i(2)) ; 

      // Residual
      r.segment(i_trk * 2, 2) = h_meas - h;

      // Jacobians
      Eigen::Vector3d j_alpha = delta_rot_i2 * u_x;
      Eigen::Vector3d j_beta  = delta_rot_i2 * u_y;
      Eigen::Vector3d j_rho   = delta_pos_i2;

      Eigen::Matrix3d j0;
      j0 << j_alpha, j_beta, j_rho;
      
      Eigen::MatrixXd j1(2,3);
      j1 << -1.0 / h_i(2), 0.0        , h_i(0) / std::pow(h_i(2), 2),
            0.0          , -1 / h_i(2), h_i(1) / std::pow(h_i(2), 2);

      j.block(i_trk * 2, 0, 2, 3) = j1 * j0;
    }

    // Delta correction
    const Eigen::Vector3d delta = (j.transpose() * j).inverse() * j.transpose() * r;
    alpha = alpha - delta(0);
    beta  = beta - delta(1);
    rho   = rho - delta(2);
    
    // Save residuals for termination criterion
    r_norm_last = r_norm;
    r_norm = r.norm();
  }
  
  // Output
  pt_ivd = Eigen::Vector3d(alpha, beta, rho);
}

void Triangulation::pose2proj(const Eigen::Matrix3d& rot,
                              const Eigen::Vector3d& pos,
                              cv::Mat& proj) const
{
  // Construct projection matrix with Eigen
  Eigen::MatrixXd proj_tmp(3,4);
  proj_tmp << rot, -rot * pos;
  
  // Convert to cv::Mat
  cv::eigen2cv(proj_tmp, proj);
}
