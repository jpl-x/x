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

#include <x/vio/slam_update.h>
#include <x/vio/tools.h>
#include <x/ekf/state.h>
#include <boost/math/distributions.hpp>

using namespace x;
using namespace Eigen;

SlamUpdate::SlamUpdate(const x::TrackList& trks,
                       const x::AttitudeList& quats,
                       const x::TranslationList& poss,
                       const MatrixXd& feature_states,
                       const std::vector<int>& anchor_idxs,
                       const MatrixXd& cov_s,
                       const int n_poses_max,
                       const double sigma_img)
{
  // Number of features
  const size_t n_trks = trks.size();

  // Initialize Kalman update matrices
  const size_t rows = 2 * n_trks;
  const size_t cols = cov_s.cols();
  jac_ = MatrixXd::Zero(rows, cols);
  cov_m_diag_ = VectorXd::Ones(rows);
  res_ = MatrixXd::Zero(rows, 1);
  
  // For each track, compute residual, Jacobian and covariance block
  const double var_img = sigma_img * sigma_img;
  for (size_t i = 0, row_h = 0; i < n_trks; ++i) {
    processOneTrack(trks[i],
                    quats,
                    poss,
                    feature_states,
                    anchor_idxs,
                    cov_s,
                    n_poses_max,
                    var_img,
                    i,
                    row_h);
  }
}

void SlamUpdate::processOneTrack(const x::Track& track,
                                 const x::AttitudeList& C_q_G,
                                 const x::TranslationList& G_p_C,
                                 const MatrixXd& feature_states,
                                 const std::vector<int>& anchor_idxs,
                                 const MatrixXd& P,
                                 const int n_poses_max,
                                 const double var_img,
                                 const size_t& j,
                                 size_t& row_h)
{
  const size_t cols = P.cols();
  MatrixXd h_j(MatrixXd::Zero(2, cols));
  MatrixXd Hf_j(MatrixXd::Zero(2, cols));
  MatrixXd res_j(MatrixXd::Zero(2, 1));
 
  //==========================================================================
  // Feature information
  //==========================================================================
  // A-priori inverse-depth parameters in last observation frame
  double alpha = feature_states(j * 3, 0);
  double beta = feature_states(j * 3 + 1, 0);
  double rho = feature_states(j * 3 + 2, 0);

  // Anchor pose
  unsigned int anchor_idx = anchor_idxs[j];
  x::Quaternion Ca_q_G;
  Ca_q_G.x() = C_q_G[anchor_idx].ax;
  Ca_q_G.y() = C_q_G[anchor_idx].ay;
  Ca_q_G.z() = C_q_G[anchor_idx].az;
  Ca_q_G.w() = C_q_G[anchor_idx].aw;

  Vector3d G_p_Ca(G_p_C[anchor_idx].tx, G_p_C[anchor_idx].ty, G_p_C[anchor_idx].tz);

  // Coordinate of feature in global frame
  Vector3d G_p_fj = 1 / (rho)*Ca_q_G.normalized().toRotationMatrix() * Vector3d(alpha, beta, 1) + G_p_Ca;

  // FOR LAST FEATURE OBSERVATION
  x::Translation G_p_Cn(G_p_C.back());
  x::Attitude Cn_q_G(C_q_G.back());

  x::Quatern attitude_to_quaternion;
  Quaterniond Ci_q_G_ = attitude_to_quaternion(Cn_q_G);
  Vector3d G_p_Ci_(G_p_Cn.tx, G_p_Cn.ty, G_p_Cn.tz);

  // Feature position expressed in camera frame.
  Vector3d Ci_p_fj;
  Ci_p_fj << Ci_q_G_.normalized().toRotationMatrix().transpose() * (G_p_fj - G_p_Ci_);

  // eq. 20(a)
  Vector2d z;
  const size_t track_size(track.size());
  const unsigned int i = track_size - 1;
  z(0) = track[i].getX();
  z(1) = track[i].getY();

  Vector2d z_hat(z);
  assert(Ci_p_fj(2));
  z_hat(0) = Ci_p_fj(0) / Ci_p_fj(2);
  z_hat(1) = Ci_p_fj(1) / Ci_p_fj(2);

  // eq. 20(b)
  res_j(0, 0) = z(0) - z_hat(0);
  res_j(1, 0) = z(1) - z_hat(1);

  //============================
  // Measurement Jacobian matrix
  //============================

  const unsigned int pos = C_q_G.size() - 1;
  if (anchor_idx == pos)  // Handle special case
  {
    // Inverse-depth feature coordinates jacobian
    MatrixXd mat(MatrixXd::Zero(2, 3));
    mat(0, 0) = 1.0;
    mat(1, 1) = 1.0;

    // Update stacked Jacobian matrices associated to the current feature
    unsigned int row = 0;
    unsigned int col = (n_poses_max * 2 + j) * kJacCols;
    h_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = mat;
  }
  else
  {
    // Set Jacobian of pose for i'th measurement of feature j (eq.22, 23)
    VisJacBlock J_i(VisJacBlock::Zero());
    // first row
    J_i(0, 0) = 1.0 / Ci_p_fj(2);
    J_i(0, 1) = 0.0;
    J_i(0, 2) = -Ci_p_fj(0) / std::pow((double)Ci_p_fj(2), 2);
    // second row
    J_i(1, 0) = 0.0;
    J_i(1, 1) = 1.0 / Ci_p_fj(2);
    J_i(1, 2) = -Ci_p_fj(1) / std::pow((double)Ci_p_fj(2), 2);

    // Attitude
    Vector3d skew_vector = Ci_q_G_.normalized().toRotationMatrix().transpose() * (G_p_fj - G_p_Ci_);
    VisJacBlock J_attitude = J_i * x::Skew(skew_vector(0), skew_vector(1), skew_vector(2)).matrix;

    // Position
    VisJacBlock J_position = -J_i * Ci_q_G_.normalized().toRotationMatrix().transpose();

    // Anchor attitude
    VisJacBlock J_anchor_att = -1 / rho * J_i * Ci_q_G_.normalized().toRotationMatrix().transpose() *
                                    Ca_q_G.normalized().toRotationMatrix() * x::Skew(alpha, beta, 1).matrix;

    // Anchor position
    VisJacBlock J_anchor_pos = -J_position;

    // Inverse-depth feature coordinates
    MatrixXd mat(MatrixXd::Identity(3, 3));
    mat(0, 2) = -alpha / rho;
    mat(1, 2) = -beta / rho;
    mat(2, 2) = -1 / rho;
    VisJacBlock Hf_j1 = 1 / rho * J_i * Ci_q_G_.normalized().toRotationMatrix().transpose() *
                             Ca_q_G.normalized().toRotationMatrix() * mat;

    // Update stacked Jacobian matrices associated to the current feature
    unsigned int row = 0;
    const unsigned int pos = C_q_G.size() - 1;
    unsigned int col = pos * kJacCols;
    h_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_position;

    col += n_poses_max * kJacCols;
    h_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_attitude;

    col = anchor_idx * kJacCols;
    h_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_anchor_pos;

    col += n_poses_max * kJacCols;
    h_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_anchor_att;

    col = (n_poses_max * 2 + j) * kJacCols;
    h_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = Hf_j1;
  }

  //==========================================================================
  // Outlier rejection
  //==========================================================================
  const VectorXd r_j_diag = var_img * VectorXd::Ones(2); 
  MatrixXd r_j = var_img * MatrixXd::Identity(2, 2);
  MatrixXd S_inv = (h_j * P * h_j.transpose() + r_j).inverse();
  MatrixXd gamma = res_j.transpose() * S_inv * res_j;
  boost::math::chi_squared_distribution<> my_chisqr(2 * track_size);
  double chi = quantile(my_chisqr, 0.9);  // 95-th percentile

  if (gamma(0, 0) < chi)  // Inlier
  {
    jac_.block(row_h,          // startRow
             0,               // startCol
             2,               // numRows
             cols) = h_j;  // numCols

    // Residual vector (one feature)
    res_.block(row_h, 0, 2, 1) = res_j;

    // Measurement covariance matrix
    cov_m_diag_.segment(row_h, 2) = r_j_diag;

    row_h += 2;
  }
}

void SlamUpdate::computeInverseDepthsNew(const x::TrackList& new_trks,
                                         const double rho_0,
                                         MatrixXd& ivds) const
{
  const size_t n_new_slam_std_trks = new_trks.size();
  ivds = MatrixXd::Zero(n_new_slam_std_trks * 3, 1);

  // For each standard SLAM feature to init
  for (size_t j = 0; j < n_new_slam_std_trks; ++j) {
    // Compute inverse-depth coordinates of new feature state
    // (last observation in track)
    computeOneInverseDepthNew(new_trks[j].back(), rho_0, j, ivds);
  }
}

void SlamUpdate::computeOneInverseDepthNew(const x::Feature& feature,
                                           const double rho_0,
                                           const unsigned int& idx,
                                           MatrixXd& ivds) const
{
  // Inverse-depth parameters anchored in last observation frame
  const double alpha = feature.getX();
  const double beta  = feature.getY();
  // const double rho = 1.0 / G_p_Ci.back().tz; // height-based init:

  ivds(3*idx)     = alpha;
  ivds(3*idx + 1) = beta;
  ivds(3*idx + 2) = rho_0;
}
