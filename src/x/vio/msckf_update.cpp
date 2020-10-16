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

#include <x/vio/msckf_update.h>
#include <x/vio/tools.h>
#include <x/ekf/state.h>
#include <boost/math/distributions.hpp>

using namespace x;
using namespace Eigen;

MsckfUpdate::MsckfUpdate(const x::TrackList& trks,
                         const x::AttitudeList& quats,
                         const x::TranslationList& pos,
                         const Triangulation& triangulator,
                         const MatrixXd& cov_s,
                         const int n_poses_max,
                         const double sigma_img)
{
  // Number of features
  const size_t n_trks = trks.size();

  // Number of feature observations
  size_t n_obs = 0;
  for(size_t i=0; i < n_trks; i++)
    n_obs += trks[i].size();

  // Initialize Kalman update matrices
  const size_t rows = 2 * n_obs - n_trks * 3;
  const size_t cols = cov_s.cols();
  jac_ = MatrixXd::Zero(rows, cols);
  cov_m_diag_ = VectorXd::Ones(rows);
  res_ = MatrixXd::Zero(rows, 1);
  
  // For each track, compute residual, Jacobian and covariance block
  const double var_img = sigma_img * sigma_img;
  for (size_t i = 0, row_h = 0; i < n_trks; ++i) {
    processOneTrack(trks[i], 
                    quats,
                    pos,
                    triangulator,
                    cov_s,
                    n_poses_max,
                    var_img,
                    i,
                    row_h);
  }
}

void MsckfUpdate::processOneTrack(const x::Track& track,
                                  const x::AttitudeList& C_q_G,
                                  const x::TranslationList& G_p_C,
                                  const Triangulation& triangulator,
                                  const MatrixXd& P,
                                  const int n_poses_max,
                                  const double var_img,
                                  const size_t& j,
                                  size_t& row_h)
{
  // Initialization
  const size_t track_size = track.size();
  unsigned int rows_track_j = track_size * 2;
  const size_t cols = P.cols();
  MatrixXd jac_j(MatrixXd::Zero(rows_track_j, cols));
  MatrixXd Hf_j(MatrixXd::Zero(jac_j.rows(), kJacCols));
  MatrixXd res_j(MatrixXd::Zero(rows_track_j, 1));

  // Triangulate feature j
  // Inverse-depth parameters in last observation frame
  Vector3d feature; // inverse-depth parameters in last observation frame
  triangulator.triangulateGN(track, feature);
  const double alpha = feature(0);
  const double beta  = feature(1);
  const double rho   = feature(2);

  // Coordinate of feature in global frame
  x::Quaternion Cn_q_G;
  Cn_q_G.x() = C_q_G.back().ax;
  Cn_q_G.y() = C_q_G.back().ay;
  Cn_q_G.z() = C_q_G.back().az;
  Cn_q_G.w() = C_q_G.back().aw;

  Vector3d G_p_Cn(G_p_C.back().tx, G_p_C.back().ty, G_p_C.back().tz);

  Vector3d G_p_fj = 1 / (rho)*Cn_q_G.normalized().toRotationMatrix() * Vector3d(alpha, beta, 1) + G_p_Cn;

  x::Quatern attitude_to_quaternion;

  // LOOP OVER ALL FEATURE OBSERVATIONS
  for (size_t i = 0; i < track_size; ++i)
  {
    const unsigned int pos = C_q_G.size() - track_size + i;

    Quaterniond Ci_q_G_ = attitude_to_quaternion(C_q_G[pos]);
    Vector3d G_p_Ci_(G_p_C[pos].tx, G_p_C[pos].ty, G_p_C[pos].tz);

    // Feature position expressed in camera frame.
    Vector3d Ci_p_fj;
    Ci_p_fj << Ci_q_G_.normalized().toRotationMatrix().transpose() * (G_p_fj - G_p_Ci_);

    // eq. 20(a)
    Vector2d z;
    z(0) = track[i].getX();
    z(1) = track[i].getY();

    Vector2d z_hat(z);
    assert(Ci_p_fj(2));
    z_hat(0) = Ci_p_fj(0) / Ci_p_fj(2);
    z_hat(1) = Ci_p_fj(1) / Ci_p_fj(2);

    // eq. 20(b)
    res_j(i * 2, 0) = z(0) - z_hat(0);
    res_j(i * 2 + 1, 0) = z(1) - z_hat(1);

    // Set Jacobian of pose for i'th measurement of feature j (eq.22, 23)
    VisJacBlock J_i = VisJacBlock::Zero();
    // first row
    J_i(0, 0) = 1.0 / Ci_p_fj(2);
    J_i(0, 1) = 0.0;
    J_i(0, 2) = -Ci_p_fj(0) / std::pow((double)Ci_p_fj(2), 2);
    // second row
    J_i(1, 0) = 0.0;
    J_i(1, 1) = 1.0 / Ci_p_fj(2);
    J_i(1, 2) = -Ci_p_fj(1) / std::pow((double)Ci_p_fj(2), 2);

    unsigned int row = i * kVisJacRows;

    // Measurement Jacobians wrt attitude, position and feature
    // Position
    VisJacBlock J_position = -J_i * Ci_q_G_.normalized().toRotationMatrix().transpose();

    Vector3d dP = Ci_q_G_.normalized().toRotationMatrix().transpose() * (G_p_fj - G_p_Ci_);

    // Attitude
    VisJacBlock J_attitude = J_i * x::Skew(dP(0), dP(1), dP(2)).matrix;

    // Feature
    Hf_j.block<kVisJacRows, kJacCols>(row, 0) = -J_position;

    // Update stacked Jacobian matrix associated to the current feature
    unsigned int col = pos * kJacCols;
    jac_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_position;

    col += n_poses_max * kJacCols;

    jac_j.block<kVisJacRows, kJacCols>(row, kSizeCoreErr + col) = J_attitude;
  }  // LOOP OVER ALL FEATURE OBSERVATIONS

  //========================================================================
  // Left nullspace projection
  //========================================================================
  // Nullspace computation
  MatrixXd q = Hf_j.householderQr().householderQ();
  MatrixXd A = x::MatrixBlock(q, 0, 3);

  // Projections
  MatrixXd res0_j = A.transpose() * res_j;
  MatrixXd jac0_j = A.transpose() * jac_j;

  // Noise measurement matrix
  const VectorXd r0_j_diag = var_img * VectorXd::Ones(rows_track_j - 3);
  const MatrixXd r0_j = r0_j_diag.asDiagonal();

  //==========================================================================
  // Outlier rejection
  //==========================================================================
  MatrixXd S_inv = (jac0_j * P * jac0_j.transpose() + r0_j).inverse();
  MatrixXd gamma = res0_j.transpose() * S_inv * res0_j;
  boost::math::chi_squared_distribution<> my_chisqr(2 * track_size - 3); // 2*Mj-3 DOFs
  double chi = quantile(my_chisqr, 0.95); // 95-th percentile

  if (gamma(0, 0) < chi)  // Inlier
  {
#ifdef VERBOSE
    inliers_.push_back(G_p_fj);
#endif

    jac_.block(row_h,            // startRow
             0,                 // startCol
             rows_track_j - 3,  // numRows
             cols) = jac0_j;    // numCols

    // Residual vector (for this track)
    res_.block(row_h, 0, rows_track_j - 3, 1) = res0_j;

    // Measurement covariance matrix diagonal
    cov_m_diag_.segment(row_h, rows_track_j - 3) = r0_j_diag;

    row_h += rows_track_j - 3;
  }
  else  // outlier
  {
#ifdef VERBOSE
    outliers_.push_back(G_p_fj);
#endif
  }
}
