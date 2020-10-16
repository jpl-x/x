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

#include <x/vio/range_update.h>
#include <x/vio/tools.h>
#include <x/ekf/state.h>
#include <boost/math/distributions.hpp>

using namespace x;
using namespace Eigen;

RangeUpdate::RangeUpdate(const x::RangeMeasurement& range_meas,
                         const std::vector<int>& tr_feat_ids,
                         const x::AttitudeList& quats,
                         const x::TranslationList& poss,
                         const MatrixXd& feature_states,
                         const std::vector<int>& anchor_idxs,
                         const MatrixXd& cov_s,
                         const int n_poses_max,
                         const double sigma_range)
{
  // Initialize Kalman update matrices
  const size_t cols = cov_s.cols();
  jac_ = MatrixXd::Zero(1, cols);
  cov_m_diag_ = VectorXd::Ones(1);
  res_ = MatrixXd::Zero(1, 1);
  
  // Compute residual, Jacobian and covariance block
  processRangedFacet(range_meas,
                     tr_feat_ids,
                     quats,
                     poss,
                     feature_states,
                     anchor_idxs,
                     cov_s,
                     n_poses_max,
                     sigma_range);
  /*processRangedFeature(range_meas,
                         quats,
                         poss,
                         feature_states,
                         P,
                         n_poses_max,
                         sigma_range,
                         0);*/
}

void RangeUpdate::processRangedFacet(const x::RangeMeasurement& range_meas,
                                     const std::vector<int>& tr_feat_ids,
                                     const x::AttitudeList& C_q_G,
                                     const x::TranslationList& G_p_C,
                                     const MatrixXd& feature_states,
                                     const std::vector<int>& anchor_idxs,
                                     const MatrixXd& P,
                                     const int n_poses_max,
                                     const double sigma_range)
{
  /********************************************************
   * Cartesian world coordinates of the triangle features *
   ********************************************************/
 
  // Array storing the  of the three features
  std::array<Vector3d, 3> G_p_fj; // world position
  std::array<Vector3d, 3> G_p_Ca; // anchor position
  std::array<x::Quaternion, 3> Ca_q_G; // anchor quaternion
  std::array<int, 3> anchor_idx; // anchor quaternion
  std::array<double, 3> alpha; // Normalized undistored image coordinates X
  std::array<double, 3> beta; // Normalized undistored image coordinates Y
  std::array<double, 3> rho; // Inverse depth

  for(unsigned int j=0; j<3; j++)
  {
    // Inverse-depth parameters in last observation frame
    alpha[j] = feature_states(tr_feat_ids[j] * 3, 0);
    beta[j] = feature_states(tr_feat_ids[j] * 3 + 1, 0);
    rho[j] = feature_states(tr_feat_ids[j] * 3 + 2, 0);

    // Anchor pose
    anchor_idx[j] = anchor_idxs[tr_feat_ids[j]];
    Ca_q_G[j].x() = C_q_G[anchor_idx[j]].ax;
    Ca_q_G[j].y() = C_q_G[anchor_idx[j]].ay;
    Ca_q_G[j].z() = C_q_G[anchor_idx[j]].az;
    Ca_q_G[j].w() = C_q_G[anchor_idx[j]].aw;

    const Vector3d anchor_pos(G_p_C[anchor_idx[j]].tx, G_p_C[anchor_idx[j]].ty, G_p_C[anchor_idx[j]].tz);
    G_p_Ca[j] = anchor_pos;

    // Coordinate of feature in global frame
    G_p_fj[j] = 1 / (rho[j])*Ca_q_G[j].normalized().toRotationMatrix() * Vector3d(alpha[j], beta[j], 1) + G_p_Ca[j];
  }

  /***********************
   * Current camera pose *
   ***********************/
  
  // Orientation
  x::Attitude Cn_q_G(C_q_G.back());
  x::Quatern attitude_to_quaternion;
  x::Quaternion Ci_q_G  = attitude_to_quaternion(Cn_q_G);

  // Position
  x::Translation G_p_Cn(G_p_C.back());
  Vector3d G_p_Ci(G_p_Cn.tx, G_p_Cn.ty, G_p_Cn.tz);

  /************
   * Residual *
   ************/
  
  // Compute normal vector to the triangle plane in world coordinates
  const Vector3d G_n = (G_p_fj[0] - G_p_fj[1]).cross(G_p_fj[2] - G_p_fj[1]);

  // Undistorted normalized homogeneous 2D coordinates of the LRF impact point on the ground
  const Vector3d lrf_img_pt_nh(range_meas.img_pt_n.getX(), range_meas.img_pt_n.getY(), 1.0);

  // Measurement prediction
  const double a = (G_p_fj[1] - G_p_Ci).dot(G_n);
  const double b = lrf_img_pt_nh.dot(Ci_q_G.normalized().toRotationMatrix().transpose() * G_n);
  double range_hat = a/b;

  // Measurement
  const double range(range_meas.range);

  // Actual residual
  MatrixXd res_j(MatrixXd::Zero(1, 1));
  res_j(0, 0) = range - range_hat;

  /*******************************
   * Measurement Jacobian matrix *
   *******************************/

  const size_t cols = P.cols();
  MatrixXd h_j(MatrixXd::Zero(1, cols));
  
  // Camera Position
  const RangeJacobian J_pc = - 1.0 / b * G_n.transpose();

  // Camera attitude
  const RangeJacobian J_qc = a / std::pow(b, 2.0)
    * G_n.transpose() * Ci_q_G.normalized().toRotationMatrix() 
    * x::Skew(lrf_img_pt_nh(0), lrf_img_pt_nh(1), lrf_img_pt_nh(2)).matrix;
  
  // Position of the LRF target point in world frame
  const Vector3d G_p_r = a / b * Ci_q_G.normalized().toRotationMatrix() * lrf_img_pt_nh + G_p_Ci;
 
  // Barycenter
  const Vector3d G_p_bary = 1.0/3.0 * (G_p_fj[0]+G_p_fj[1]+G_p_fj[2]);
  /* Triangle feature 1 */
  // Jacobian wrt cartesian feature coordinates
  const RangeJacobian J_f0 = 1.0 / b * (1.0/3.0 * G_n + (G_p_fj[2] - G_p_fj[1]).cross(G_p_bary - G_p_r)).transpose();
  // Anchor position
  const RangeJacobian J_pc0 = J_f0;
  // Anchor attitude 
  const RangeJacobian J_qc0 = - 1.0 / rho[0] * J_f0 * Ca_q_G[0].normalized().toRotationMatrix()
    * x::Skew(alpha[0], beta[0], 1.0).matrix;
  // Inverse-depth feature coordinates
  MatrixXd mat(MatrixXd::Identity(3, 3));
  mat(0, 2) = - alpha[0] / rho[0];
  mat(1, 2) = - beta[0] / rho[0];
  mat(2, 2) =  - 1 / rho[0];
  RangeJacobian J_fi0 = 1 / rho[0] * J_f0 * Ca_q_G[0].normalized().toRotationMatrix() * mat;

  /* Triangle feature 2 */
  // Jacobian wrt cartesian feature coordinates
  const RangeJacobian J_f1 = 1.0 / b * (1.0/3.0 * G_n + (G_p_fj[0] - G_p_fj[2]).cross(G_p_bary - G_p_r)).transpose();
  // Anchor position
  const RangeJacobian J_pc1 = J_f1;
  // Anchor attitude 
  const RangeJacobian J_qc1 = - 1.0 / rho[1] * J_f1 * Ca_q_G[1].normalized().toRotationMatrix()
    * x::Skew(alpha[1], beta[1], 1.0).matrix;
  // Inverse-depth feature coordinates
  mat = MatrixXd::Identity(3, 3);
  mat(0, 2) = - alpha[1] / rho[1];
  mat(1, 2) = - beta[1] / rho[1];
  mat(2, 2) = - 1 / rho[1];
  RangeJacobian J_fi1 = 1 / rho[1] * J_f1 * Ca_q_G[1].normalized().toRotationMatrix() * mat;

  // Triangle feature 3
  // Jacobian wrt cartesian feature coordinates
  const RangeJacobian J_f2 = 1.0 / b * (1.0/3.0 * G_n + (G_p_fj[1] - G_p_fj[0]).cross(G_p_bary - G_p_r)).transpose();
  // Anchor position
  const RangeJacobian J_pc2 = J_f2;
  // Anchor attitude 
  const RangeJacobian J_qc2 = - 1.0 / rho[2] * J_f2 * Ca_q_G[2].normalized().toRotationMatrix()
    * x::Skew(alpha[2], beta[2], 1.0).matrix;
  // Inverse-depth feature coordinates
  mat = MatrixXd::Identity(3, 3);
  mat(0, 2) = - alpha[2] / rho[2];
  mat(1, 2) = - beta[2] / rho[2];
  mat(2, 2) = - 1 / rho[2];
  RangeJacobian J_fi2 = 1 / rho[2] * J_f2 * Ca_q_G[2].normalized().toRotationMatrix() * mat;

  // Update stacked Jacobian matrices associated to the current feature
  const unsigned int row = 0;
  const unsigned int pos = C_q_G.size()- 1;
  unsigned int col = pos * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_pc;

  col += n_poses_max * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_qc;

  col = anchor_idx[0] * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) =
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) + J_pc0;
  col += n_poses_max * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = 
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) + J_qc0;
  col = (n_poses_max * 2 + tr_feat_ids[0]) * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_fi0;
 
  col = anchor_idx[1] * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) =
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) + J_pc1; 
  col += n_poses_max * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = 
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) + J_qc1;
  col = (n_poses_max * 2 + tr_feat_ids[1]) * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_fi1;

  col = anchor_idx[2] * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) =
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) + J_pc2;
  col += n_poses_max * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = 
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) + J_qc2;
  col = (n_poses_max * 2 + tr_feat_ids[2]) * kJacCols;
  h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_fi2;

  //==========================================================================
  // Outlier rejection
  //==========================================================================
  VectorXd r_j(1);
  const double var_range = sigma_range * sigma_range;
  r_j << var_range;
  MatrixXd S_inv = (h_j * P * h_j.transpose() + r_j).inverse();
  MatrixXd gamma = res_j.transpose() * S_inv * res_j;
  boost::math::chi_squared_distribution<> my_chisqr(1);
  double chi = quantile(my_chisqr, 0.9);  // 95-th percentile

  if (gamma(0, 0) < chi)  // Inlier
  {
    jac_.block(0,               // startRow
             0,               // startCol
             1,               // numRows
             cols) = h_j;  // numCols

    // Residual vector
    res_.block(0, 0, 1, 1) = res_j;

    // Measurement covariance matrix
    cov_m_diag_ = r_j;
  }
}

void RangeUpdate::processRangedFeature(const x::RangeMeasurement& range_meas,
                                       const x::AttitudeList& C_q_G,
                                       const x::TranslationList& G_p_C,
                                       const MatrixXd& feature_states,
                                       const std::vector<int>& anchor_idxs,
                                       const MatrixXd& P,
                                       const int n_poses_max,
                                       const double sigma_range,
                                       const size_t& j)
{
  const size_t cols = P.cols();
  MatrixXd h_j(MatrixXd::Zero(1, cols));
  MatrixXd res_j(MatrixXd::Zero(1, 1));

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

  //==========
  // Residual
  //==========
  double range(range_meas.range);
  double range_hat(Ci_p_fj(2));
  res_j(0, 0) = range - range_hat;

  //============================
  // Measurement Jacobian matrix
  //============================
  const unsigned int pos = C_q_G.size() - 1;
  if (anchor_idx == pos)  // Handle special case
  {
    // Inverse-depth feature coordinates jacobian
    RangeJacobian mat(MatrixXd::Zero(1, 3));
    mat(0, 2) = -1 / std::pow(rho, 2);

    // Update stacked Jacobian matrices associated to the current feature
    unsigned int row = 0;
    unsigned int col = (n_poses_max * 2 + j) * kJacCols;
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = mat;
  }
  else
  {
    // Set Jacobian of pose for i'th measurement of feature j (eq.22, 23)
    RangeJacobian J_i(MatrixXd::Zero(1, 3));
    J_i(0, 2) = 1.0;

    // Attitude
    Vector3d skew_vector = Ci_q_G_.normalized().toRotationMatrix().transpose() * (G_p_fj - G_p_Ci_);
    RangeJacobian J_attitude = J_i * x::Skew(skew_vector(0), skew_vector(1), skew_vector(2)).matrix;

    // Position
    RangeJacobian J_position = -J_i * Ci_q_G_.normalized().toRotationMatrix().transpose();

    // Anchor attitude
    RangeJacobian J_anchor_att = -1 / rho * J_i * Ci_q_G_.normalized().toRotationMatrix().transpose() *
                                         Ca_q_G.normalized().toRotationMatrix() * x::Skew(alpha, beta, 1).matrix;

    // Anchor position
    RangeJacobian J_anchor_pos = -J_position;

    // Inverse-depth feature coordinates
    MatrixXd mat(MatrixXd::Identity(3, 3));
    mat(0, 2) = -alpha / rho;
    mat(1, 2) = -beta / rho;
    mat(2, 2) = -1 / rho;
    RangeJacobian Hf_j1 = 1 / rho * J_i * Ci_q_G_.normalized().toRotationMatrix().transpose() *
                                  Ca_q_G.normalized().toRotationMatrix() * mat;

    // Update stacked Jacobian matrices associated to the current feature
    unsigned int row = 0;
    const unsigned int pos = C_q_G.size() - 1;
    unsigned int col = pos * kJacCols;
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_position;

    col += n_poses_max * kJacCols;
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_attitude;

    col = anchor_idx * kJacCols;
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_anchor_pos;

    col += n_poses_max * kJacCols;
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = J_anchor_att;

    col = (n_poses_max * 2 + j) * kJacCols;
    h_j.block<1, kJacCols>(row, kSizeCoreErr + col) = Hf_j1;
  }

  //==========================================================================
  // Outlier rejection
  //==========================================================================
  VectorXd r_j(1);
  const double var_range = sigma_range * sigma_range;
  r_j << var_range;
  // MatrixXd S_inv = (h_j * P * h_j.transpose() + R_j).inverse();
  // MatrixXd gamma = res_j.transpose() * S_inv * res_j;
  // chi_squared_distribution<> my_chisqr(2 * features[j].size());
  // double chi = quantile(my_chisqr, 0.9);  // 95-th percentile

  if (true)  // gamma(0, 0) < chi)  // Inlier
  {
    jac_.block(0,               // startRow
             0,               // startCol
             1,               // numRows
             cols) = h_j;  // numCols

    // Residual vector
    res_.block(0, 0, 1, 1) = res_j;

    // Measurement covariance matrix
    cov_m_diag_ = r_j;
  }
}
