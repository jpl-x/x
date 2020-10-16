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

#include <x/vio/state_manager.h>
#include <x/vio/tools.h>

using namespace x;

void StateManager::clear()
{
  n_poses_   = 0;
  n_features_ = 0;
  std::vector<x::Time> pose_times(n_poses_max_, -1);
  std::vector<int> persistent_features_anchor_idx(n_features_max_, -1);
  anchor_idxs_ = persistent_features_anchor_idx;

  stateHasBeenFilledBefore_ = false;
}

void StateManager::manage(State& state,
                          std::vector<unsigned int> del_feat_idx) {
  // Retrieve current sliding-windows poses
  Eigen::VectorXd att = state.getOrientationArray();
  Eigen::VectorXd pos = state.getPositionArray();

  // Retrieve latest state estimate
  Quaternion cae = state.computeCameraOrientation();
  Vector3 cpe = state.computeCameraPosition();

  // Retrieve latest covariance
  Matrix cov = state.getCovariance();

  // Access the current feature state vector and initialize the new one
  Eigen::VectorXd new_features = state.getFeatureArray();

  //============================================================================
  // Persistent feature removal
  //============================================================================

  // For each feature to remove
  std::sort (del_feat_idx.begin(), del_feat_idx.end());
  size_t i = del_feat_idx.size();
  while(i)
  {
    // Current feature index the feature states
    const unsigned int idx = del_feat_idx[i-1];

    //=====================
    // State vector removal
    //=====================
    // Remove it from the feature state vector and
    // shift upward those coming after it
    const unsigned int n1 = n_features_ - idx - 1; // number of coming features after it
    new_features.block(idx * 3, 0, n1 * 3, 1) =
        new_features.block((idx + 1) * 3, 0, n1 * 3, 1);
    new_features.block((n_features_ - 1) * 3, 0, 3, 1) << 0u, 0u, 0u;

    //===============
    // Anchor removal
    //===============
    anchor_idxs_.erase(anchor_idxs_.begin() + idx);
    anchor_idxs_.push_back(-1);

    //==========================
    // Covariance matrix removal
    //==========================

    // Actual size of covariance matrix
    const size_t n = cov.rows();
    // Starting index of row/col of the feature to be deleted
    const unsigned int idx0 = 21 + n_poses_max_ * 6 + idx * 3;
    // Starting index of row/col after the feature to be deleted
    const unsigned int idx1 = idx0 + 3;
    // Number of active feature cols/rows after the one to be deleted
    const size_t dim0 = (n_features_max_-idx-1) * 3;

    // Draw the block matrix makes it easier the understand the following
    // Retrieve columns after the feature to be removed
    Matrix cols_after = cov.block(0,
                                           idx1,
                                           n,
                                           dim0);

    // Remove the rows corresponding to the removed feature in them
    cols_after.block(idx0,
                     0,
                     dim0,
                     dim0)
                         = cols_after.block(idx1,
                                            0,
                                            dim0,
                                            dim0);

    // Retrieve rows after the feature to be removed
    // No need to retrieve the column block after the feature as it is already
    // stored in cols_after
    Matrix rows_after = cov.block(idx1,
                                           0,
                                           dim0,
                                           idx0);

    // Shift cols_after and rows_after one index up in the covariance matrix
    cov.block(0,
              idx0,
              n,
              dim0) = cols_after;
    cov.block(idx0,
              0,
              dim0,
              idx0) = rows_after;

    // Replace the last 3 columns and rows of the covariance matrix with zeros
    cov.block(0, n-3, n, 3) = Matrix::Zero(n, 3);
    cov.block(n-3, 0, 3, n) = Matrix::Zero(3, n);

    // Update counts
    i--;
		n_features_--;
  }

  //============================================================================
  // Sliding window and persistent feature reparametrization
  //============================================================================

  // If there is no slot available in the sliding window, slide it up
  if (n_poses_ == n_poses_max_)
  {
    // Reparametrize persistent features anchored to the oldest pose
    reparametrizeFeatures(att, pos, new_features, cov);

    // Slide the window of poses
    slideWindow(att, pos, cov);
  }

  //============================================================================
  // Pose augmentation
  //============================================================================

  //TODO(jeff): Write a unit test looking for zeros in the sliding window
  // states. Can detect augmentation propagation errors.
  att.block<4, 1>(n_poses_ * 4, 0) << cae.x(), cae.y(), cae.z(), cae.w();
  pos.block<3, 1>(n_poses_ * 3, 0) << cpe.x(), cpe.y(), cpe.z();

  // Augment the state covariance matrix with the new pose
  augmentCovariance(state, n_poses_, cov);

  n_poses_++;

  //============================================================================
  // Finalize changes
  //============================================================================

  state.setCovariance(cov);
  state.setOrientationArray(att);
  state.setPositionArray(pos);
  state.setFeatureArray(new_features);
}

void StateManager::initMsckfSlamFeatures(State& state,
                                    	   const x::MsckfSlamMatrices& init_mats,
                                    	   const Matrix& correction,
                                         const double sigma_img)
{
	// Error covariance matrix
	Matrix P = state.getCovariance();

	// Update feature state using Li (2012)
	// Note: H2 is singular when camera is in perfect hovering
	// (and MSCKF cannot be applied)
	const Matrix H2_inv = init_mats.H2.inverse();
	const Matrix H2_inv_H1 = H2_inv * init_mats.H1;
	const Matrix new_features = init_mats.features
      - H2_inv_H1 * correction + H2_inv * init_mats.r1;

	// Compute new covariance blocks
  const double var_img = sigma_img * sigma_img;
  Matrix P_cross_new = - H2_inv_H1 * P;
  Matrix P_diag_new = H2_inv_H1 * P * H2_inv_H1.transpose()
      + var_img * H2_inv * H2_inv.transpose();

	// Add to state and covariance
	addFeatureStates(state, new_features, P_diag_new, P_cross_new);  
}

void StateManager::initStandardSlamFeatures(State& state,
																		  			const Matrix& new_features,
                                            const double sigma_img,
                                            const double sigma_rho_0)
{
  const size_t n_new_states ( new_features.size() );
	const size_t n ( state.getCovariance().rows() );
	
	// No correlation between new feature states and other states
	Matrix P_cross_new = Matrix::Zero(n_new_states, n);

  // Compute initial feature variances from standard deviations
  const double var_img = sigma_img * sigma_img;
  const double var_rho_0 = sigma_rho_0 * sigma_rho_0; 

	// Inverse-depth set like in Civera, alpha and beta follow image noise
  Matrix P_diag_new = var_img
      * Matrix::Identity(n_new_states, n_new_states);
  for(unsigned int i=0; i < n_new_states / 3; i++)
	  P_diag_new(3*i + 2, 3*i + 2) = var_rho_0;

	// Add to state and covariance
	addFeatureStates(state, new_features, P_diag_new, P_cross_new);  
}

void StateManager::addFeatureStates(State& state,
																	  const Matrix& new_features,
																	  const Matrix& cov,
																	  const Matrix& cross)
{
	// Determine number of new states
	const size_t n_new_states = new_features.size();

	// Append new feature at the end of the state vector
  Eigen::VectorXd features = state.getFeatureArray();
	assert(n_features_ < n_features_max_);
  features.block(n_features_ * 3, 0, n_new_states, 1) = new_features;
  state.setFeatureArray(features);

  // Augment covariance matrix
	Matrix& P =state.getCovarianceRef();
  const size_t n = P.rows();
  const size_t n_states = 21 + n_poses_max_ * 6 + n_features_ * 3;
  P.block(n_states, 0, n_new_states, n) = cross;
  P.block(0, n_states, n, n_new_states) = cross.transpose();
  P.block(n_states, n_states, n_new_states, n_new_states) = cov;
  
  // Add current pose as anchor the new SLAM features
  const size_t n_new_features( n_new_states / 3 );

	for(unsigned int i=0; i<n_new_features; i++)
		anchor_idxs_[n_features_ + i] = n_poses_ - 1;
  
  // Increment SLAM feature count
	n_features_ += n_new_features;
}

/** \brief Returns the cartesian coordinates of SLAM features for the input state
 */
std::vector<Eigen::Vector3d>
  StateManager::computeSLAMCartesianFeaturesForState(
    const State& state) const
{
  // Retrieve the required states
  const Matrix& atts = state.getOrientationArray();
  const Matrix& poss = state.getPositionArray();
  const Matrix& features = state.getFeatureArray();
  
  // Declare array to fill with the cartesian coordinates of the SLAM features
  std::vector<Eigen::Vector3d> features_xyz(n_features_);

  // For each persistent feature state
  for (unsigned int i = 0; i < n_features_; i++)
  {
    // Inverse-depth coordinates in anchor pose
    double alpha = features(3 * i, 0);
    double beta  = features(3 * i + 1, 0);
    double rho   = features(3 * i + 2, 0);

    // Index of the anchor in the pose states
    unsigned int idx = anchor_idxs_[i];

    // Anchor pose
    x::Quaternion Ci_q_G;
    Ci_q_G.x() = atts(4 * idx, 0);
    Ci_q_G.y() = atts(4 * idx + 1, 0);
    Ci_q_G.z() = atts(4 * idx + 2, 0);
    Ci_q_G.w() = atts(4 * idx + 3, 0);

    Vector3 G_p_Ci = poss.block<3, 1>(idx * 3, 0);

    // Convert to world coordinates
    Eigen::Vector3d G_p_fi =
        1 / (rho)
        * Ci_q_G.normalized().toRotationMatrix()
        * Eigen::Vector3d(alpha, beta, 1) + G_p_Ci;

    // Populate the array
   features_xyz[i] = G_p_fi; 
  }

  return features_xyz;
}

void StateManager::augmentCovariance(const State& state,
                                     const int pos,
                                     Matrix& covariance)
{
  Matrix jacobianCamPoseWrtStates;
  if (not stateHasBeenFilledBefore_) {
    jacobianCamPoseWrtStates = Matrix::Zero(21 + n_poses_max_ * 6 + n_features_max_ * 3,
                                            21 + n_poses_max_ * 6 + n_features_max_ * 3);
  } else {
    jacobianCamPoseWrtStates = Matrix::Identity(21 + n_poses_max_ * 6 + n_features_max_ * 3,
                                                21 + n_poses_max_ * 6 + n_features_max_ * 3);
  }

  // COVARIANCE AUGMENTATION
  // Unchanged: previous Core and cam position
  jacobianCamPoseWrtStates.block(0, 0, 21 + (pos + 1) * 3, 21 + (pos + 1) * 3) =
      Matrix::Identity(21 + (pos + 1) * 3, 21 + (pos + 1) * 3);

  // Unchanged: previous cam attitude
  jacobianCamPoseWrtStates.block(21 + n_poses_max_ * 3,
                21 + n_poses_max_ * 3,
                (pos + 1) * 3,
                (pos + 1) * 3)
    = Matrix::Identity((pos + 1) * 3, (pos + 1) * 3);

  // Unchanged: features
  jacobianCamPoseWrtStates.block(21 + n_poses_max_ * 6,
                                 21 + n_poses_max_ * 6,
                                 n_features_ * 3,
                                 n_features_ * 3)
    = Matrix::Identity(n_features_ * 3, n_features_ * 3);

  // cam_position::G_p_I
  // Derivative of camera position error wrt imu position error = I3x3
  jacobianCamPoseWrtStates.block(21 + pos*3, 0, 3, 3) = Matrix::Identity(3, 3);

  // Derivative of camera position error wrt imu orientation error = - C(qI2G) * [p_ic x]
  Vector3 pci = state.getPositionExtrinsics();
  jacobianCamPoseWrtStates.block(21 + pos*3, 6, 3, 3)
    = - state.getOrientation().normalized().toRotationMatrix() 
    * pci.toCrossMatrix();

  // Derivative of camera attitude error wrt imu orientation error = C(qI2C)
  jacobianCamPoseWrtStates.block(21 + n_poses_max_ * 3 + pos * 3, 6, 3, 3) =
      state.getOrientationExtrinsics().conjugate().normalized().toRotationMatrix();

  Matrix P_current = covariance;

  // Delete rows associated with position
  P_current.block(21 + pos * 3, 0, 3, P_current.cols()) =
      Matrix::Zero(3, P_current.cols());
  // Delete rows associated with attitude
  P_current.block(21 + n_poses_max_ * 3 + pos * 3, 0, 3, P_current.cols()) =
      Matrix::Zero(3, P_current.cols());

  // Delete cols associated with position
  P_current.block(0, 21 + pos * 3, P_current.rows(), 3) =
      Matrix::Zero(P_current.cols(), 3);
  // Delete cols associated with attitude
  P_current.block(0, 21 + n_poses_max_ * 3 + pos * 3, P_current.rows(), 3) =
      Matrix::Zero(P_current.cols(), 3);

  // If the next pose will be out of bounds
  if (pos + 1 == n_poses_max_)
    stateHasBeenFilledBefore_ = true;

  // Calculate the covariance incorporating the new camera poses
  P_current =
      jacobianCamPoseWrtStates *
      P_current *
      jacobianCamPoseWrtStates.transpose();

  covariance = P_current;
}

void StateManager::reparametrizeFeatures(Eigen::VectorXd const& atts_old,
                                         Eigen::VectorXd const& poss_old,
                                         Eigen::VectorXd& features,
                                         Matrix& covariance)
{
  // Retrieve oldest pose in the sliding window
  x::Quaternion Ci_q_G_old;
  Ci_q_G_old.x() = atts_old[0];
  Ci_q_G_old.y() = atts_old[1];
  Ci_q_G_old.z() = atts_old[2];
  Ci_q_G_old.w() = atts_old[3];

  Vector3 G_p_Ci_old = poss_old.block<3, 1>(0, 0);

  // Find indexes of features anchored to that pose
  std::vector<unsigned int> idx_to_chg;
  for (unsigned int i = 0; i < n_features_; i++)
  {
    if (anchor_idxs_[i] == 0)
    {
      idx_to_chg.push_back(i);
    }
  }

  // Initialize covariance reparametrization matrix
  const size_t n = 21 + n_poses_max_ * 6 + n_features_max_ * 3;
  Matrix J(Matrix::Identity(n,n));

  // For each persistent feature whose anchor has to be changed
  for (unsigned int i = 0; i < idx_to_chg.size(); i++)
  {
    //========================
    // State reparametrization
    //========================

    x::Quaternion Ci_q_G_new;
    const unsigned int idx1 = n_poses_max_ - 1;
    Ci_q_G_new.x() = atts_old[4 * idx1];
    Ci_q_G_new.y() = atts_old[4 * idx1 + 1];
    Ci_q_G_new.z() = atts_old[4 * idx1 + 2];
    Ci_q_G_new.w() = atts_old[4 * idx1 + 3];

    const Vector3 G_p_Ci_new1 = poss_old.block<3, 1>(idx1 * 3, 0);

    // Old feature parameters
    const unsigned int j = idx_to_chg[i];
    const double alpha_old = features[3 * j];
    const double beta_old  = features[3 * j + 1];
    const double rho_old   = features[3 * j + 2];

    // Compute new feature parameters, Eq. 38 in Li (RSS supplementals, 2012)
    const Vector3 new_params =
        Ci_q_G_new.normalized().toRotationMatrix().transpose()
        * (- G_p_Ci_new1 + G_p_Ci_old
           + 1 / rho_old
           * Ci_q_G_old.normalized().toRotationMatrix()
           * Eigen::Vector3d(alpha_old, beta_old, 1)
          );

    const double rho_new   = 1 / new_params[2];
    const double alpha_new = new_params[0] * rho_new;
    const double beta_new  = new_params[1] * rho_new;

    // Update feature states
    features.block<3, 1>(j * 3, 0) << alpha_new, beta_new, rho_new;

    // New anchor is the last pose in the window
    anchor_idxs_[j] = idx1;

    //====================================
    // Covariance matrix reparametrization
    //====================================

    // Old anchor attitude
    Eigen::Vector3d skew_vector =
        Eigen::Vector3d(alpha_old, beta_old, 1);
    const StateManager::Jacobian J_anchor_att_old =
        - 1 / rho_old
        * Ci_q_G_new.normalized().toRotationMatrix().transpose()
        * Ci_q_G_old.normalized().toRotationMatrix()
        * x::Skew(skew_vector(0),
                    skew_vector(1),
                    skew_vector(2)).matrix;

    // New anchor attitude
    skew_vector =
        Ci_q_G_new.normalized().toRotationMatrix().transpose() *
        (
        - G_p_Ci_new1 + G_p_Ci_old
        + 1 / rho_old
        * Ci_q_G_old.normalized().toRotationMatrix()
        * Eigen::Vector3d(alpha_old, beta_old, 1)
        );
    const StateManager::Jacobian J_anchor_att_new =
        x::Skew(skew_vector(0),
                  skew_vector(1),
                  skew_vector(2)).matrix;

    // Old anchor position
    const StateManager::Jacobian J_anchor_pos_old =
        Ci_q_G_new.normalized().toRotationMatrix().transpose();

    // New anchor position
    const StateManager::Jacobian J_anchor_pos_new =
        - Ci_q_G_new.normalized().toRotationMatrix().transpose();

    // Old feature parameters
    Matrix mat(Matrix::Identity(3,3));
    mat(0, 2) = - alpha_old / rho_old;
    mat(1, 2) = - beta_old / rho_old;
    mat(2, 2) = - 1 / rho_old;
    const StateManager::Jacobian J_anchor_feat_old =
        1 / rho_old
        * Ci_q_G_new.normalized().toRotationMatrix().transpose()
        * Ci_q_G_old.normalized().toRotationMatrix()
        * mat;

    // Initialize Jacobian of eq.(38) wrt to state in Li (supplementals, 2012)
    Matrix A_j(Matrix::Zero(3,n));

    unsigned int col = idx1 * jac_cols;

    A_j.block<jac_rows, jac_cols>(0, kSizeCoreErr + col) =
        J_anchor_pos_new;

    col += n_poses_max_ * jac_cols;
    A_j.block<jac_rows, jac_cols>(0, kSizeCoreErr + col) =
        J_anchor_att_new;

    col = 0;
    A_j.block<jac_rows, jac_cols>(0, kSizeCoreErr + col) =
        J_anchor_pos_old;

    col += n_poses_max_ * jac_cols;
    A_j.block<jac_rows, jac_cols>(0, kSizeCoreErr + col) =
        J_anchor_att_old;

    col = j * 3;
    A_j.block<jac_rows, jac_cols>(0, kSizeCoreErr + n_poses_max_ * 6 + col) =
        J_anchor_feat_old;

    // Populate covariance reparametrization matrix
    mat = Matrix::Identity(3,3);
    mat(0, 2) = - alpha_new;
    mat(1, 2) = - beta_new;
    mat(2, 2) = - rho_new;

    const size_t n1 = kSizeCoreErr + n_poses_max_ * 6 + j * 3;
    J.block(n1, 0, 3, n) = rho_new * mat * A_j;
  }

  // Reparametrize covariance
  covariance = J * covariance * J.transpose();
}

void StateManager::slideWindow(Eigen::VectorXd& atts,
                               Eigen::VectorXd& poss,
                               Matrix& covariance) {
  // Slide poses in the state vector
  atts.block(0, 0, (n_poses_max_-1) * 4, 1)
    = atts.block(4, 0, (n_poses_max_-1) * 4, 1);
  poss.block(0, 0, (n_poses_max_-1) * 3, 1)
    = poss.block(3, 0, (n_poses_max_-1) * 3, 1);
  atts.block<4, 1>((n_poses_max_-1) * 4, 0) = Matrix::Zero(4, 1);
  poss.block<3, 1>((n_poses_max_-1) * 3, 0) = Matrix::Zero(3, 1);

  // Slides poses in covariance matrix
  const size_t n = covariance.rows();

  Matrix left_mult = Matrix::Zero(n, n);
  left_mult.block(0, 0, 21, 21) = Matrix::Identity(21, 21);

  if (n_features_max_) {
    left_mult.block(21 + n_poses_max_ * 6,
                    21 + n_poses_max_ * 6,
                    n_features_max_ * 3,
                    n_features_max_ * 3)
      = Matrix::Identity(n_features_max_ * 3, n_features_max_ * 3);
  }

  Matrix right_mult = left_mult;

  left_mult.block(21, 24, (n_poses_max_ - 1) * 3, (n_poses_max_ - 1) * 3)
    = Matrix::Identity((n_poses_max_ - 1) * 3, (n_poses_max_ - 1) * 3);

  left_mult.block(21 + n_poses_max_ * 3,
                  24 + n_poses_max_ * 3,
                  (n_poses_max_ - 1) * 3,
                  (n_poses_max_ - 1) * 3)
    = Matrix::Identity((n_poses_max_ - 1) * 3, (n_poses_max_ - 1) * 3);

  right_mult.block(24, 21, (n_poses_max_ - 1) * 3, (n_poses_max_ - 1) * 3)
    = Matrix::Identity((n_poses_max_ - 1) * 3, (n_poses_max_ - 1) * 3);

  right_mult.block(24 + n_poses_max_ * 3,
                   21 + n_poses_max_ * 3,
                   (n_poses_max_ - 1) * 3,
                   (n_poses_max_ - 1) * 3)
    = Matrix::Identity((n_poses_max_ - 1) * 3, (n_poses_max_ - 1) * 3);

  covariance = left_mult * covariance * right_mult;

  // Slide indexes of anchors
  for (unsigned int i = 0; i < n_features_; i++)
  {
    anchor_idxs_[i]--;
  }

  // Correction sliding window position index
  n_poses_--;
}

AttitudeList
StateManager::convertCameraAttitudesToList(const State& state,
                                           const int max_size) const {
  // Determine output list size
  size_t size_out;
  if(max_size)
    size_out = std::min(max_size, n_poses_);
  else
    size_out = n_poses_;

  // Get camera orientation window from state
  const Matrix orientation_array = state.getOrientationArray();

  // Initialize list of n attitudes
  AttitudeList attitude_list(size_out, x::Attitude());

  // Assign each translation
  const size_t start_idx(n_poses_ - size_out);
  for (int i = start_idx; i < n_poses_; i++) {
    const x::Attitude attitude(orientation_array(4*i, 0),
                               orientation_array(4*i+1, 0),
                               orientation_array(4*i+2, 0),
                               orientation_array(4*i+3, 0));
    attitude_list[i - start_idx] = attitude;
  }

  return attitude_list;
}

x::TranslationList StateManager::convertCameraPositionsToList(const State& state) const
{
  // Get camera position from state
  const Matrix positions = state.getPositionArray();

  // Initialize list of n translations
  x::TranslationList position_list(n_poses_, x::Translation());

  // Assign each translation
  for (int i=0; i < n_poses_; i++)
  {
    const x::Translation translation(positions(3*i, 0), positions(3*i+1, 0), positions(3*i+2, 0));
    position_list[i] = translation;
  }

  return position_list;
}
