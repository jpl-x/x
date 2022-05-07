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

#include <x/vio/vio_updater.h>
#include <x/vio/msckf_update.h>
#include <x/vio/msckf_slam_update.h>
#include <x/vio/range_update.h>
#include <x/vio/solar_update.h>

using namespace x;

VioUpdater::VioUpdater(Tracker& tracker,
                       StateManager& state_manager,
                       TrackManager& track_manager,
                       const double sigma_img,
                       const double sigma_range,
                       const double rho_0,
                       const double sigma_rho_0,
                       const int min_track_length,
                       const int iekf_iter)
  : tracker_(tracker)
  , state_manager_(state_manager)
  , track_manager_(track_manager)
  , sigma_img_ { sigma_img }
  , sigma_range_ { sigma_range }
  , rho_0_ { rho_0 }
  , sigma_rho_0_ { sigma_rho_0 }
  , min_track_length_ { min_track_length }
{
  iekf_iter_ = iekf_iter;
}

double VioUpdater::getTime() const {
  return measurement_.timestamp;
}

TiledImage& VioUpdater::getMatchImage() {
  return match_img_;
}

TiledImage& VioUpdater::getFeatureImage() {
  return feature_img_;
}

Vector3dArray VioUpdater::getMsckfInliers() const {
  return msckf_inliers_;
}

Vector3dArray VioUpdater::getMsckfOutliers() const {
  return msckf_outliers_;
}

void VioUpdater::setMeasurement(const VioMeasurement& measurement) {
  measurement_ = measurement;
}

void VioUpdater::preProcess(const State& state) {
  // If the length of matches_ is 0, that means we used the image constructor
  // for the current object and the tracker needs to be run
  if (measurement_.matches.size() == 0) {
    // Track features
    match_img_ = measurement_.image.clone();
    tracker_.track(match_img_, measurement_.timestamp, measurement_.seq);

    // If we are processing images and last image didn't go back in time
    if (tracker_.checkMatches())
      measurement_.matches = tracker_.getMatches();
  }

  // Construct list of camera orientation states.
  // Note: pose window has not been slid yet and only goes up to
  // the previous frame. We need to crop the first pose out of the
  // list and add the current pose manually.
  // This is used to used check MSCKF track baseline.
  const int n_poses_max = state.nPosesMax();
  const int list_sz = n_poses_max - 1;
  AttitudeList cam_rots 
    = state_manager_.convertCameraAttitudesToList(state, list_sz);
  const Attitude last_rot = state.computeCameraAttitude();
  cam_rots.push_back(last_rot);

  // Sort matches into tracks
  feature_img_ = measurement_.image.clone();
  const int n_slam_features_max = state.nFeaturesMax();
  track_manager_.manageTracks(measurement_.matches,
                              cam_rots,
                              n_poses_max,
                              n_slam_features_max,
                              min_track_length_,
                              feature_img_);

  // Get image measurements
  slam_trks_           = track_manager_.normalizeSlamTracks(n_poses_max);
  msckf_trks_          = track_manager_.getMsckfTracks();
  new_slam_std_trks_   = track_manager_.getNewSlamStdTracks();
  new_msckf_slam_trks_ = track_manager_.getNewSlamMsckfTracks();

  // Collect indexes of persistent tracks that were lost at this time step
  lost_slam_trk_idxs_ = track_manager_.getLostSlamTrackIndexes();
}

bool VioUpdater::preUpdate(State& state) {
  // Manage vision states to be added, removed, reparametrized or sled
  state_manager_.manage(state, lost_slam_trk_idxs_);

  // Return true if there are any visual update measurements 
  return msckf_trks_.size()
         || slam_trks_.size()
         || new_slam_std_trks_.size()
         || new_msckf_slam_trks_.size();
}

void VioUpdater::constructUpdate(const State& state,
                                  Matrix& h,
                                  Matrix& res,
                                  Matrix& r) {
  // Construct list of pose states (the pose window has been slid
  // and includes current camera pose)
  const TranslationList G_p_C
    = state_manager_.convertCameraPositionsToList(state);
  const AttitudeList C_q_G
    = state_manager_.convertCameraAttitudesToList(state);

  // Retrieve state covariance prior
  Matrix P = state.getCovariance();

  // Set up triangulation for MSCKF // TODO(jeff) set from params
  const unsigned int max_iter = 10; // Gauss-Newton max number of iterations
  const double term = 0.00001; // Gauss-Newton termination criterion
  const Triangulation triangulator(C_q_G, G_p_C, max_iter, term);

  /* MSCKF */

  // Construct update
  const int n_poses_max = state.nPosesMax();
  const MsckfUpdate msckf(msckf_trks_,
                          C_q_G,
                          G_p_C,
                          triangulator,
                          P,
                          n_poses_max,
                          sigma_img_);

  // Get matrices
  const Matrix& res_msckf = msckf.getResidual();
  const Matrix& h_msckf = msckf.getJacobian();
  const Eigen::VectorXd& r_msckf_diag = msckf.getCovDiag();
  const Vector3dArray& msckf_inliers  = msckf.getInliers();
  const Vector3dArray& msckf_outliers = msckf.getOutliers();
  const size_t rows_msckf = h_msckf.rows();

  /* MSCKF-SLAM */

  // Construct update
  msckf_slam_ = MsckfSlamUpdate(new_msckf_slam_trks_,
                                C_q_G,
                                G_p_C,
                                triangulator,
                                P,
                                n_poses_max,
                                sigma_img_);

  // Get matrices
  const Matrix& res_msckf_slam = msckf_slam_.getResidual();
  const Matrix& h_msckf_slam = msckf_slam_.getJacobian();
  const Eigen::VectorXd& r_msckf_slam_diag = msckf_slam_.getCovDiag();
  const Vector3dArray& msckf_slam_inliers  = msckf_slam_.getInliers();
  const Vector3dArray& msckf_slam_outliers = msckf_slam_.getOutliers();
  const size_t rows_msckf_slam = h_msckf_slam.rows();

  // Stack all MSCKF inliers/outliers
  msckf_inliers_ = msckf_inliers;
  msckf_inliers_.insert(msckf_inliers_.end(),
      msckf_slam_inliers.begin(),
      msckf_slam_inliers.end());

  msckf_outliers_ = msckf_outliers;
  msckf_outliers_.insert(msckf_outliers_.end(),
      msckf_slam_outliers.begin(),
      msckf_slam_outliers.end());

  /* SLAM */

  // Get SLAM feature priors and inverse-depth pose anchors
  const Matrix& feature_states = state.getFeatureArray();
  const std::vector<int> anchor_idxs = state_manager_.getAnchorIdxs();

  // Contruct update
  slam_ = SlamUpdate(slam_trks_,
                     C_q_G,
                     G_p_C,
                     feature_states,
                     anchor_idxs,
                     P,
                     n_poses_max,
                     sigma_img_);
  const Matrix& res_slam = slam_.getResidual();
  const Matrix& h_slam = slam_.getJacobian();
  const Eigen::VectorXd& r_slam_diag = slam_.getCovDiag();
  const size_t rows_slam = h_slam.rows();

  /* Range-SLAM */

  size_t rows_lrf = 0;
  Matrix h_lrf = Matrix::Zero(0, P.cols()), res_lrf = Matrix::Zero(0, 1);
  Eigen::VectorXd r_lrf_diag;

  if (measurement_.range.timestamp > 0.1 && slam_trks_.size()) {
    // 2D image coordinates of the LRF impact point on the ground
    Feature lrf_img_pt;
    lrf_img_pt.setXDist(320.5);
    lrf_img_pt.setYDist(240.5);

    // IDs of the SLAM features in the triangles surrounding the LRF
    const std::vector<int> tr_feat_ids = track_manager_.featureTriangleAtPoint(lrf_img_pt, feature_img_);

    // If we found a triangular facet to construct the range update
    if (tr_feat_ids.size())
    {
      // Contruct update
      const RangeUpdate range_slam(measurement_.range,
                                   tr_feat_ids,
                                   C_q_G,
                                   G_p_C,
                                   feature_states,
                                   anchor_idxs,
                                   P,
                                   n_poses_max,
                                   sigma_range_);
      res_lrf = range_slam.getResidual();
      h_lrf = range_slam.getJacobian();
      r_lrf_diag = range_slam.getCovDiag();
      rows_lrf = 1;

      // Don't reuse measurement
      measurement_.range.timestamp = -1;
    }
  }

  /* Sun Sensor */

  size_t rows_sns = 0;
  Matrix h_sns, res_sns;
  Eigen::VectorXd r_sns_diag;

  if (measurement_.sun_angle.timestamp > -1) {
    // Retrieve body orientation
    const Quaternion& att = state.getOrientation();

    // Construct solar update
    const SolarUpdate solar_update(measurement_.sun_angle, att, P);
    res_sns = solar_update.getResidual();
    h_sns = solar_update.getJacobian();
    r_sns_diag = solar_update.getCovDiag();
    rows_sns = 2;

    // Don't reuse measurement
    measurement_.sun_angle.timestamp = -1;
  }

  /* Combined update */

  const size_t rows_total
    = rows_msckf + rows_msckf_slam + rows_slam + rows_lrf + rows_sns;
  const size_t cols = P.cols();
  h = Matrix::Zero(rows_total, cols);
  Eigen::VectorXd r_diag = Eigen::VectorXd::Ones(rows_total);
  res = Matrix::Zero(rows_total, 1);

  h << h_msckf,
    h_msckf_slam,
    h_slam,
    h_lrf,
    h_sns;

  r_diag << r_msckf_diag,
         r_msckf_slam_diag,
         r_slam_diag,
         r_lrf_diag,
         r_sns_diag;
  r = r_diag.asDiagonal();

  res << res_msckf,
      res_msckf_slam,
      res_slam,
      res_lrf,
      res_sns;

  // QR decomposition of the update Jacobian
  applyQRDecomposition(h, res, r);
}

void VioUpdater::postUpdate(State& state, const Matrix& correction) {
  // MSCKF-SLAM feature init
  // Insert all new MSCKF-SLAM features in state and covariance
  if (new_msckf_slam_trks_.size()) {
    // TODO(jeff) Do not initialize features which have failed the
    // Mahalanobis test. They need to be removed from the track
    // manager too.
    state_manager_.initMsckfSlamFeatures(state,
                                         msckf_slam_.getInitMats(),
                                         correction,
                                         sigma_img_);
  }

  // STANDARD SLAM feature initialization
  if (new_slam_std_trks_.size()) {
    // Compute inverse-depth coordinates of new SLAM features
    Matrix features_slam_std;
    slam_.computeInverseDepthsNew(new_slam_std_trks_, rho_0_, features_slam_std);

    // Insert it in state and covariance
    state_manager_.initStandardSlamFeatures(state,
                                            features_slam_std,
                                            sigma_img_,
                                            sigma_rho_0_);
  }
}

void VioUpdater::applyQRDecomposition(Matrix& h, Matrix& res, Matrix& R)
{
  // Check if the QR decomposition is actually necessary
  unsigned int rowsh(h.rows()), colsh(h.cols());
  bool QR = rowsh > colsh + 1;

  // QR decomposition using Householder transformations (same computational
  // complexity as Givens)
  if (QR)
  {
    // Compute QR of the augmented [h|res] matrix to avoid forming Q1
    // explicitly (Dongarra et al., Linpack users's guide, 1979)
    Matrix hRes(rowsh, colsh + 1);
    hRes << h, res;
    Eigen::HouseholderQR<Matrix> qr(hRes);
    // Get the upper triangular matrix of the augmented hRes QR
    Matrix Thz = qr.matrixQR().triangularView<Eigen::Upper>();

    // Extract the upper triangular matrix of the h QR
    h = Thz.block(0, 0, colsh, colsh);
    // Extract the projected residual vector
    res = Thz.block(0, colsh, colsh, 1);
    // Form new Kalman update covariance matrix
    const double var_img = sigma_img_ * sigma_img_;
    R = var_img * Matrix::Identity(colsh, colsh);
  }
  // else we leave the inputs unchanged
}
