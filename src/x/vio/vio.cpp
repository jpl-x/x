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

#include <x/vio/vio.h>
#include <x/vio/tools.h>
#include <x/vision/types.h>

#include <iostream>

// if Boost was compiled with BOOST_NO_EXCEPTIONS defined, it expects a user
// defined trow_exception function, so define a dummy here, if this is the case
#include <exception>

using namespace x;

namespace boost
{
#ifdef BOOST_NO_EXCEPTIONS
void throw_exception(std::exception const & e) {}; // user defined
#endif
}


VIO::VIO() 
  : ekf_ { Ekf(vio_updater_) }
{
  // Initialize with invalid last range measurement
  // todo: Replace -1 with -min_delay
  x::RangeMeasurement no_measurement_range;
  no_measurement_range.timestamp = -1;
  last_range_measurement_ = no_measurement_range;

  // Initialize with invalid last sun angle measurement
  // todo: Replace -1 with -min_delay
  x::SunAngleMeasurement no_measurement_sun;
  no_measurement_sun.timestamp = -1;
  last_angle_measurement_ = no_measurement_sun;
}

bool VIO::isInitialized() const {
  return initialized_;
}

void VIO::setUp(const x::Params& params) {
  const x::Camera cam(params.cam_fx, params.cam_fy, params.cam_cx, params.cam_cy, params.cam_s, params.img_width,
                            params.img_height);
  const x::Tracker tracker(cam, params.fast_detection_delta, params.non_max_supp, params.block_half_length,
                                 params.margin, params.n_feat_min, params.outlier_method, params.outlier_param1,
                                 params.outlier_param2);

  // Compute minimum MSCKF baseline in normal plane (square-pixel assumption)
  msckf_baseline_n_ = params.msckf_baseline / (params.img_width * params.cam_fx);

  // Set up tracker and track manager
  const TrackManager track_manager(cam, msckf_baseline_n_);
  params_ = params;
  camera_ = cam;
  tracker_ = tracker;
  track_manager_ = track_manager;

  // Set up VIO state manager
  const int n_poses_state = params.n_poses_max;
  const int n_features_state = params.n_slam_features_max;
  const StateManager state_manager(n_poses_state, n_features_state);
  state_manager_ = state_manager;

  // Gravity - TODO(jeff) Read from params
  Vector3 g(0.0, 0.0, -9.81);

  // IMU noise
  x::ImuNoise imu_noise;
  imu_noise.n_w = params.n_w;
  imu_noise.n_bw = params.n_bw;
  imu_noise.n_a = params.n_a;
  imu_noise.n_ba = params.n_ba;

  // Updater setup
  x::MatchList matches; // unused empty match list since it's image callback
  TiledImage img;
  vio_updater_ = VioUpdater(tracker_,
                            state_manager_,
                            track_manager_,
                            params_.sigma_img,
                            params_.sigma_range,
                            params_.rho_0,
                            params_.sigma_rho_0,
                            params_.min_track_length,
                            params_.iekf_iter);

  // EKF setup
  const size_t state_buffer_sz = 250; // TODO(jeff) Read from params
  const State default_state = State(n_poses_state, n_features_state);
  const double a_m_max = 50.0;
  const unsigned int delta_seq_imu = 1;
  const double time_margin_bfr = 0.02;
  ekf_.set(vio_updater_,
           g,
           imu_noise,
           state_buffer_sz,
           default_state,
           a_m_max,
           delta_seq_imu,
           time_margin_bfr);
}

void VIO::setLastRangeMeasurement(x::RangeMeasurement range_measurement) {
  last_range_measurement_ = range_measurement;
}

void VIO::setLastSunAngleMeasurement(x::SunAngleMeasurement angle_measurement) {
  last_angle_measurement_ = angle_measurement;
}

State VIO::processImageMeasurement(double timestamp,
                                   const unsigned int seq,
                                   TiledImage& match_img,
                                   TiledImage& feature_img) {
  // Time correction
  const double timestamp_corrected = timestamp + params_.time_offset;

  // Pass measurement data to updater
  MatchList empty_list; // TODO(jeff) get rid of image callback and process match
                        // list from a separate tracker module.
  VioMeasurement measurement(timestamp_corrected,
                             seq,
                             empty_list,
                             match_img,
                             last_range_measurement_,
                             last_angle_measurement_);

  vio_updater_.setMeasurement(measurement);

  // Process update measurement with xEKF
  State updated_state = ekf_.processUpdateMeasurement();

  // Set state timestamp to original image timestamp for ID purposes in output.
  // We don't do that if that state is invalid, since the timestamp also carries
  // the invalid signature.
  if(updated_state.getTime() != kInvalid)
    updated_state.setTime(timestamp);

  // Populate GUI image outputs
  match_img = vio_updater_.getMatchImage();
  feature_img = vio_updater_.getFeatureImage();

  return updated_state;
}

State VIO::processMatchesMeasurement(double timestamp,
                                     const unsigned int seq,
                                     const std::vector<double>& match_vector,
                                     TiledImage& match_img,
                                     TiledImage& feature_img) {
  // Time correction
  const double timestamp_corrected = timestamp + params_.time_offset;

  // Import matches (except for first measurement, since the previous needs to
  // enter the sliding window)
  x::MatchList matches;
  if (vio_updater_.state_manager_.poseSize())
    matches = importMatches(match_vector, seq, match_img);

  // Compute 2D image coordinates of the LRF impact point on the ground
  x::Feature lrf_img_pt;
  lrf_img_pt.setXDist((camera_.getWidth()+1) / 2);
  lrf_img_pt.setYDist((camera_.getHeight()+1) / 2);
  camera_.undistort(lrf_img_pt);
  last_range_measurement_.img_pt = lrf_img_pt; 
  last_range_measurement_.img_pt_n = camera_.normalize(lrf_img_pt);

  // Pass measurement data to updater
  VioMeasurement measurement(timestamp_corrected,
                             seq,
                             matches,
                             feature_img,
                             last_range_measurement_,
                             last_angle_measurement_);
  
  vio_updater_.setMeasurement(measurement);

  // Process update measurement with xEKF
  State updated_state = ekf_.processUpdateMeasurement();

  // Set state timestamp to original image timestamp for ID purposes in output.
  // We don't do that if that state is invalid, since the timestamp carries the
  // invalid signature.
  if(updated_state.getTime() != kInvalid)
    updated_state.setTime(timestamp);

  // Populate GUI image outputs
  match_img = match_img;
  feature_img = vio_updater_.getFeatureImage();

  return updated_state;
}

/** Calls the state manager to compute the cartesian coordinates of the SLAM features.
 */
std::vector<Eigen::Vector3d>
VIO::computeSLAMCartesianFeaturesForState(
    const State& state) {
  return vio_updater_.state_manager_.computeSLAMCartesianFeaturesForState(state);
}

void VIO::initAtTime(double now) {
  ekf_.lock();
  initialized_ = false;
  vio_updater_.track_manager_.clear();
  vio_updater_.state_manager_.clear();

  // Initial IMU measurement (specific force, velocity)
  Vector3 a_m, w_m;
  a_m << 0.0, 0.0, 9.81;
  w_m << 0.0, 0.0, 0.0;

  // Initial time cannot be 0, which may happen when using simulated Clock time
  // before the first message has been received.
  const double timestamp  =
    now > 0.0 ? now : std::numeric_limits<double>::epsilon();

  //////////////////////////////// xEKF INIT ///////////////////////////////////
 
  // Initial core covariance matrix
  // TODO(jeff) read from params
  const double sigma_dp_x = 0.0;
  const double sigma_dp_y = 0.0;
  const double sigma_dp_z = 0.0;
  const double sigma_dv_x = 0.05;
  const double sigma_dv_y = 0.05;
  const double sigma_dv_z = 0.05;
  const double sigma_dtheta_x = 3.0*M_PI/180.0;
  const double sigma_dtheta_y = 3.0*M_PI/180.0;
  const double sigma_dtheta_z = 3.0*M_PI/180.0;
  const double sigma_dbw_x = 6.0*M_PI/180.0;
  const double sigma_dbw_y = 6.0*M_PI/180.0;
  const double sigma_dbw_z = 6.0*M_PI/180.0;
  const double sigma_dba_x = 0.3;
  const double sigma_dba_y = 0.3;
  const double sigma_dba_z = 0.3;
  const double sigma_dtheta_ic_x = 1.0 * M_PI / 180.0;
  const double sigma_dtheta_ic_y = 1.0 * M_PI / 180.0;
  const double sigma_dtheta_ic_z = 1.0 * M_PI / 180.0;
  const double sigma_dp_ic_x = 0.01;
  const double sigma_dp_ic_y = 0.01;
  const double sigma_dp_ic_z = 0.01;

  const size_t n_poses_state = params_.n_poses_max;
  const size_t n_features_state = params_.n_slam_features_max;

  // Initial vision state estimates and uncertainties are all zero
  const Matrix p_array = Matrix::Zero(n_poses_state * 3, 1);
  const Matrix q_array = Matrix::Zero(n_poses_state * 4, 1);
  const Matrix f_array = Matrix::Zero(n_features_state * 3, 1);
  const Eigen::VectorXd sigma_p_array = Eigen::VectorXd::Zero(n_poses_state * 3);
  const Eigen::VectorXd sigma_q_array = Eigen::VectorXd::Zero(n_poses_state * 3);
  const Eigen::VectorXd sigma_f_array = Eigen::VectorXd::Zero(n_features_state * 3);

  // Construct initial covariance matrix
  const size_t n_err = kSizeCoreErr + n_poses_state * 6 + n_features_state * 3;
  Eigen::VectorXd sigma_diag(n_err);
  sigma_diag << sigma_dp_x, sigma_dp_y, sigma_dp_z,
                sigma_dv_x, sigma_dv_y, sigma_dv_z,
                sigma_dtheta_x, sigma_dtheta_y, sigma_dtheta_z,
                sigma_dbw_x, sigma_dbw_y, sigma_dbw_z,
                sigma_dba_x, sigma_dba_y, sigma_dba_z,
                sigma_dtheta_ic_x, sigma_dtheta_ic_y, sigma_dtheta_ic_z,
                sigma_dp_ic_x, sigma_dp_ic_y, sigma_dp_ic_z,
                sigma_p_array, sigma_q_array, sigma_f_array;

  const Eigen::VectorXd cov_diag = sigma_diag.array() * sigma_diag.array();
  const Matrix cov = cov_diag.asDiagonal(); 

  // Construct initial state
  const unsigned int dummy_seq = 0;
  State init_state(timestamp,
                   dummy_seq,
                   params_.p,
                   params_.v,
                   params_.q,
                   params_.b_w,
                   params_.b_a,
                   p_array,
                   q_array,
                   f_array,
                   cov,
                   params_.q_ic,
                   params_.p_ic,
                   w_m,
                   a_m);

  // Try to initialize the filter with initial state input
  try {
    ekf_.initializeFromState(init_state);
  } catch (std::runtime_error& e) {
    std::cerr << "bad input: " << e.what() << std::endl;
  } catch (init_bfr_mismatch) {
    std::cerr << "init_bfr_mismatch: the size of dynamic arrays in the "
                 "initialization state match must match the size allocated in "
                 "the buffered states." << std::endl;
  }
  ekf_.unlock();

  initialized_ = true;
}

/** \brief Gets 3D coordinates of MSCKF inliers and outliers.
 *
 *  These are computed in the Measurement class instance.
 */
void VIO::getMsckfFeatures(x::Vector3dArray& inliers,
                           x::Vector3dArray& outliers) {
  inliers = vio_updater_.getMsckfInliers();
  outliers = vio_updater_.getMsckfOutliers();
}

State VIO::processImu(const double timestamp,
                      const unsigned int seq,
                      const Vector3& w_m,
                      const Vector3& a_m) {
  return ekf_.processImu(timestamp, seq, w_m, a_m);
}

x::MatchList VIO::importMatches(const std::vector<double>& match_vector,
                                const unsigned int seq,
                                x::TiledImage& img_plot) const {
  // 9N match vector structure:
  // 0: time_prev
  // 1: x_dist_prev
  // 2: y_dist_prev
  // 3: time_curr
  // 4: x_dist_curr
  // 5: x_dist_curr
  // 6,7,8: 3D coordinate of feature

  // Number of matches in the input vector
  assert(match_vector.size() % 9 == 0);
  const unsigned int n_matches = match_vector.size() / 9;

  // Declare new lists
  x::MatchList matches(n_matches);

  // Store the match vector into a match list
  for (unsigned int i = 0; i < n_matches; ++i)
  {
    // Undistortion
    const double x_dist_prev = match_vector[9 * i + 1];
    const double y_dist_prev = match_vector[9 * i + 2];

    const double x_dist_curr = match_vector[9 * i + 4];
    const double y_dist_curr = match_vector[9 * i + 5];

    // Features and match initializations
    x::Feature previous_feature(match_vector[9 * i] + params_.time_offset, seq - 1, 0.0, 0.0, x_dist_prev,
                                      y_dist_prev);
    camera_.undistort(previous_feature);

    x::Feature current_feature(match_vector[9 * i + 3] + params_.time_offset, seq, 0.0, 0.0, x_dist_curr,
                                     y_dist_curr);
    camera_.undistort(current_feature);

    x::Match current_match;
    current_match.previous = previous_feature;
    current_match.current = current_feature;

    // Add match to list
    matches[i] = current_match;
  }

  // Publish matches to GUI
  x::Tracker::plotMatches(matches, img_plot);

  return matches;
}
