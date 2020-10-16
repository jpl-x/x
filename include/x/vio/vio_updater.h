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

#ifndef X_VIO_VIO_UPDATER_H_
#define X_VIO_VIO_UPDATER_H_

#include <x/ekf/updater.h>
#include <x/vio/state_manager.h>
#include <x/vio/track_manager.h>
#include <x/vio/tools.h>
#include <x/vio/types.h>
#include <x/vio/msckf_slam_update.h>
#include <x/vio/slam_update.h>
#include <x/ekf/state.h>
#include <Eigen/QR>

namespace x {
  class VioUpdater : public Updater
  {
    public:
      /**
       * Default constructor.
       */
      VioUpdater() {};

      /**
       * Construct with known parameters.
       */
      VioUpdater(Tracker& tracker,
                 StateManager& state_manager,
                 TrackManager& track_manager,
                 const double sigma_img,
                 const double sigma_range,
                 const double rho_0,
                 const double sigma_rho_0,
                 const int min_track_length,
                 const int iekf_iter = 1);

      void setMeasurement(const VioMeasurement& measurement);

      /**
       * Get measurement timestamp.
       */
      double getTime() const;

      /**
       * Get reference to the tracker debug image.
       */
      TiledImage& getMatchImage();

      /**
       * Get reference to the track manager debug image.
       */
      TiledImage& getFeatureImage();

      /**
       * Get list of MSCKF inliers' 3D coordinates.
       */
      Vector3dArray getMsckfInliers() const;

      /**
       * Get list of MSCKF outliers' 3D coordinates.
       */
      Vector3dArray getMsckfOutliers() const;

    private:
      /**
       * VIO measurement: image + optional range and sun angle.
       */
      VioMeasurement measurement_;

      Tracker tracker_;
      StateManager state_manager_;
      TrackManager track_manager_;
      
      /**
       * Tracker debug image.
       */
      TiledImage match_img_;

      /**
       * Track manager debug image.
       */
      TiledImage feature_img_;

      /**
       * Standard deviation of feature measurement [in normalized coordinates].
       */
      double sigma_img_;

      /**
       * Standard deviation of range measurement noise [m].
       */
      double sigma_range_;

      /**
       * Initial inverse depth of SLAM features [1/m].
       */
      double rho_0_;

      /**
       * Initial standard deviation of SLAM inverse depth [1/m].
       */
      double sigma_rho_0_;

      /**
       * Minimum track length for a visual feature to be processed.
       */
      int min_track_length_;

      
      // All track members assume normalized undistorted feature coordinates.
      TrackList msckf_trks_;  // Normalized tracks for MSCKF update
      // New SLAM features initialized with semi-infinite depth uncertainty
      // (standard SLAM)
      TrackList new_slam_std_trks_;
      // New SLAM features initialized with MSCKF tracks (MSCKF-SLAM)
      TrackList new_msckf_slam_trks_;
      TrackList slam_trks_;
      std::vector<unsigned int> lost_slam_trk_idxs_;
      Vector3dArray msckf_inliers_;   //!< 3D coordinates prior for MSCKF feature
      //!< inliers.
      Vector3dArray msckf_outliers_;  //!< 3D coordinates prior for MSCKF feature
      //!< outliers.

      /**
       * SLAM update object.
       */
      SlamUpdate slam_;

      /**
       * MSCKF-SLAM update object
       */
      MsckfSlamUpdate msckf_slam_;

      /**
       * Measurement processing
       *
       * Only happens once, and is stored if measurement later re-applied. Here this
       * where image processing takes place.
       *
       * @param[in] state State prior
       */
      void preProcess(const State& state);

      /**
       * Pre-update work
       *
       * Stuff that needs happen before the Kalman update. This function will NOT
       * be called at  each IEKF iteration. Here, this corresponds to state
       * management.
       *
       * @param[in,out] state Update state
       * @return True if an update should be constructed
       */
      bool preUpdate(State& state);

      /**
       * Update construction.
       *
       * Prepares measurement Jacobians, residual and noice matrices and applies
       * (iterated) EKF update. This function WILL be called at each IEKF iteration.
       *
       * @param[in] state Update state
       * @param[out] h Measurement Jacobian matrix
       * @param[out] res Measurement residual vector
       * @param[out] r Measurement noise covariance matrix
       */
      void constructUpdate(const State& state,
                           Matrix& h,
                           Matrix& res,
                           Matrix& r);
      
      /**
       * Post-update work.
       *
       * Stuff that need to happen after the Kalman update. Here this SLAM feature
       * initialization.
       *
       * @param[in,out] state Update state.
       * @param[in] correction Kalman state correction.
       */
      void postUpdate(State& state, const Matrix& correction);

      /** @brief QR decomposition
       *
       *  Computes the QR decomposition of the MSCKF update
       *  Jacobian and updates the other terms according to Mourikis'
       *  2007 paper.
       *
       *  @param[in,out] h Jacobian matrix of MSCKF measurements
       *  @param[in,out] res MSCKF measurement residuals
       *  @param[in,out] R Measurement noise matrix
       */
      void applyQRDecomposition(Matrix& h, Matrix& res, Matrix& R);

      friend class VIO;
  };
} // namespace x

#endif  // XVIO_MEASUREMENT_H_
