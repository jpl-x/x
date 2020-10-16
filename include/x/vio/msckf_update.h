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

#ifndef X_VIO_MSCKF_UPDATE_H_
#define X_VIO_MSCKF_UPDATE_H_

#include <x/vio/update.h>
#include <x/vio/types.h>
#include <x/vision/triangulation.h>

namespace x
{
  /**
   * MSCKF update.
   *
   * Implementation of the Multi-State Contraint Kalman Filter.
   * (MSCKF). As presented Mourikis 2007 ICRA paper.
   */
  class MsckfUpdate : public Update
  {
    public:
      /**
       * Constructor.
       *
       * Does the full update matrix construction job.
       *
       * @param[in] tkrs Feature tracks in normalized coordinates
       * @param[in] quats Camera quaternion states
       * @param[in] poss Camera position states
       * @param[in] triangulator Feature triangulator
       * @param[in] cov_s Error state covariance
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] sigma_img Standard deviation of feature measurement
       *                      [in normalized coordinates]
       */
      MsckfUpdate(const x::TrackList& trks,
                  const x::AttitudeList& quats,
                  const x::TranslationList& poss,
                  const Triangulation& triangulator,
                  const Eigen::MatrixXd& cov_s,
                  const int n_poses_max,
                  const double sigma_img);

      /***************************** Getters **********************************/
      
      /**
       * Returns a constant reference to the inliers 3D cartesian coodinates.
       */
      const Vector3dArray& getInliers() const { return inliers_; };

      /**
       * Returns a constant reference to the outliers 3D cartesian coodinates.
       */
      const Vector3dArray& getOutliers() const { return outliers_; };

    private:
      /**
       * 3D cartesian coordinates prior for MSCKF feature inliers
       */
      Vector3dArray inliers_;

      /**
       * 3D cartesian coordinates prior for MSCKF feature outliers
       */
      Vector3dArray outliers_;
      
      /**
       * Process one feature track.
       *
       * @param[in] track Feature track in normalized coordinates
       * @param[in] C_q_G Camera attitude state list 
       * @param[in] G_p_C Camera position state list
       * @param[in] triangulator Feature triangulator
       * @param[in] P State covariance
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] var_img Variance of feature measurement.
       *                    [in normalized coordinates].
       * @param[in] j Index of current track in the MSCKF track list
       * @param[out] row_h Current rows in stacked Jacobian
       */
      void processOneTrack(const x::Track& track,
                          const x::AttitudeList& C_q_G,
                          const x::TranslationList& G_p_C,
                          const Triangulation& triangulator,
                          const Eigen::MatrixXd& P,
                          const int n_poses_max,
                          const double var_img,
                          const size_t& j,
                          size_t& row_h);


  };
}

#endif  // X_VIO_MSCKF_UPDATE_H_
