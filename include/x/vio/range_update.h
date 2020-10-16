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

#ifndef X_VIO_RANGE_UPDATE_H_
#define X_VIO_RANGE_UPDATE_H_

#include <x/vio/update.h>
#include <x/vio/types.h>

namespace x
{
  /**
   * Range visual update.
   *
   * This update requires SLAM states. As presented in xVIO tech report.
   */
  class RangeUpdate : public Update
  {
    public:

      /**
       * Constructor.
       *
       * Does the full update matrix construction job.
       *
       * @param[in] range_meas Range measurement
       * @param[in] tr_feat_ids Indexes of the 3 triangular facet vertices
       *                        in the SLAM state vector.
       * @param[in] quats Camera quaternion states
       * @param[in] poss Camera position states
       * @param[in] feature_states Feature state vector 
       * @param[in] anchor_idxs Anchor pose indexes of inverse-depth SLAM
       *            features
       * @param[in] cov_s Error state covariance
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] sigma_range Standard deviation of range measurement noise
       *                        [m].
       */
      RangeUpdate(const x::RangeMeasurement& range_meas,
                  const std::vector<int>& tr_feat_ids,
                  const x::AttitudeList& quats,
                  const x::TranslationList& poss,
                  const Eigen::MatrixXd& feature_states,
                  const std::vector<int>& anchor_idxs,
                  const Eigen::MatrixXd& cov_s,
                  const int n_poses_max,
                  const double sigma_range);

    private:
      
      using RangeJacobian = Eigen::Matrix<double, 1, kJacCols>;
      
      /** 
       * Processes a range measurement with the facet model.
       *
       * @param[in] range_meas Range measurement
       * @param[in] tr_feat_ids Indexes of the 3 triangular facet vertices
       *                        in the SLAM state vector.
       * @param[in] C_q_G Camera attitude state list 
       * @param[in] G_p_C Camera position state list
       * @param[in] feature_states Feature state vector
       * @param[in] anchor_idxs Anchor pose indexes of inverse-depth SLAM
       *            features
       * @param[in] P State covariance
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] sigma_range Standard deviation of range measurement noise
       *                        [m].
       */
      void processRangedFacet(const x::RangeMeasurement& range_meas,
                              const std::vector<int>& tr_feat_ids,
                              const x::AttitudeList& C_q_G,
                              const x::TranslationList& G_p_C,
                              const Eigen::MatrixXd& feature_states,
                              const std::vector<int>& anchor_idxs,
                              const Eigen::MatrixXd& P,
                              const int n_poses_max,
                              const double sigma_range);
      
      /**
       * Processes a range measurement with the feature model.
       *
       * @param[in] range_meas Range measurement
       * @param[in] C_q_G Camera attitude state list 
       * @param[in] G_p_C Camera position state list
       * @param[in] feature_states Feature state vector
       * @param[in] anchor_idxs Anchor pose indexes of inverse-depth SLAM
       *            features
       * @param[in] P State covariance
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] sigma_range Standard deviation of range measurement noise
       *                        [m].
       * @param[in] j Index of feature in the list
       */
      void processRangedFeature(const x::RangeMeasurement& range_meas,
                                const x::AttitudeList& C_q_G,
                                const x::TranslationList& G_p_C,
                                const Eigen::MatrixXd& feature_states,
                                const std::vector<int>& anchor_idxs,
                                const Eigen::MatrixXd& P,
                                const int n_poses_max,
                                const double sigma_range,
                                const size_t& j);
  };
}

#endif  // X_VIO_SLAM_UPDATE_H_
