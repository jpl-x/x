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

#ifndef X_VIO_SLAM_UPDATE_H_
#define X_VIO_SLAM_UPDATE_H_

#include <x/vio/update.h>
#include <x/vio/types.h>

namespace x
{
	/**
	 * SLAM update.
	 *
	 * Inverse-depth SLAM. As presented in xVIO tech report.
	 */
	class SlamUpdate : public Update
	{
		public:

			/**
			 * Default constructor
			 */
			SlamUpdate() {};

			/**
			 * Constructor.
			 *
			 * Does the full update matrix construction job.
			 *
			 * @param[in] tkrs Feature tracks in normalized coordinates.
			 * @param[in] quats Camera quaternion states.
			 * @param[in] poss Camera position states.
			 * @param[in] feature_states Feature state vector.
			 * @param[in] anchor_idxs Anchor pose indexes of inverse-depth SLAM
			 *            features.
			 * @param[in] cov_s Error state covariance.
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] sigma_img Standard deviation of feature measurement
       *                      [in normalized coordinates]
			 */
			SlamUpdate(const x::TrackList& trks,
					const x::AttitudeList& quats,
					const x::TranslationList& poss,
					const Eigen::MatrixXd& feature_states,
					const std::vector<int>& anchor_idxs,
					const Eigen::MatrixXd& cov_s,
          const int n_poses_max,
          const double sigma_img);

			/**
			 * Compute inverse-depth 3D coordinates based for the new SLAM features.
			 *
			 * This method assumed the last pose state is the anchor. These
			 * coordinates will become the new SLAM feature states.
			 *
			 * @param[in] new_trks N image tracks to initialize as states.
       * @param[in] rho_0 Initial inverse depth of SLAM features [1/m].
			 * @param[out] ivds 3N vector {alpha, beta, rho}i of inverse-depth.
			 *                  coordinates
			 */
			void computeInverseDepthsNew(const x::TrackList& new_trks,
                                   const double rho_0,
					                         Eigen::MatrixXd& ivds) const;

		private:

			/**
			 * Process one feature track.
			 *
			 * @param[in] track Feature track in normalized coordinates
			 * @param[in] C_q_G Camera attitude state list 
			 * @param[in] G_p_C Camera position state list
			 * @param[in] feature_states Feature state vector
			 * @param[in] anchor_idxs Anchor pose indexes of inverse-depth SLAM
			 *            features
			 * @param[in] P State covariance
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] var_img Variance of feature measurement
       *                    [in normalized coordinates].
			 * @param[in] j Index of current track in the MSCKF track list
			 * @param[out] row_h Current rows in stacked Jacobian
			 */
			void processOneTrack(const x::Track& track,
					const x::AttitudeList& C_q_G,
					const x::TranslationList& G_p_C,
					const Eigen::MatrixXd& feature_states,
					const std::vector<int>& anchor_idxs,
					const Eigen::MatrixXd& P,
          const int n_poses_max,
          const double var_img,
					const size_t& j,
					size_t& row_h);

			/** 
			 * Computes inverse-depth 3D coordinates of 1 new feature.
			 *
			 * @param[in] feature New feature normalized coordinates.
       * @param[in] rho_0 Initial inverse depth of SLAM features [1/m].
			 * @param[in] idx Index at which the coordinates should be inserted at in ivds.
			 * @param[out] ivds 3N vector {alpha, beta, rho}i of inverse-depth.
			 *              coordinates
			 */
			void computeOneInverseDepthNew(const x::Feature& feature,
                                     const double rho_0,
					                           const unsigned int& idx,
					                           Eigen::MatrixXd& ivds) const;
	};
}

#endif  // X_VIO_SLAM_UPDATE_H_
