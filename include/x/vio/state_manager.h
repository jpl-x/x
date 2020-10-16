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

#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include <x/vio/tools.h>
#include <x/vio/types.h>

#include<x/ekf/state.h>
#include <x/vision/tracker.h>

namespace x {
  class StateManager {
    public:
      StateManager(int n_poses_max = 0, int n_features_max = 0)
        : n_poses_max_ { n_poses_max }
        , n_features_max_ { n_features_max }
        , anchor_idxs_(n_features_max , -1)
      {}

      /******************************* Getters ************************************/

      size_t getNFeatures() const { return n_features_; };

      std::vector<int> getAnchorIdxs() const { return anchor_idxs_; };

      /****************************************************************************/

      void clear();

      /**
       * SLAM and MSCKF state management: slide window, add, remove, reparam.
       *
       * Adds and removes necessary states from the closest state to timestamp 
       * in the buffer. Changes are propagated to the latest state. New states
       * are the latest camera position and attitude estimates. Removed states
       * are lost persistent features. This method also clears the sliding
       * window when it has reached max size.
       *
       * @param[in,out] state State to manage.
       * @param[in] del_feat_idx Indexes of features to be deleted in the state
       *                         vector.
       */
      void manage(State& state, std::vector<unsigned int> del_feat_idx);

      /**
       * Initializes MSCKF-SLAM features in state and covariance in one batch.
       * 
       * @param[in,out] state state to be augmented.
       * @param[in] init_mats MSCKF-SLAM initialization matrices.
       * @param[in] correction Correction vector after feature's opportunistic
       *                       update.
       * @param[in] sigma_img Standard deviation of feature measurement
       *                      [in normalized coordinates].
      */
      void initMsckfSlamFeatures(State& state,
                                 const x::MsckfSlamMatrices& init_mats,
                                 const Matrix& correction,
                                 const double sigma_img);

      /**
       * Initializes a new SLAM the standard way (see Civera).
       *
       * @param[in,out] state state to be augmented.
       * @param[in] feature Inverse-depth feature coordinates (size 3N).
       * @param[in] sigma_img Standard deviation of feature measurement
       *                      [in normalized coordinates].
       * @param[in] sigma_rho_0 Initial standard deviation of SLAM inverse
       *                        depth [1/m].
       */
      void initStandardSlamFeatures(State& state,
                                    const Matrix& feature,
                                    const double sigma_img,
                                    const double sigma_rho_0);

      unsigned int anchorPoseIdxForFeatureAtIdx(unsigned int i)
      {
        assert(i < n_features_);
        return anchor_idxs_[i];
      }

      std::vector<Eigen::Vector3d> computeSLAMCartesianFeaturesForState(
          const State& state) const;

      /// The value of poseSize() is the number of poses stored in the most recent state.
      /// \return The current number of poses.
      size_t poseSize(void) const { return n_poses_; }

      /**
       * Convert attitude states to an attitude list.
       *
       * @param[in] state State input.
       * @param[in] max_size Max size of output list, cropping from the end (default 0: no cropping)
       * @return The attitude list.
       */
      AttitudeList convertCameraAttitudesToList(const State& state,
                                                const int max_size = 0) const;

      /**
       * Convert position states to a translation list.
       *
       * @param[in] state State input.
       * @return The position list.
      */
      x::TranslationList convertCameraPositionsToList(const State& state) const;

      /// Retrieve a specific entry from an attitudes buffer.
      /// \param attitudes The buffer to retrieve the attitude from
      /// \param idx The index of the attitude in attitudes
      /// \return The value of the camera attitudes at position idx in state.
      Eigen::Vector4d Attitude(Eigen::VectorXd const& attitudes, int idx) const
      {
        return x::MatrixBlock(attitudes, idx * 4, 0, 4, 1);
      }

      /// Retrieve a specific entry from an positions buffer.
      /// \param positions The buffer to retrieve the position from
      /// \param idx The index of the position in positions
      /// \return The value of the camera positions at position idx in state.
      Eigen::Vector3d Position(Eigen::VectorXd const& positions, int idx) const
      {
        return x::MatrixBlock(positions, idx * 3, 0, 3, 1);
      }

    private:
      /**
       * Maximum number of poses in state sliding window.
       *
       * Fixed for all states.
       */
      int n_poses_max_;

      /**
       * Maximum number of features in state.
       *
       * Fixed for all states.
       */
      int n_features_max_;

      /**
       * Current number of poses in state vector.
       */
      int n_poses_ = 0;

      size_t n_features_ = 0; ///< Current number of persistent features

      /**
       * Anchor pose indexes of the SLAM feature states.
       *
       * i-th entry is the index of the pose state which is the anchor for SLAM
       * feature state i, in inverse-depth coordinates.
       */
      std::vector<int> anchor_idxs_;

      bool stateHasBeenFilledBefore_ = false;

      static unsigned constexpr jac_rows = 3;  ///< Rows of the Jacobian matrix entries
      static unsigned constexpr jac_cols = 3;  ///< Cols of the Jacobian matrix entries

      using Jacobian = Eigen::Matrix<double, jac_rows, jac_cols>;

      /**
       * Augment state covariance matrix with the new poses.
       *
       * @param[in] state Input state
       * @param[in] pos Current pose index in the sliding window.
       * @param[in,out] Covariance matrix of input state
       * TODO(jeff) Unsafe, covariance matrix is already in the state object.
       */
      void augmentCovariance(const State& state,
                             const int pos,
                             Matrix& covariance);

      /// Reparametrizes persistent features which are anchored to the oldest pose
      /// in the sliding window.
      /// \param atts_old sliding windows of attitudes before cleanup
      /// \param poss_old sliding windows of positions before cleanup
      /// \param features feature states to be reparametrized
      /// \param covariance covariance matrix to be reparametrized
      void reparametrizeFeatures(Eigen::VectorXd const& atts_old,
                                 Eigen::VectorXd const& poss_old,
                                 Eigen::VectorXd& features,
                                 Matrix& covariance);

      /// Slides the window of poses in the state vector and covariance matrix. This
      /// adds the newest pose, but also removes the oldest one when the window is
      /// full.
      /// \param atts Sliding windows of attitudes
      /// \param poss Sliding windows of positions
      /// \param covariance Covariance matrix
      void slideWindow(Eigen::VectorXd& atts,
                       Eigen::VectorXd& poss,
                       Matrix& covariance);

      /* \brief Append new features to the state vector and covariance
       *        matrix
       *
       * \param state_ptr
       * \param new_features 3n feature states to insert
       * \param cov 		 3n x 3n feature covariance matrix
       * \param cross 	 3n x n cross-covariance matrix with other states
       */	
      void addFeatureStates(State& state_ptr,
                            const Matrix& new_features,
                            const Matrix& cov,
                            const Matrix& cross);
  };
} // namespace x
#endif /* STATE_MANAGER_H */
