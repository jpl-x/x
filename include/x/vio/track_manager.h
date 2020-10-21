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

#ifndef TRACK_MANAGER_H_
#define TRACK_MANAGER_H_

#include <x/vio/types.h>
#include <x/vision/types.h>
#include <x/vision/camera.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace x {
  /**
   * Track manager class
   *
   * Assembles feature matches, coming from the tracker class, into tracks.
   * When a track is long enough, it is assigned to a VIO update type: SLAM,
   * MSCKF, or SLAM-MSCKF.
   */
  class TrackManager
  {
    public:
      TrackManager();
      TrackManager(const Camera& camera, const double min_baseline_n);
      void setCamera(Camera camera);

      // All track getters are in normalized image coordinates
      TrackList getMsckfTracks() const; // Tracks ready for MSCKF update
      TrackList getNewSlamStdTracks() const;
      TrackList getNewSlamMsckfTracks() const;

      /**
       * Normalize SLAM tracks.
       *
       * @param[in] size_out Max size of output tracks, cropping from the end
       *                     (default 0: no cropping)
       */
      TrackList normalizeSlamTracks(const int size_out) const;

      // Clear all the tracks (for re-init)
      void clear();
      void removePersistentTracksAtIndex(const unsigned int idx);
      // Remove invalid new persistent features at given indexes
      void removeNewPersistentTracksAtIndexes(const std::vector<unsigned int> invalid_tracks_idx);

      /** @brief Get indexes of SLAM tracks that were just lost
       */
      std::vector<unsigned int> getLostSlamTrackIndexes() const;

      /**
       * Sorts input matches in various track types.
       *
       * Manages the various types of feature tracks based on the latest set of
       *  matches, e.g.:
       *    - append match to an existing track,
       *    - detect the end of a track,
       *    - change a track's type (e.g. opportunistic to persistent).
       *    
       * @param[in] matches Latest output from the feature tracker (image front end)
       * @param[in] cam_rots Camera orientations (up to current frame), for MSCKF baseline check
       * @param[in] n_poses_max Maximum number of poses in sliding window.
       * @param[in] n_slam_features_max Maximum number of SLAM features allowed in
       *                                the state vector.
       * @param[in] min_track_length Minimum track length for a visual feature to be
       *                             processed.
       * @param[out] img Output image for GUI
       */ 
      void manageTracks(MatchList& matches,
          const AttitudeList cam_rots,
          const int n_poses_max,
          const int n_slam_features_max,
          const int min_track_length,
          TiledImage& img);

      std::vector<int> featureTriangleAtPoint(const Feature& lrf_img_pt, TiledImage& img) const;
      void draw_delaunay(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color) const;
    private:
      Camera camera_;
      TrackList slam_trks_;    // persistent feature tracks (excluding new ones)
      // New SLAM tracks (both semi-infinite depths uncertainty and MSCKF)
      TrackList new_slam_trks_;

      /** @brief Indexes of SLAM tracks that were just lost.
       */
      std::vector<unsigned int> lost_slam_idxs_;

      TrackList opp_trks_;      // opportunistic feature tracks
      TrackList msckf_trks_n_;  // Normalized tracks for MSCKF update
      // New SLAM features initialized with semi-infinite depth uncertainty
      // (regular SLAM)
      TrackList new_slam_std_trks_n_;
      // New SLAM features initialized with MSCKF tracks (MSCKF-SLAM)
      TrackList new_slam_msckf_trks_n_;
      // Minimum baseline for MSCKF measurements (in normal plane)
      double min_baseline_n_ = 0.0;

      /** \brief MSCKF baseline check.
       *
       *  Checks if the camera has enough baseline for triangulation (for
       *  MSCKF). This is done by rectifying all the feature observations
       *  as if the camera maintained the same attitude, here that of the
       *  last camera, and comparing the maximum displacement in pixels
       *  to a threshold. This is done in pixel space to be independent
       *  of scale. 
       *
       *  @param track Feature measurement list (size n_obs)
       *  @param cam_att Camera attitude list (size n_quats >= n_obs)
       *  @return True if baseline is larger than threshold
       */
      bool checkBaseline(const Track& track, const AttitudeList& C_q_G) const;

      /**
       * Track manager debug image.
       *
       * @param[in] img Camera image
       * @param[in] min_track_length Minimum track length for a visual feature to be
       *                             processed.
       */
      void plotFeatures(TiledImage& img, const int min_track_length);
  };
} // namespace x

#endif // TRACK_MANAGER_H
