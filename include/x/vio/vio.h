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

#ifndef X_VIO_VIO_H_
#define X_VIO_VIO_H_

#include <x/vio/types.h>
#include <x/vio/vio_updater.h>
#include <x/vio/state_manager.h>
#include <x/vio/track_manager.h>
#include <x/ekf/ekf.h>
#include <x/vision/camera.h>
#include <x/vision/tiled_image.h>
#include <x/vision/tracker.h>

namespace x {
  class VIO
  {
    public:
      VIO();
      bool isInitialized() const;
      void setUp(const Params& params);
      void setLastRangeMeasurement(RangeMeasurement range_measurement);
      void setLastSunAngleMeasurement(SunAngleMeasurement angle_measurement);
      void initAtTime(double now);
      void getMsckfFeatures(Vector3dArray& inliers, Vector3dArray& outliers);

      /**
       * Pass IMU measurements to EKF for propagation.
       *
       * @param[in] timestamp
       * @param[in] msg_seq Message ID.
       * @param[in] w_m Angular velocity (gyroscope).
       * @param[in] a_m Specific force (accelerometer).
       * @return The propagated state.
       */
      State processImu(const double timestamp,
          const unsigned int seq,
          const Vector3& w_m,
          const Vector3& a_m);

      /**
       * Creates an update measurement from image and pass it to EKF.
       *
       * @param[in] timestamp Image timestamp.
       * @param[in] seq Image sequence ID.
       * @param[in,out] match_img Image input, overwritten as tracker debug image
       *                          in output.
       * @param[out] feature_img Track manager image output.
       * @return The updated state, or invalide if the update could not happen.
       */
      State processImageMeasurement(double timestamp,
                                    const unsigned int seq,
                                    TiledImage& match_img,
                                    TiledImage& feature_img);

      /**
       * Creates an update measurement from visual matches and pass it to EKF.
       *
       * @param[in] timestamp Image timestamp.
       * @param[in] seq Image sequence ID.
       * @param[in,out] match_vector Feature matches vector.
       * @param[out] match_img Tracker image output.
       * @param[out] feature_img Track manager image output.
       * @return The updated state, or invalid if the update could not happen.
       */
      State processMatchesMeasurement(double timestamp,
                                      const unsigned int seq,
                                      const std::vector<double>& match_vector,
                                      TiledImage& match_img,
                                      TiledImage& feature_img);

      /**
       * Compute cartesian coordinates of SLAM features for input state.
       *
       * @param[in] state Input state.
       * @return A vector with the 3D cartesian coordinates.
       */
      std::vector<Vector3>
        computeSLAMCartesianFeaturesForState(const State& state);

    private:
      /**
       * Extended Kalman filter estimation back end.
       */
      Ekf ekf_;
      
      /**
       * VIO EKF updater.
       *
       * Constructs and applies an EKF update from a VIO measurement. The EKF
       * class owns a reference to this object through Updater abstract class,
       * which it calls to apply the update.
       */
      VioUpdater vio_updater_;

      Params params_;

      /**
       * Minimum baseline for MSCKF (in normalized plane).
       */
      double msckf_baseline_n_;

      Camera camera_;
      Tracker tracker_;
      TrackManager track_manager_;
      StateManager state_manager_;
      RangeMeasurement last_range_measurement_;
      SunAngleMeasurement last_angle_measurement_;
      bool initialized_ { false };

      /**
       * Import a feature match list from a std::vector.
       *
       * @param[in] match_vector Input vector of matches.
       * @param[in] seq Image sequence ID.
       * @param[out] img_plot Debug image.
       * @return The list of matches.
       */
      MatchList importMatches(const std::vector<double>& match_vector,
                              const unsigned int seq,
                              x::TiledImage& img_plot) const;

      double GetTimestamp(size_t index);
  };
} // namespace x

#endif  // X_VIO_VIO_H_
