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

#ifndef X_VIO_TYPES_H
#define X_VIO_TYPES_H

#include <x/vision/types.h>
#include <x/vision/feature.h>
#include <x/vision/tiled_image.h>

#include <vector>
#include <Eigen/Dense>

/**
 * This header defines common types used in xVIO.
 */
namespace x
{
  /**
   * User-defined parameters
   */
  struct Params
  {
    Vector3 p;
    Vector3 v;
    Quaternion q;
    Vector3 b_w;
    Vector3 b_a;
    double cam_fx;
    double cam_fy;
    double cam_cx;
    double cam_cy;
    double cam_s;
    int img_height;
    int img_width;
    Vector3 p_ic;
    Quaternion q_ic;
   
    /**
     * Standard deviation of feature measurement [in normalized coordinates]
     */
    double sigma_img;

    /**
     * Standard deviation of range measurement noise [m].
     */
    double sigma_range;

    Quaternion q_sc;
    Vector3 w_s;
    double n_a;
    double n_ba;
    double n_w;
    double n_bw;
    int fast_detection_delta;
    bool non_max_supp;
    int block_half_length;
    int margin;
    int n_feat_min;
    int outlier_method;
    double outlier_param1;
    double outlier_param2;
    int n_tiles_h;
    int n_tiles_w;
    int max_feat_per_tile;
    double time_offset;

    /**
     * Maximum number of poses in the sliding window.
     */
    int n_poses_max;

    /**
     * Maximum number of SLAM features.
     */
    int n_slam_features_max;

    /**
     * Initial inverse depth of SLAM features [1/m].
     *
     * This is when SLAM features can't be triangulated for MSCKK-SLAM. By
     * default, this should be 1 / (2 * d_min), with d_min the minimum
     * expected feature depth (2-sigma) [Montiel, 2006]
     */
    double rho_0;

    /**
     * Initial standard deviation of SLAM inverse depth [1/m].
     *
     * This is when SLAM features can't be triangulated for MSCKK-SLAM. By
     * default, this should be 1 / (4 * d_min), with d_min the minimum
     * expected feature depth (2-sigma) [Montiel, 2006].
     */
    double sigma_rho_0;

    /**
     * Number of IEKF iterations (EKF <=> 1).
     */
    int iekf_iter;

    /**
     * Minimum baseline to trigger MSCKF measurement (pixels).
     */
    double msckf_baseline;

    /**
     * Minimum track length for a visual feature to be processed.
     */
    int min_track_length;

    double traj_timeout_gui;
    bool init_at_startup;
  };

  /**
   * MSCKF-SLAM matrices
   */
  struct MsckfSlamMatrices {
    /**
     * H1 matrix in Li's paper (stacked for all features)
     */
    Matrix H1;

    /**
     * H2 matrix in Li's paper (stacked for all features)
     */
    Matrix H2;

    /**
     * z1 residual vector in Li's paper (stacked for all features)
     */
    Matrix r1;

    /**
     * New SLAM feature vectors (stacked for all features).
     *
     * Features' inverse-depth coordinates, assuming the latest camera as an
     * anchor. These estimates are taken from the MSCKF triangulation prior.
     */
    Matrix features;
  };

  /**
   * Range measurement.
   *
   * Coming from a Laser Range Finder (LRF).
   */
  struct RangeMeasurement
  {
    double timestamp { kInvalid };

    double range { 0.0 };

    /**
     * Image coordinate of the LRF beam.
     *
     * This is assumed to be a single point in the image, i.e. the LRF axis
     * passes by the optical center of the camera.
     */
    Feature img_pt { Feature() };

    /**
     * Normalized image coordinates of the LRF beam.
     *
     * Follows the same assumptions as img_pt.
     */
    Feature img_pt_n { Feature() };
  };

  /**
   * Sun angle measurement.
   *
   * Coming from a sun sensor.
   */
  struct SunAngleMeasurement
  {
    double timestamp { kInvalid };
    double x_angle { 0.0 };
    double y_angle { 0.0 };
  };

  /**
   * VIO update measurement.
   *
   * This struct includes all sensor the measurements that will processed to
   * create an EKF update. This should be at least an image (or a set of visual
   * matches) in xVIO, along side optional additional sensor measurements that
   * are assumed to be synced with the visual data (LRF and sun sensor).
   */
  struct VioMeasurement {
    /**
     * Timestamp.
     *
     * This timestamp is the visual measurement timestamp (camera or match
     * list). Range and sun angle measurement timestamps might different but
     * will be processed at the sam timestamp as a single EKF update.
     */
    double timestamp { kInvalid };

    /**
     * Sequence ID.
     *
     * Consecutively increasing ID associated with the visual measurement
     * (matches or image).
     */
    unsigned int seq { 0 };
    
    /**
     * Visual match measurements.
     *
     * Output of a visual feature tracker (or simulation).
     */
    MatchList matches;

    /**
     * Image measurement.
     *
     * Will only be used if the visual match list struct member has size zero,
     * in which case a feature track will run on that image.
     */
    TiledImage image;

    /**
     * Range measurement.
     */
    RangeMeasurement range;

    /**
     * Sun angle measurement.
     */
    SunAngleMeasurement sun_angle;

    /**
     * Default constructor.
     */
    VioMeasurement() {}

    /**
     * Full constructor.
     */
    VioMeasurement(const double timestamp,
                   const unsigned int seq,
                   const MatchList& matches,
                   const TiledImage& image,
                   const RangeMeasurement& range,
                   const SunAngleMeasurement& sun_angle)
      : timestamp { timestamp }
      , seq { seq }
      , matches { matches }
      , image { image }
      , range { range }
      , sun_angle { sun_angle }
      {}
  };
  
  using Vector3dArray = std::vector<Eigen::Vector3d>;

  using Time = double;
}

#endif // X_VIO_TYPES_H
