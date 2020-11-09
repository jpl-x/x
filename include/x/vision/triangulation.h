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

#ifndef X_TRIANGULATION_H_
#define X_TRIANGULATION_H_

#include <x/vision/types.h>
#include <x/vision/track.h>
#include <Eigen/Core>

/**
 * A triangulation class.
 *
 * A triangulation object is defined by a series of camera poses.
 * It include methods to  compute the 3D coordinates of a feature
 * from on a series of 2D image observations taken at the object's
 * poses. Methods include Direct Linear Transform and
 * Gauss-Newton non-linear least squares.
 */
namespace x
{
class Triangulation
{
public:
  
  /**
   * Constructors
   */
  Triangulation(const x::QuaternionArray& quat_l,
                const x::Vector3Array& pos_l,
                const unsigned int max_iter = 10 ,
                const double term = 0.00001);

  Triangulation(const x::AttitudeList& attitudes,
                const x::TranslationList& translations,
                const unsigned int max_iter = 10 ,
                const double term = 0.00001);
  
  /**
   * 2-view linear triangulation.
   *
   * This function is a wrapper around OpenCV's Direct Linear
   * Transform (DLT) implementation.
   *
   * @param[in] obs1, obs2 Image feature coordinates
   * @param[in] i1, i2 Matching indices in the pose vectors
   * @param[out] pt_xyz Triangulated cartesian 3D coordinates
   */
  void triangulateDlt(const x::Feature& obs1,
                      const x::Feature& obs2,
                      const int i1,
                      const int i2,
                      Eigen::Vector3d& pt_xyz) const;

  /**
   * Non-linear Gauss-Newton triangulation.
   *
   * This function is an implementation of the non-linear least
   * squares multi-view triangulation from the 2007 MSCKF paper.
   * The last track observation is assumed to correspond to the
   * last pose.
   *
   * @param[in] track Feature observations (size <= n_poses_)
   * @param[out] pt_ivd Triangulated inverse-depth coordinates in
   *                    last frame
   */
  void triangulateGN(const x::Track& track,
                     Eigen::Vector3d& pt_ivd) const;
private:

  /**
   * Number of poses.
   */
  size_t n_poses_;
  
  /**
   * Camera rotation matrices.
   */
  std::vector<Eigen::Matrix3d> rotations_;
  
  /**
   * Camera positions.
   */
  Vector3Array positions_;

  /**
   * Camera projection matrices
   */
  std::vector<cv::Mat> projs_;

  /**
   * Maximum number of iterations.
   * Only used in non-linear methods.
   */
  unsigned int max_iter_;

  /**
   * Termination criterion.
   * Residual change threshold only used in non-linear methods.
   */
  double term_;

  /**
   * @brief Convert quaternion and position to projection matrix
   *
   * @param[in]  rot Rotation matrix
   * @param[in]  pos  Position
   * @param[out] proj 3x4 projection matrix
   */
  void pose2proj(const Eigen::Matrix3d& rot,
                 const Eigen::Vector3d& pos,
                 cv::Mat& proj) const;
};
}

#endif  // X_TRIANGULATION_H_
