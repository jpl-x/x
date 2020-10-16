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

#ifndef X_COMMON_TYPES_H
#define X_COMMON_TYPES_H

#include <Eigen/Dense>

namespace x {

  using Vector3 = Eigen::Vector3d;
  using Quaternion = Eigen::Quaterniond;
  using Matrix = Eigen::MatrixXd;

  /**
   * A structure to pass IMU noise parameters.
   *
   * Default values: ADXRS610 gyros, MXR9500G/M accels (Astec).
   */
  struct ImuNoise {
    /**
     * Gyro noise spectral density [rad/s/sqrt(Hz)]
     */
    double n_w = 0.0083;

    /**
     * Gyro bias random walk [rad/s^2/sqrt(Hz)]
     */
    double n_bw = 0.00083;

    /**
     * Accel noise spectral density [m/s^2/sqrt(Hz)]
     */
    double n_a = 0.0013;

    /**
     * Accel bias random walk [m/s^3/sqrt(Hz)]
     */
    double n_ba = 00013;
  };
  
  /**
   * Denotes an invalid object through a -1 timestamp.
   */
  constexpr double kInvalid = -1.0;


} // namespace x

#endif // #ifndef X_COMMON_TYPES_H
