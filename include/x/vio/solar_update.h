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

#ifndef X_VIO_SOLAR_UPDATE_H_
#define X_VIO_SOLAR_UPDATE_H_

#include <x/vio/update.h>
#include <x/vio/types.h>

namespace x
{
  /**
   * Solar update.
   *
   * As presented in 2020 IEEE Aerospace paper.
   */
  class SolarUpdate : public Update
  {
    public:

      /**
       * Constructor.
       *
       * Does the full update matrix construction job.
       *
       * @param[in] range_meas Range measurement.
       * @param[in] quats Camera quaternion states.
       * @param[in] cov_s Error state covariance matrix.
       */
      SolarUpdate(const x::SunAngleMeasurement& angle,
                  const x::Quaternion& quat,
                  const Eigen::MatrixXd& cov_s);

    private:
      
      using SunAngleJacobian = Eigen::Matrix<double, 2, kJacCols>;

      /** @brief Processes a sun sensor measurement.
       *
       * @param[in] angle Sun angle measurement
       * @param[in] quat IMU attitude in state vector
       */
      void processSunAngle(const x::SunAngleMeasurement& angle,
                           const x::Quaternion& quat);
  };
}

#endif  // X_VIO_SOLAR_UPDATE_H_
