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

#ifndef X_EKF_UPDATER_H_
#define X_EKF_UPDATER_H_

#include <x/ekf/state.h>

namespace x
{
  /**
   * The xEKF updater abstract class.
   *
   * An abstract class that implements the (iterated) EKF update and defines
   * the pure virtual methods for pre-update work, update matrices construction
   * and post-update work. These methods are to be implemented by a derived
   * concrete class which is application-specific (e.g VIO or GPS update).
   */
  class Updater
  {
    public:
       /**
       * A pure virtual method: to get the measurement timestamp.
       */
      virtual double getTime() const = 0;

      /** 
       * Updates input state with current measurement member.
       *
       * This is the most important function. xEKF calls this function to update
       * the input state based on the current measurement.
       *
       * @param[in] state Update state
       */
      void update(State& state);

    protected:
      /**
       * Number of IEKF iterations.
       *
       * The filter simplifies to EKF when equals to 1.
       */
      int iekf_iter_ { 1 };

      /**
       * Compute Kalman update and apply to state.
       *
       * @param[in,out] state Update state 
       * @param[in] H Measurement Jacobian
       * @param[in] res Measurement residual
       * @param[in] R Measurement covariance matrix
       * @param[in,out] correction Correction vector (total correction for IEKF)
       * @param[in] cov_update False if the covariance should not be updated (only
       *                       (for IEKF iterations, default = true)
       */
      void applyUpdate(State& state,
                       const Eigen::MatrixXd& H,
                       const Eigen::MatrixXd& res,
                       const Eigen::MatrixXd& R,
                       Matrix& correction_total,
                       const bool cov_update = true);
      
      /**
       * A pure virtual method for measurement processing.
       *
       * Only happens once, and is stored if measurement later re-applied. Here this
       * where image processing takes place.
       *
       * @param[in] state State prior.
       */
      virtual void preProcess(const State& state) = 0;

      /**
       * A pure virtual method for pre-update work
       *
       * Stuff that needs happen before the Kalman update. This function will NOT
       * be called at  each IEKF iteration. Here, this corresponds to state
       * management.
       *
       * @param[in,out] state Update state
       * @return True if an update should be constructed
       */
      virtual bool preUpdate(State& state) = 0;

      /**
       * A pure virtual method for update construction.
       *
       * Prepares measurement Jacobians, residual and noice matrices and applies
       * (iterated) EKF update. This function WILL be called at each IEKF iteration.
       *
       * @param[in] state Update state
       * @param[out] h Measurement Jacobian matrix
       * @param[out] res Measurement residual vector
       * @param[out] r Measurement noise covariance matrix
       */
      virtual void constructUpdate(const State& state,
                                   Matrix& h,
                                   Matrix& res,
                                   Matrix& r) = 0;
      
      /**
       * A pure virtual method for post-update work.
       *
       * Stuff that need to happen after the Kalman update. Here this SLAM feature
       * initialization.
       *
       * @param[in,out] state Update state.
       * @param[in] correction Kalman state correction.
       */
      virtual void postUpdate(State& state, const Matrix& correction) = 0;
  };
} // namespace x

#endif  // X_EKF_UPDATER_H_
