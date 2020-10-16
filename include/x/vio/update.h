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

#ifndef X_VIO_UPDATE_H_
#define X_VIO_UPDATE_H_

#include <x/vio/types.h>
#include <x/vision/triangulation.h>

namespace x
{
  /**
   * An abstract Kalman update class.
   *
   * It defines the members and getters to be used by any derived
   * update class: Jacobian matrix, residual and measurement
   * covariance diagonal.
   */
  class Update
  {
    public:

      /******************************* Getters ********************************/

      /**
       * Returns a constant reference to the residual vector.
       */
      const Matrix& getResidual() const { return res_; };

      /**
       * Returns a constant reference to the Jacobian matrix.
       */
      const Matrix& getJacobian() const { return jac_; };

      /**
       * Returns a constant reference to the diagonal of the measurement
       * covariance matrix.
       */
      const Eigen::VectorXd& getCovDiag() const { return cov_m_diag_; };
      
    protected:
      // Column number for generic Jacobian block (3 error states)
      static unsigned constexpr kJacCols = 3;

      // Vision Jacobian block
      static unsigned constexpr kVisJacRows = 2;
      using VisJacBlock = Eigen::Matrix<double,
                                        kVisJacRows,
                                        kJacCols>;

      /**
       * Residual vector.
       */
      Matrix res_;

      /**
       * Jacobian matrix.
       */
      Matrix jac_;

      /**
       * Diagonal of the measurement covariance matrix.
       */
      Eigen::VectorXd cov_m_diag_;
  };
}

#endif  // X_VIO_UPDATE_H_
