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

#ifndef X_COMMON_EIGEN_MATRIX_BASE_PLUGIN_H_
#define X_COMMON_EIGEN_MATRIX_BASE_PLUGIN_H_

/*******************************************************************************
 * This header defines custom extensions of the Eigen MatrixBase class for xEKF.
 ******************************************************************************/

/**
 * Converts a 3-vector to the associated 3x3 cross-product matrix.
 *
 * The matrix must be a vector of size 3. The cross-product matrix is a 3x3
 * skew-symmetric matrix.
 *
 * @return The 3x3 cross-product matrix
 */
inline Matrix< Scalar, 3, 3 >
toCrossMatrix() const {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
  Matrix< Scalar, 3, 3 > mat;
  mat <<            0.0, -derived().z(),  derived().y(),
          derived().z(),            0.0, -derived().x(),
         -derived().y(),  derived().x(),          0.0;
  return mat;
}

/**
 * Converts a 3-vector to the associated 4x4 quaternion differentiation matrix.
 *
 * The matrix must be a vector of size 3 and represent angular rate.
 *
 * This is the transform from Eq. (108) in "Indirect Kalman Filter for 3D
 * Attitude Estimation" by Nik Trawny and Stergios Roumeliotis (Univ. of
 * Minnesota). This is equivalent to Eq. (2.17) of xVIO tech report,
 * adapted to the (x,y,z,w) quaternion coefficient order from Eigen.
 *
 * @return The 4x4 quaternion differentiation matrix.
 */
inline Matrix< Scalar, 4, 4 >
toOmegaMatrix() const {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
  Matrix< Scalar, 4, 4 > mat;
  mat <<            0.0,  derived().z(), -derived().y(), derived().x(),
         -derived().z(),            0.0,  derived().x(), derived().y(),
          derived().y(), -derived().x(),            0.0, derived().z(),
         -derived().x(), -derived().y(), -derived().z(),           0.0;
  return mat;
}

#endif  // X_COMMON_EIGEN_MATRIX_BASE_PLUGIN_H_
