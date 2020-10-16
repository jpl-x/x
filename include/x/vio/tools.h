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

#ifndef _TOOLS_H_
#define _TOOLS_H_

#include <x/vio/types.h>

namespace x
{
template <typename Derived>
Eigen::Block<const Derived> const MatrixBlock(
    Eigen::MatrixBase<Derived> const& matrix, int row, int col, int rows, int cols) {
  return Eigen::Block<const Derived>(matrix.derived(), row, col, rows, cols);
}

template <typename Derived>
Eigen::Block<Derived> MatrixBlock(
    Eigen::MatrixBase<Derived>& matrix, int row, int col, int rows, int cols) {
  return Eigen::Block<Derived>(matrix.derived(), row, col, rows, cols);
}

template <typename Derived>
Eigen::Block<Derived> MatrixBlock(Eigen::MatrixBase<Derived>& matrix,
                                  int row,
                                  int col) {
  return MatrixBlock(matrix.derived(),
                     row,
                     col,
                     matrix.rows() - row,
                     matrix.cols() - col);
}
template <typename Derived>
Eigen::Block<const Derived> const MatrixBlock(
    Eigen::MatrixBase<Derived> const& matrix, int row, int col)
{
  return MatrixBlock(matrix.derived(),
                     row,
                     col,
                     matrix.rows() - row,
                     matrix.cols() - col);
}

struct Skew {
  Eigen::Matrix3d matrix;
  Eigen::Vector3d vector;
  Skew(double x, double y, double z) : matrix(), vector(x, y, z)
  {
    matrix << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1),
        vector(0), 0;
  }
};

struct Quatern {
  x::Quaternion q;
  x::Quaternion const& operator()(x::Attitude const& attitude) {
    q.x() = attitude.ax;
    q.y() = attitude.ay;
    q.z() = attitude.az;
    q.w() = attitude.aw;
    return q;
  }

  x::Quaternion const& operator()(Eigen::Vector4d const& attitude) {
    q.x() = attitude(0, 0);
    q.y() = attitude(1, 0);
    q.z() = attitude(2, 0);
    q.w() = attitude(3, 0);
    return q;
  }
};
} // end namespace x
#endif /* _TOOLS_H_ */
