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

#include <x/vio/solar_update.h>
#include <x/vio/tools.h>
#include <x/ekf/state.h>
#include <boost/math/distributions.hpp>

using namespace x;
using namespace Eigen;

SolarUpdate::SolarUpdate(const x::SunAngleMeasurement& angle,
                         const x::Quaternion& quat,
                         const MatrixXd& cov_s)
{
  // Initialize Kalman update matrices
  const size_t cols = cov_s.cols();
  jac_ = MatrixXd::Zero(2, cols);
  cov_m_diag_ = VectorXd::Ones(2);
  res_ = MatrixXd::Zero(2, 1);

  // Construct update
  processSunAngle(angle, quat);
}

void SolarUpdate::processSunAngle(const x::SunAngleMeasurement& angle,
                                  const x::Quaternion& quat)
{
  //========================
  // Calibration parameters
  //========================
  // TODO(Harel) Import these from param file

  // Sun angle measurement noise (variance in deg^2)
  const double var_sun_angle = 10000 * 0.01777777777;

  // Sun sensor orientation relative to IMU
  x::Quaternion S_q_I(0.360346005598587, -0.063338979194957, 0.007502445522018, 0.930635612981541);

  // Sun vector in world frames (should be calibrated before each flight)
  // z+ points to zenith, x+ points north
  Vector3d G_sun(-0.29385515271891938, -0.55080445540063927, 0.78119370269565391);
  G_sun.normalize();

  //==========
  // Residual
  //==========
  Vector2d angles(angle.x_angle, angle.y_angle);
  Vector3d S_sun_hat = S_q_I.normalized().toRotationMatrix().transpose() *
                              quat.normalized().toRotationMatrix().transpose() *
                              G_sun;
  S_sun_hat.normalize();
  constexpr double RAD2DEG = 57.2957795130;
  Vector2d angles_hat(RAD2DEG * std::atan2(S_sun_hat(0), S_sun_hat(2)),
                             RAD2DEG * std::atan2(S_sun_hat(1), S_sun_hat(2)));

  res_ = angles - angles_hat;

  //============================
  // Measurement Jacobian matrix
  //============================

  MatrixXd mat(MatrixXd::Zero(2, 3));
  mat(0, 0) = S_sun_hat(2) / (std::pow(S_sun_hat(0), 2) + std::pow(S_sun_hat(2), 2));
  mat(1, 1) = S_sun_hat(2) / (std::pow(S_sun_hat(1), 2) + std::pow(S_sun_hat(2), 2));
  mat(0, 2) = -S_sun_hat(0) / (std::pow(S_sun_hat(0), 2) + std::pow(S_sun_hat(2), 2));
  mat(1, 2) = -S_sun_hat(1) / (std::pow(S_sun_hat(1), 2) + std::pow(S_sun_hat(2), 2));

  Vector3d skew_vector = quat.normalized().toRotationMatrix().transpose() * G_sun;

  SolarUpdate::SunAngleJacobian J_attitude = RAD2DEG * mat * S_q_I.normalized().toRotationMatrix().transpose() *
                                        x::Skew(skew_vector(0), skew_vector(1), skew_vector(2)).matrix;

  jac_.block<2, kJacCols>(0, kIdxQ) = J_attitude;

  //==================
  // Noise covariance
  //==================

  cov_m_diag_ = var_sun_angle * VectorXd::Ones(2);
}
