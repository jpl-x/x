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

#include <x/ekf/state.h>
#include <iomanip>

using namespace x;
 
State::State(const size_t n_poses, const size_t n_features) {
  // Allocate and initialize vision with zero values
  // Note: other states are already initialed at default values by the default
  // constructor.
  p_array_ = Matrix::Zero(n_poses * 3, 1); 
  q_array_ = Matrix::Zero(n_poses * 4, 1);
  f_array_ = Matrix::Zero(n_features * 3, 1);
  const size_t n = kSizeCoreErr + n_poses * 6 + n_features * 3;
  cov_ = Matrix::Identity(n, n);
}   

State::State(const double time,
             const unsigned int seq,
             const Vector3& p,
             const Vector3& v,
             const Quaternion& q,
             const Vector3& b_w,
             const Vector3& b_a,
             const Matrix& p_array,
             const Matrix& q_array,
             const Matrix& f_array,
             const Matrix& cov,
             const Quaternion& q_ic,
             const Vector3& p_ic,
             const Vector3& w_m,
             const Vector3& a_m)
  : time_ { time }
  , seq_ { seq }
  , p_ { p }
  , v_ { v }
  , q_ { q }
  , b_w_ { b_w }
  , b_a_ { b_a }
  , p_array_ { p_array }
  , q_array_ { q_array }
  , f_array_ { f_array }
  , cov_ { cov }
  , q_ic_ { q_ic }
  , p_ic_ { p_ic }
  , w_m_ { w_m }
  , a_m_ { a_m }
{
  // The number of positions and orientations should always be the same
  assert( p_array.rows() / 3 == q_array.rows() / 4
       || p_array.cols() == 1 || q_array.cols() == 1);
}

double State::getTime() const {
  return time_;
}

unsigned int State::getSeq() const {
  return seq_;
}

Vector3 State::getPosition() const {
  return p_;
}

Quaternion State::getOrientation() const {
  return q_;
}

Matrix State::getPositionArray() const {
  return p_array_;
}

Matrix State::getOrientationArray() const {
  return q_array_;
}

Matrix State::getFeatureArray() const {
  return f_array_;
}

Quaternion State::getOrientationExtrinsics() const {
  return q_ic_;
}

Vector3 State::getPositionExtrinsics() const {
  return p_ic_;
}

Matrix State::getCovariance() const {
  return cov_;
}

Matrix State::getPoseCovariance() const {
  Matrix pose_cov(6,6);

  // Fill in the four 3x3 blocks from the full covariance matrix
  pose_cov.topLeftCorner(3,3)     = cov_.topLeftCorner(3,3);
  pose_cov.bottomRightCorner(3,3) = cov_.block<3,3>(kIdxQ,kIdxQ);
  pose_cov.topRightCorner(3,3)    = cov_.block<3,3>(kIdxP,kIdxQ);
  pose_cov.bottomLeftCorner(3,3)  = cov_.block<3,3>(kIdxQ,kIdxP);

  return pose_cov;
}

Matrix& State::getCovarianceRef() {
  return cov_;
}

void State::setTime(const double time) {
  time_ = time;
}

void State::setPositionArray(const Matrix& p_array) {
  // The number of positions and orientations should always be the same
  assert( p_array.rows() / 3 == q_array_.rows() / 4 || p_array.cols() == 1);
  p_array_ = p_array;
}

void State::setOrientationArray(const Matrix& q_array) {
  // The number of positions and orientations should always be the same
  assert( q_array.rows() / 4 == p_array_.rows() / 3 || q_array.cols() == 1);
  q_array_ = q_array;
}

void State::setFeatureArray(const Matrix& f_array) {
  f_array_ = f_array;
}

void State::setCovariance(const Matrix& cov) {
  cov_ = cov;
}

void State::setImu(const double time,
                   const unsigned int seq,
                   const Vector3& w_m,
                   const Vector3& a_m) {
  time_ = time;
  seq_ = seq;
  w_m_ = w_m;
  a_m_ = a_m;
}

void State::setStaticStatesFrom(const State& state) {
  b_w_ = state.b_w_;
  b_a_ = state.b_a_;
  q_ic_ = state.q_ic_;
  p_ic_ = state.p_ic_;
  p_array_ = state.p_array_;
  q_array_ = state.q_array_;
  f_array_ = state.f_array_;
}

void State::reset() {
  time_ = kInvalid;
}

int State::nPosesMax() const {
  return p_array_.rows() / 3;
}

int State::nFeaturesMax() const {
  return f_array_.rows() / 3;
}

int State::nErrorStates() const {
  return kSizeCoreErr
       + p_array_.rows()
       + q_array_.rows() / 4 * 3
       + f_array_.rows();
}

void State::computeUnbiasedImuMeasurements(Vector3& e_w, Vector3& e_a) const {
  // Bias-free gyro measurement
  e_w = w_m_ - b_w_;
  // Bias-free accel measurement
  e_a = a_m_ - b_a_;
}

Attitude State::computeCameraAttitude() const {
  const Quaternion quat = q_.normalized() * q_ic_.normalized();
  return Attitude(quat.x(), quat.y(), quat.z(), quat.w());
}

Vector3 State::computeCameraPosition() const {
  return p_ + q_.normalized().toRotationMatrix() * p_ic_;
}

Quaternion State::computeCameraOrientation() const {
  return q_.normalized() * q_ic_.normalized();
}

void State::correct(const Eigen::VectorXd& correction) {
  // Check correction size
  assert(correction.rows() == nErrorStates() && correction.cols() == 1);

  // Parse state corrections
  const int n_p_array = p_array_.rows();
  const int n_delta_theta_array = n_p_array;
  const int n_f_array = f_array_.rows();
  const Vector3 delta_p = correction.segment(kIdxP, 3);
  const Vector3 delta_v = correction.segment(kIdxV, 3);
  const Vector3 delta_theta = correction.segment(kIdxQ, 3);
  const Vector3 delta_b_w = correction.segment(kIdxBw, 3);
  const Vector3 delta_b_a = correction.segment(kIdxBa, 3);
  const Eigen::VectorXd delta_p_array
    = correction.segment(kSizeCoreErr, n_p_array);
  const Eigen::VectorXd delta_theta_array
    = correction.segment(kSizeCoreErr + n_p_array, n_delta_theta_array);
  const Eigen::VectorXd delta_f_array
    = correction.segment(kSizeCoreErr + n_p_array + n_delta_theta_array, n_f_array);

  // Correct all states but quaternions (additive correction)
  p_ += delta_p;
  v_ += delta_v;
  b_w_ += delta_b_w;
  b_a_ += delta_b_a;
  p_array_ += delta_p_array;
  f_array_ += delta_f_array;

  // Correct current orientation (quaternion multiplication)
  const Quaternion delta_q = errorQuatFromSmallAngles(delta_theta);
  q_ = (q_ * delta_q).normalized();

  // Correction orientation array
  const int n_poses = n_p_array / 3;
  for (int i = 0; i < n_poses; ++i) {
    // Delta quaternion
    const Vector3 delta_theta_i
      = delta_theta_array.segment(3*i, 3);
    const Quaternion delta_q_i = errorQuatFromSmallAngles(delta_theta_i);

    // Prior quaternion estimate (stored as x,y,z,w in state array)
    // TODO(jeff) Change to w,z,y,z. This is confusing wrt Eigen.
    Quaternion q_i(q_array_(4*i+3, 0),  // w
                   q_array_(4*i, 0),    // x
                   q_array_(4*i+1, 0),  // y
                   q_array_(4*i+2, 0)); // z

    // Quaternion posterior
    q_i = q_i * delta_q_i;
    q_i.normalize();

    // Insert in orientation array
    q_array_.block<4, 1>(4*i, 0) << q_i.x(), q_i.y(), q_i.z(), q_i.w();
  }
}

std::string State::toString() const {
  std::stringstream s;
s << "---------------------------------------"
     "---------------------------------------" << std::endl;
  s << "Timestamp: " << std::setprecision(19) << time_ << std::endl;
  s <<std::setprecision(6);
  s << "p [x,y,z]: " << p_.transpose() << std::endl;
  s << "v [x,y,z]: " << v_.transpose() << std::endl;
  s << "q [w,x,y,z]: " << q_.w() << " " <<  q_.x() << " " << q_.y() << " " << q_.z() << std::endl;
  s << "b_w [x,y,z]: " << b_w_.transpose() << std::endl;
  s << "b_a [x,y,z]: " << b_a_.transpose() << std::endl;
  s << "p_array: " << p_array_.transpose() << std::endl;
  s << "q_array: " << q_array_.transpose() << std::endl;
  s << "f_array: " << f_array_.transpose() << std::endl;
  s << "---------------------------------------"
       "---------------------------------------" << std::endl;
  return s.str();
}

Quaternion State::errorQuatFromSmallAngles(const Vector3& delta_theta) const {
  if (delta_theta.norm() == 0.0)
    return Quaternion::Identity();
  else {
    const Vector3 axis = delta_theta.normalized();
    const double angle = delta_theta[0] / axis[0];
    const Eigen::AngleAxisd aa(angle, axis);
    const Quaternion delta_q(aa);
    return delta_q;
  }
}
