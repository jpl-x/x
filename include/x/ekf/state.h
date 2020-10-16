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

#ifndef X_EKF_STATE_H_
#define X_EKF_STATE_H_

#include <x/common/types.h>
#include <x/vio/types.h>

namespace x
{
  /**
   * Start indices and total size of the core (inertial) error states. 
   *
   * Refer to the State class for the definition of each state.
   */
  enum {
    kIdxP = 0,
    kIdxV = 3,
    kIdxQ = 6,
    kIdxBw = 9,
    kIdxBa = 12,
    kSizeCoreErr = 21
  };

  /**
   * The xEKF inertial state class.
   *
   * This class stores the inertial, or core, states propagated by xEKF, the
   * error covariance matrix, and the IMU measurements at a given time. This is
   * an abstract class. Vision states are defined as optional additional dynamic
   * states. The errors states are in the following order in the covariance
   * matrix: p, v, q, b_w, b_a, p_array, q_array, f_array.
   * {W} world frame
   * {I} IMU frame
   */
  class State
  {
    public:
      /////////////////////////////// CONSTRUCTORS /////////////////////////////
      /**
       * Default constructor (zero state).
       *
       * Zero states. The dynamic vision state arrays p_array_, q_array_ and
       * f_arrays_ are size zero.
       */
      State() {};

      /**
       * Zero-state constructor, allocating vision states.
       *
       * Zero states. This constructor allocates the input size for the vision
       * state arrays p_array_, q_array_ and f_arrays_.
       *
       * @param[in] n_poses Number of poses in the sliding window (i.e. number
       *                    of positions in p_array_ / quaternions in q_array_).
       * @param[in] n_features Number of features in f_array_.
       */
      State(const size_t n_poses, const size_t n_features);

      /**
       * Full state constructor.
       *
       * This constructor sets all the state vector and covariance matrix
       * entries, i.e. all the class member variables. Refer to the description
       * of the member for a description of the inputs.
       */
      State(const double time,
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
            const Vector3& a_m);

      //////////////////////////// GETTERS & SETTERS ///////////////////////////
      
      double getTime() const;

      unsigned int getSeq() const;

      Vector3 getPosition() const;

      Quaternion getOrientation() const;

      Matrix getPositionArray() const;

      Matrix getOrientationArray() const;

      Matrix getFeatureArray() const;

      Quaternion getOrientationExtrinsics() const;

      Vector3 getPositionExtrinsics() const;

      Matrix getCovariance() const;

      /**
       * Assembles and returns 6x6 covariance of the pose error (dp,dq).
       *
       * @return The pose covariance matrix.
       */
      Matrix getPoseCovariance() const;

      Matrix& getCovarianceRef();

      void setTime(const double time);

      void setPositionArray(const Matrix& p_array);

      void setOrientationArray(const Matrix& q_array);

      void setFeatureArray(const Matrix& f_array);

      void setCovariance(const Matrix& cov);

      /**
       * Sets the IMU members of the state.
       *
       * This is used before propagating that state when a new IMU message came
       * in.
       *
       * @param[in] time IMU timestamp.
       * @param[in] seq IMU sequence ID.
       * @param[in] w_m Gyroscopes' angular rate measurements.
       * @param[in] a_m Accelerometers' specific force measurements.
       */
      void setImu(const double time,
                  const unsigned int seq,
                  const Vector3& w_m,
                  const Vector3& a_m);
      
      /**
       * Set static members from the input state.
       *
       * "Static" state members are those for which the estimate does not vary
       * at IMU rate, e.g. inertial biases, or vision states.
       */
      void setStaticStatesFrom(const State& state);
      
      //////////////////////////////////////////////////////////////////////////
      /**
       * Reset state.
       *
       * Flags the state as invalid. Its members will have to be set again for
       * use in the EKF.
       */
      void reset();

      /**
       * Compute the maximum number of poses allowed in the sliding window.
       */
      int nPosesMax() const;

      /**
       * Compute the maximum number of features allowed in the state.
       *
       */
      int nFeaturesMax() const;

      /**
       * Compute number of error states corresponding to the current state object.
       *
       * @return The number of errors states.
       */
      int nErrorStates() const;

      /**
       * Compute unbiased IMU measurements.
       *
       * Substracts biases to actual measurements.
       *
       * @param[out] e_w Unbiased gyros measurement.
       * @param[out] e_a Unbiased accel measurement.
       */
      void computeUnbiasedImuMeasurements(Vector3& e_w, Vector3& e_a) const;

      /**
       * Compute camera orientation from that of the IMU and extrinsics.
       *
       * @return A unit quaternion modeling the rotation changing the axes of
       *         {W} into {C}.
       */
      Attitude computeCameraAttitude() const;
       
      /**
       * Compute camera position from that of the IMU state and extrinsics.
       *
       * @return A 3-vector for the camera position in world frame.
       */
      Vector3 computeCameraPosition() const;

      /**
       * Compute camera orientation from that of the IMU and extrinsics.
       *
       * @return A unit quaternion modeling the rotation changing the axes of
       *         {W} into {C}.
       */
      Quaternion computeCameraOrientation() const;

      /**
       * Correct state based on input.
       *
       * @param[in] correction State correction vector.
       */
      void correct(const Eigen::VectorXd& correction);

      /**
       * Create a string that prints all state information.
       */
      std::string toString() const;
     
    private:
      /**
       * Timestamp
       *
       * Also acts as a flag that the state is valid and ready fo use in the
       * EKF.
       */
      double time_ { kInvalid };

      /**
       * Sequence ID.
       *
       * This sequence ID is used to identify the sensor measurement (e.g. IMU)
       * that corresponds to the current state. It should match the data's
       * sequence ID.
       */
      unsigned int seq_ { 0 };

      /**
       * Position of {I} wrt {W}, expressed in the axes of {W}.
       */
      Vector3 p_ { Vector3::Zero() };

      /**
       * Velocity of {I} wrt {W}, expressed in the axes of {W}.
       */
      Vector3 v_ { Vector3::Zero() };  

      /**
       * Unit quaternion modeling the rotation changing the axes of {W} into
       * {I}.
       */
      Quaternion q_ { Quaternion(1.0, 0.0, 0.0, 0.0) };

      /**
       * Gyroscope bias.
       */
      Vector3 b_w_ { Vector3::Zero() };

      /**
       * Accelerometer bias.
       */
      Vector3 b_a_ { Vector3::Zero() };
      
      /**
       * Array of positions of {C} wrt {W}, expressed in the axes of {W}.
       *
       * These static position states are used for MSCKF updates.
       */
      //Vector3Array p_array_;
      Matrix p_array_;

      /**
       * Array of quaternions modeling the rotation changing the axes of {W}
       * into {C}.
       *
       * These static orientation states are used for MSCKF update.
       */
      //QuaternionArray q_array_;
      Matrix q_array_;

      /**
       * Array of 3D feature states.
       *
       * These feature states are expressed in inverse-depth coordinates and
       * used for SLAM.
       */
      //Vector3Array f_array_;
      Matrix f_array_;

      /**
       * Error covariance matrix.
       */
      Matrix cov_;

      ////////////////////////// DEPRECATED STATES /////////////////////////////
      // TODO: Remove these states and modify measurement jacobians after xEKF
      // is implemented

      /**
       * Unit quaternion modeling the rotation changing the axes of {I} into
       * {C}.
       */
      Quaternion q_ic_ { Quaternion(1.0, 0.0, 0.0, 0.0) };

      /**
       * Position of {C} wrt {I}, expressed in the axes of {I}.
       */
      Vector3 p_ic_ { Vector3::Zero() };
      
      /////////////////////////// IMU MEASUREMENT /////////////////////////////

      /**
       * Angular rate measurement from gyroscopes [rad/s]
       */
      Vector3 w_m_ { Vector3::Zero() };

      /**
       * Specific force [m/s^2]
       */
      Vector3 a_m_ { Vector3::Zero() };

      //////////////////////////////////////////////////////////////////////////
     
      /**
       * Computes the error quaternion associated to a set of 3 small angles.
       *
       * @param[in] delta_theta 3-vector of small angles (x,y,z).
       * @return The error quaternion.
       */
      Quaternion errorQuatFromSmallAngles(const Vector3& delta_theta) const;
      
      friend class Propagator;
      friend class Ekf;
  };
} // namespace x

#endif  // X_EKF_STATE_H_
