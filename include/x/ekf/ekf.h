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

#ifndef X_EKF_EKF_H_
#define X_EKF_EKF_H_

#include <x/common/types.h>
#include <x/ekf/propagator.h>
#include <x/ekf/updater.h>
#include <boost/thread.hpp>

namespace x {
  /**
   * Filter initialization status.
   */
  enum InitStatus {
    /**
     * Filter is not initialized and doesn't process IMU messages.
     */
    kNotInitialized,
    /**
     * Filter received the init command and is standing by for the first IMU
     * measurement.
     */
    kStandBy,
    /**
     * Filter is initialized and processes IMU and update measurements.
     */
    kInitialized,
  };

  /**
   * A Extended Kalman filter class.
   *
   * Carries propagation and update of the states contained in a buffer.
   */
  class Ekf
  {
    public:
      /**
       * Minimum constructor.
       */
      Ekf(Updater& updater);

      /**
       * Copy constructor
       */
      Ekf(const Ekf& ekf);

      /**
       * Sets all EKF parameters before filter initialization.
       *
       * @param[in] updater Updater object, specific to measurement type.
       * @param[in] g Gravity vector, expressed in world frame.
       * @param[in] imu_noise IMU noise parameters (continuous time).
       * @param[in] state_buffer_sz Size to allocate for the state buffer.
       * @param[in] default_state Default state instance to initialize buffer
       *                          elements at.
       * @param[in] a_m_max Max specific force norm threshold.
       * @param[in] delta_seq_imu Expected difference between successive IMU
       *                          sequence IDs.
       * @param[in] time_margin_bfr Time margin, in seconds, around the buffer
       *                            time range.
       */
      void set(const Updater& updater,
               Vector3& g,
               const ImuNoise& imu_noise,
               const size_t state_buffer_sz,
               const State& default_state,
               const double a_m_max,
               const unsigned int delta_seq_imu,
               const double time_margin_bfr);

      /**
       * Get a reference to the updater.
       */
      Updater& getUpdaterRef();

      /**
       * (Re-)Initialize the filter from input state.
       *
       * @param[in] init_state State to initialize from. This state is required
       *                       have the same dynamic state size that have been
       *                       allocated by the set function in the buffer.
       */
      void initializeFromState(const State& init_state);

      /**
       * Process IMU measurements.
       *
       * Propagates state and covariance in the buffer.
       *
       * @param[in] timestamp
       * @param[in] seq IMU sequence ID.
       * @param[in] w_m Angular velocity (gyroscope).
       * @param[in] a_m Specific force (accelerometer).
       * @return The propagated state.
       */
      State processImu(const double timestamp,
                       const unsigned int seq,
                       const Vector3& w_m,
                       const Vector3& a_m);

      /**
       * Process update measurement.
       *
       * Retrieve the closest state prior to the measurement and call updater's
       * EKF update function.
       */
      State processUpdateMeasurement();
      
      /**
       * Locks mutex (thread safety).
       */
      void lock();

      /**
       * Unlock mutex (thread safety).
       */
      void unlock();

    private:
      /**
       * State and covariance propagator.
       */
      Propagator propagator_;

      /**
       * A reference to an EKF updater concrete class.
       *
       * Constructs and applies an EKF from a measurement. Updater is the
       * abstract class. The reference should point to a concrete updater
       * that is specific to a measurement type (e.g. VioUpdater)
       */
      Updater& updater_;
      
      /**
       * Time-sorted ring state buffer.
       *
       * Each state corresponds to an IMU message.
       */
      StateBuffer state_buffer_;
      
      /**
       * Initialization status.
       */
      InitStatus init_status_ { kNotInitialized };

      /**
       * Max specific force norm threshold.
       *
       * Used to detect accelerometer spikes. Default value 50 m/s^2.
       */
      double a_m_max_ { 50.0 };

      /**
       * Expected difference between successive IMU sequence IDs.
       *
       * Used to detects missing IMU messages. Default value 1.
       */
      unsigned int delta_seq_imu_ { 1 };
      
      /**
       * Thread safety.
       */
      boost::mutex mutex_;

      /**
       * Repropagate state buffer from input state.
       *
       * Repropagates state estimates and covariance from the input state copied
       * at input index, until last state in buffer. Returns false if the state
       * at that index in buffer doesn't have the same timestamp, i.e. if the
       * buffer was overwritten while the update was being computed (thread
       * safe).
       *
       * @param[in] state State to repropagate from.
       * @param[in] idx Index this state should copied at.
       * @return True if the repropagation was successful
       */
      bool repropagateFromStateAtIdx(const State& state, const int idx);
  };

  /**
   * Bad argument exception type when the size of the dynamic arrays (poses,
   * features, covariance matrix) in the initialization state doesn't match the
   * size allocated in the buffer.
   */
  class init_bfr_mismatch {};
} // namespace x

#endif  // X_EKF_EKF_H_
