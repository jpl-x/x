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

#include <x/ekf/ekf.h>

#include <iostream>

using namespace x;

Ekf::Ekf(Updater& updater)
  : updater_ { updater }
{}

Ekf::Ekf(const Ekf& ekf)
  : propagator_ { ekf.propagator_ }
  , state_buffer_ { ekf.state_buffer_ }
  , updater_ { ekf.updater_ }
{}

Updater& Ekf::getUpdaterRef() {
  return updater_;
}

void Ekf::set(const Updater& updater,
              Vector3& g,
              const ImuNoise& imu_noise,
              const size_t state_buffer_sz,
              const State& default_state,
              const double a_m_max,
              const unsigned int delta_seq_imu,
              const double time_margin_bfr) {
  updater_ = updater;
  state_buffer_ = StateBuffer(state_buffer_sz,
                              default_state,
                              time_margin_bfr);
  propagator_.set(g, imu_noise);
  a_m_max_ = a_m_max;
  delta_seq_imu_ = delta_seq_imu;
}

void Ekf::initializeFromState(const State& init_state) {
  // Check allocated buffer size is not zero
  if (state_buffer_.size() == 0)
    throw std::runtime_error("the EKF state buffer must have non-zero size.");

  // Check the size of the dynamic state arrays matches between the
  // initialization state and the state buffers.
  if (init_state.p_array_.rows() != state_buffer_[0].p_array_.rows()
   || init_state.p_array_.cols() != state_buffer_[0].p_array_.cols()
   || init_state.q_array_.rows() != state_buffer_[0].q_array_.rows()
   || init_state.q_array_.cols() != state_buffer_[0].q_array_.cols()
   || init_state.f_array_.rows() != state_buffer_[0].f_array_.rows()
   || init_state.f_array_.cols() != state_buffer_[0].f_array_.cols()
   || init_state.cov_.rows() != state_buffer_[0].cov_.rows()
   || init_state.cov_.cols() != state_buffer_[0].cov_.cols()) {
    throw init_bfr_mismatch {};
  }

  // Reset buffer from init_state
  state_buffer_.resetFromState(init_state);
  init_status_ = kStandBy;
}

State Ekf::processImu(const double timestamp,
                      const unsigned int seq,
                      const Vector3& w_m,
                      const Vector3& a_m) {
  // Seq ID for last IMU message
  static unsigned int last_seq = 0;

  // If the filter is not initialized, do nothing and return invalid state.
  if (init_status_ == kNotInitialized)
    return State();
  
  // Reference to last state in buffer
  lock();
  State& last_state = state_buffer_.getTailStateRef();

  // If this is the first IMU measurement, change the filter status, fill the
  // IMU data in the first state and return it.
  if (init_status_ == kStandBy) {
    if (a_m.norm() < a_m_max_) { // accel spike check
      last_state.setImu(timestamp, seq, w_m, a_m);
      unlock();
      last_seq = seq;
      init_status_ = kInitialized;
      std::cout << "Initial state:" << std::endl;
      std::cout << last_state.toString() << std::endl;
      return last_state;
    } else {
      std::cout << "Accelerometer spike detected at seq_id " << seq
        << ". Standing by for next IMU message to initialize." << std::endl;
      return State();
    }
  }

  // Check if timestamp is after the last state
  if (timestamp <= last_state.time_) {
    std::cout << "IMU going back in time. Discarding IMU seq_id " << seq << "."
      << std::endl;
    unlock();
    return State();
  }

  // Warn about missing IMU messages
  if (seq != last_seq + delta_seq_imu_) {
    std::cout << "Missing IMU messages. Expected seq_id "
      << last_seq + delta_seq_imu_ << ", but received seq_id " << seq << "."
      << std::endl;
  }
  last_seq = seq;

  // Remove accelerometer spikes
  Vector3 a_m_smoothed;
  if (a_m.norm() < a_m_max_)
    a_m_smoothed = a_m;
  else {
    a_m_smoothed = last_state.a_m_;
    std::cout << "Accelerometer spike detected at seq_id " << seq
      << ". Reading specific force from previous state in buffer." << std::endl;
  }
  
  // Set up propagated state
  State& next_state = state_buffer_.enqueueInPlace();
  next_state.setImu(timestamp, seq, w_m, a_m_smoothed);

  // Propagate state and covariance estimates
  propagator_.propagateState(last_state, next_state);
  propagator_.propagateCovariance(last_state, next_state);

  unlock();
  return next_state;
}

State Ekf::processUpdateMeasurement() {
  // If the filter is not initialized, do nothing and return invalid state.
  if (init_status_ == kNotInitialized)
    return State();

  // Get the index of the update state in buffer
  lock();
  const int update_state_idx = state_buffer_.closestIdx(updater_.getTime());
  unlock();

  // If no valid state could be found, do nothing and return invalid state.
  if (update_state_idx == kInvalidIndex)
    return State();

  // Get a copy of that state
  State update_state = state_buffer_[update_state_idx];

  // Update state
  updater_.update(update_state);

  // Repropagate buffer from update state
  bool update_success = false;
  lock();
  update_success = repropagateFromStateAtIdx(update_state, update_state_idx);
  unlock();

  // Check if update was applied (can fail if buffer was overwritten)
  if (update_success) 
    return update_state;
  else
    return State();
}

void Ekf::lock() {
#ifdef DUAL_THREAD
  mutex_.lock();
#endif
}

void Ekf::unlock() {
#ifdef DUAL_THREAD
  mutex_.unlock();
#endif
}

bool Ekf::repropagateFromStateAtIdx(const State& state, const int idx) {
  // Check the buffer slot wasn't overwritten and copy input
  if (state_buffer_[idx].getTime() == state.getTime())
    state_buffer_[idx] = state;
  else {
    std::cout << "Buffer slot at index " << idx << " was overwritten by IMU "
      "thread. Update measurement at time "
      << std::setprecision(17) << state.getTime()
      << " could not be applied." << std::endl;
    return false;
  }

  // Repropagation loop
  int curr_state_idx = idx;
  int next_state_idx = state_buffer_.nextIdx(idx);
  const int last_state_idx = state_buffer_.getTailIdx();
  while (curr_state_idx != last_state_idx) {
    State& state_curr = state_buffer_[curr_state_idx];
    State& state_next = state_buffer_[next_state_idx];
    propagator_.propagateState(state_curr, state_next);
    propagator_.propagateCovariance(state_curr, state_next);
    curr_state_idx = next_state_idx;
    next_state_idx = state_buffer_.nextIdx(next_state_idx);
  }

  return true;
}
