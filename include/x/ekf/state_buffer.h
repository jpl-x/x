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

#ifndef X_EKF_STATE_BUFFER_H_
#define X_EKF_STATE_BUFFER_H_

#include <x/ekf/state.h>

namespace x
{
  constexpr int kInvalidIndex = -1;

  /**
   * State buffer class.
   *
   * This class is a simple ring buffer of States, implemented through a
   * std::vector. 
   */
  class StateBuffer : public std::vector<State>
  {
    public:
      /**
       * Constructs a buffer of a given size, allocating states for given
       * pose sliding window size and feature size.
       *
       * @param[in] size Buffer size (default: 250 states).
       * @param[in] default_state Default state instance to initialize buffer
       *                          elements at.
       * @param[in] time_margin Time margin, in seconds, around the buffer time
       *                        range.
       */
      StateBuffer(const size_t size = 250,
                  const State& default_state = State(),
                  const double time_margin = 0.005)
        : std::vector<State>(size, default_state)
        , time_margin_ { time_margin }
      {}

      /**
       * Get index the most recent IMU state in the buffer (tail).
       */
      int getTailIdx() const { return tail_idx_; };

      /**
       * Get a reference to most recent state in buffer (tail).
       *
       * This is the state at index tail__idx_.
       *
       * @return A reference to the tail state.
       */
      State& getTailStateRef();

      /**
       * Get the index of the closest state to input timestamp.
       *
       * Returns the index of the state in buffer which is the closest to the
       * input timestamp, or invalid index if the requested is outside the
       * buffer time range.
       *
       * @param[in] timestamp Requested time.
       * @return The index of the state closest to requested time.
       */
      int closestIdx(const double timestamp) const;
       
      /**
       * Get index coming after the input index in the ring buffer.
       *
       * @param[in] idx Input index.
       * @return The next index in buffer.
       */
      int nextIdx(const int idx) const;

      /**
       * Get index coming before the input index in the ring buffer.
       *
       * @param[in] idx Input index.
       * @return The previous index in buffer.
       */
      int previousIdx(const int idx) const;

      /*
       * Enqueue new state in the buffer, to be filled in-place by the caller.
       *
       * Returns a reference to the next state in buffer, which is set as new
       * tail and has to be filled the caller.
       *
       * @return A reference to the new tail state.
       */
      State& enqueueInPlace();
            
      /**
       * Reset buffer from input state.
       *
       * Resets all states in buffer, then copies input state at the start of
       * the buffer (new tail and head).
       *
       * @param[in] init_state State to reinitialize from.
       */
      void resetFromState(const State& init_state);

    private:
      /**
       * Index of the most recent IMU state in the buffer (tail).
       */
      int tail_idx_ { kInvalidIndex };

      /**
       * Index of the oldest IMU state in the buffer (head).
       */
      int head_idx_ { kInvalidIndex };

      /**
       * Number of valid states in buffer.
       */
      size_t n_valid_states_ { 0 };

      /**
       * Time margin, in seconds, around the buffer time range.
       *
       * This sets the tolerance for how far in the future/past the closest
       * state request can be without returning an invalid state.
       */
      double time_margin_ { 0.005 } ;

  };
} // namespace x

#endif  // X_EKF_STATE_BUFFER_H_
