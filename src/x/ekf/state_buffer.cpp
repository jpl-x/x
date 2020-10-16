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

#include <x/ekf/state_buffer.h>
#include<iostream>
#include <iomanip>

using namespace x;

State& StateBuffer::getTailStateRef() {
  return at(tail_idx_);
}

int StateBuffer::closestIdx(const double timestamp) const {
  // Check if measurement is from the future
  if (timestamp > at(tail_idx_).getTime() + time_margin_) {
    std::cout << "Closest state request too far in the future. "
      "Check for time delay. "
      << "Time request: " << std::setprecision(17) << timestamp
      << ". Last time in buffer : " << at(tail_idx_).getTime() << std::endl;
    return kInvalidIndex;
  }

  // Check if measurement is from the past
  if (timestamp < at(head_idx_).getTime() - time_margin_) {
    std::cout << "Closest state request too far in the past. "
      "Increase buffer size." 
      << "Timestamp : " << std::setprecision(17) << timestamp
      << ". Oldest state in buffer : " << at(head_idx_).getTime() << std::endl;
    return kInvalidIndex;
  }

  // Browse state chronogically backwards until time difference stops reducing,
  // or we have been through the whole buffer.
  double time_offset = std::abs(timestamp - at(tail_idx_).getTime());
  int idx = previousIdx(tail_idx_);
  size_t count = 1;
  while (std::abs(timestamp - at(idx).getTime()) < time_offset
      && count < n_valid_states_) {
    time_offset = std::abs(timestamp - at(idx).getTime());
    idx = previousIdx(idx);
    ++count;
  }

  // Return closest state. Unless we have been through the whole buffer, that
  // was the previous-to-last index.
  return nextIdx(idx);
}

int StateBuffer::nextIdx(const int idx) const {
  return (idx + 1) % size();
}

int StateBuffer::previousIdx(const int idx) const {
  if (idx == 0)
    return size() - 1;
  else
    return idx - 1;
}

State& StateBuffer::enqueueInPlace() {
  // Increment tail
  ++tail_idx_ %= size();

  // Increment count and head
  if (n_valid_states_ < size())
    ++n_valid_states_;
  else
    ++head_idx_ %= size();

  // Return reference to the state to be filled by caller
  return at(tail_idx_);
}

void StateBuffer::resetFromState(const State& init_state) {
  // Reset each state in buffer
  for (size_t i = 0; i < size(); ++i)
    at(i).reset();

  // Set the first state of the buffer to init_state
  tail_idx_ = 0;
  head_idx_ = 0;
  n_valid_states_ = 1; 
  assert(size());
  at(tail_idx_) = init_state;
}
