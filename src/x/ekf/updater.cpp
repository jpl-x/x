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

#include <x/ekf/updater.h>

using namespace x;

void Updater::update(State& state) {
  // Image pre-processing
  preProcess(state);

  // Pre-update work and check if update is needed
  const bool update_requested = preUpdate(state);
  
  if (update_requested) {
    // Initialize correction vector 
    Matrix correction = Matrix::Zero(state.nErrorStates(), 1);

    // EKF Kalman update (iterated if iekf_iter_ > 1)
    for (int i=0; i < iekf_iter_; i++) {
      // Construct Jacobian, residual and noise covariance matrices
      Matrix h, res, r;
      constructUpdate(state, h, res, r);
      
      // Apply update
      const bool is_last_iter = i == iekf_iter_ - 1; // true if this is the last loop iteration
      applyUpdate(state, h, res, r, correction, is_last_iter);
    }

    // Post update: feature initialization
    postUpdate(state, correction);
  }
}

void Updater::applyUpdate(State& state,
                          const Matrix& H,
                          const Matrix& res,
                          const Matrix& R,
                          Matrix& correction_total,
                          const bool cov_update) {
  // Compute Kalman gain and state correction
  // TODO(jeff) Assert state correction doesn't have NaNs/Infs.
  Matrix& P = state.getCovarianceRef();
  const Matrix S = H * P * H.transpose() + R;
  const Matrix K = P * H.transpose() * S.inverse();
  Matrix correction = K * (res + H * correction_total) - correction_total;

  // Covariance update (skipped if this is not the last IEKF iteration)
  const size_t n = P.rows();
  if (cov_update) {
    P = (Matrix::Identity(n, n) - K * H) * P;
    // Make sure P stays symmetric.
    P = 0.5 * (P + P.transpose());
  }

  // State update
  state.correct(correction);
  
  // Add correction at current iteration to total (for IEKF)
  correction_total += correction;
}
