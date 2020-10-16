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

#include <x/ekf/propagator.h>

using namespace x;

Propagator::Propagator(const Vector3& g,
                       const ImuNoise& imu_noise)
  : g_ { g }
  , imu_noise_ { imu_noise }
{}

void Propagator::set(const Vector3& g,
                     const ImuNoise& imu_noise) {
  g_ = g;
  imu_noise_ = imu_noise;
}

void Propagator::propagateState(const State& state_0, State& state_1) {
  // Copy state estimates that don't vary at IMU rate
  state_1.setStaticStatesFrom(state_0);

  // Bias-free IMU measurements
  Vector3 e_w_1, e_a_1, e_w_0, e_a_0;
  state_1.computeUnbiasedImuMeasurements(e_w_1, e_a_1);
  state_0.computeUnbiasedImuMeasurements(e_w_0, e_a_0);

  // First-order quaternion integration
  const double dt = state_1.time_ - state_0.time_;
  const Matrix4 delta_q = quaternionIntegrator(e_w_0, e_w_1, dt);
  state_1.q_.coeffs() = delta_q * state_0.q_.coeffs(); // (x,y,z,w) order
  state_1.q_.normalize();

  // First-order velocity and position integration
  const Vector3 dv = ( state_1.q_.toRotationMatrix() * e_a_1
                     + state_0.q_.toRotationMatrix() * e_a_0 ) / 2.0;
  state_1.v_ = state_0.v_ + (dv + g_) * dt;
  state_1.p_ = state_0.p_ + (state_1.v_ + state_0.v_) / 2.0 * dt;
}

void Propagator::propagateCovariance(const State& state_0, State& state_1) {
  // Bias-free IMU measurements
  Vector3 e_w_1, e_a_1, e_w_0, e_a_0;
  state_1.computeUnbiasedImuMeasurements(e_w_1, e_a_1);
  state_0.computeUnbiasedImuMeasurements(e_w_0, e_a_0);

  // Compute coefficients of the discrete state transition matrix
  const double dt = state_1.time_ - state_0.time_;
  const CoreCovMatrix f_d
    = discreteStateTransition(dt,
                              e_w_1,
                              e_a_1,
                              state_1.q_);
  
  // Construct the discrete process noise covariance matrix
  const CoreCovMatrix q_d
    = discreteProcessNoiseCov(dt,
                              state_1.q_,
                              e_w_1,
                              e_a_1,
                              imu_noise_.n_w,
                              imu_noise_.n_bw,
                              imu_noise_.n_a,
                              imu_noise_.n_ba);

  // Covariance propagation using f_d and q_d
  propagateCovarianceMatrices(state_0.cov_, f_d, q_d,state_1.cov_);
}

Matrix4 Propagator::quaternionIntegrator(const Vector3& e_w_0,
                                         const Vector3& e_w_1,
                                         const double dt) const {
  // Angular rate matrices
  const Matrix4 omega_1 = e_w_1.toOmegaMatrix();
  const Matrix4 omega_0 = e_w_0.toOmegaMatrix();
  const Vector3 e_w_mean = (e_w_1 + e_w_0) / 2.0;
  const Matrix4 omega_mean = e_w_mean.toOmegaMatrix();

  // Matrix exponential Taylor series expansion at 4th order (eq. 130)
  int fac = 1;
  const Matrix4 a = omega_mean * 0.5 * dt;
  Matrix4 a_k = a;
  Matrix4 mat_exp = Matrix4::Identity();

  for (int k = 1; k < 5; k++) {
    fac = fac * k; // factorial(k)
    mat_exp = mat_exp + a_k / fac; // exponential series
    a_k = a_k * a; // increasing exponent k at each iteration
  }

  // Return quaternion integation matrix (eq. 131)
  return mat_exp
         + 1.0 / 48.0 * (omega_1 * omega_0 - omega_0 * omega_1) * dt * dt;
}

CoreCovMatrix
Propagator::discreteStateTransition(const double dt,
                                    const Vector3& e_w,
                                    const Vector3& e_a,
                                    const Quaternion& q) const {
  const Matrix3 w_x = e_w.toCrossMatrix();
  const Matrix3 a_x = e_a.toCrossMatrix();
  const Matrix3 eye3 = Eigen::Matrix<double, 3, 3>::Identity();

  const Matrix3 c_q = q.toRotationMatrix();

  const double dt_2_f2 = dt * dt * 0.5;
  const double dt_3_f3 = dt_2_f2 * dt / 3.0;
  const double dt_4_f4 = dt_3_f3 * dt * 0.25;
  const double dt_5_f5 = dt_4_f4 * dt * 0.2;

  const Matrix3 c_q_a_x = c_q * a_x;
  const Matrix3 a = c_q_a_x
    * (-dt_2_f2 * eye3 + dt_3_f3 * w_x - dt_4_f4 * w_x * w_x);
  const Matrix3 b = c_q_a_x
    * (dt_3_f3 * eye3 - dt_4_f4 * w_x + dt_5_f5 * w_x * w_x);
  const Matrix3 d = -a;
  const Matrix3 e = eye3 - dt * w_x + dt_2_f2 * w_x * w_x;
  const Matrix3 f = -dt * eye3 + dt_2_f2 * w_x - dt_3_f3 * (w_x * w_x);
  const Matrix3 c = c_q_a_x * f;

  // Construct the discrete state transition matrix
  CoreCovMatrix
    f_d = CoreCovMatrix::Identity();

  f_d.block<3,3>(kIdxP,kIdxV) = dt * eye3;
  f_d.block<3,3>(kIdxP,kIdxQ) = a;
  f_d.block<3,3>(kIdxP,kIdxBw) = b;
  f_d.block<3,3>(kIdxP,kIdxBa) = -c_q * dt_2_f2;

  f_d.block<3,3>(kIdxV,kIdxQ) = c;
  f_d.block<3,3>(kIdxV,kIdxBw) = d;
  f_d.block<3,3>(kIdxV,kIdxBa) = -c_q * dt;

  f_d.block<3,3>(kIdxQ,kIdxQ) = e;
  f_d.block<3,3>(kIdxQ,kIdxBw) = f;

  return f_d;
}

void Propagator::propagateCovarianceMatrices(
    const Eigen::MatrixXd& cov_0,
    const CoreCovMatrix& f_d,
    const CoreCovMatrix& q_d,
    Eigen::MatrixXd& cov_1) {
  // References to covariance blocks (no memory copy).
  // Naming based on Eq. 2.15 in xVIO tech report.
  const int n = cov_0.rows();
  assert(n > kSizeCoreErr - 1);
  const int n_v = n - kSizeCoreErr; // number of non-core states

  const Eigen::Ref<const CoreCovMatrix>
    cov_0_ii = cov_0.block<kSizeCoreErr,kSizeCoreErr>(0,0);
  const Eigen::Ref<const Eigen::MatrixXd>
    cov_0_iv = cov_0.topRightCorner(kSizeCoreErr,n_v);
  const Eigen::Ref<const Eigen::MatrixXd>
    cov_0_vi = cov_0.bottomLeftCorner(n_v,kSizeCoreErr);
  const Eigen::Ref<const Eigen::MatrixXd>
    cov_0_vv = cov_0.bottomRightCorner(n_v,n_v);

  Eigen::Ref<CoreCovMatrix>
    cov_1_ii = cov_1.block<kSizeCoreErr,kSizeCoreErr>(0,0);
  Eigen::Ref<Eigen::MatrixXd> cov_1_iv = cov_1.topRightCorner(kSizeCoreErr,n_v);
  Eigen::Ref<Eigen::MatrixXd> cov_1_vi = cov_1.bottomLeftCorner(n_v,kSizeCoreErr);
  Eigen::Ref<Eigen::MatrixXd> cov_1_vv = cov_1.bottomRightCorner(n_v,n_v);

  // Covariance block propagation
  // (eq. 2.16-18 in xVIO tech report and eq. 22 in Weiss et al.)
  cov_1_ii = f_d * cov_0_ii * f_d.transpose() + q_d;
  cov_1_iv = f_d * cov_0_iv;
  // Note: doing  cov_1_vi = cov_1_iv.transpose() behaves differently.
  // Specifically, it causes divergence if the IMU runs for too long (e.g.
  // 0.5s) before the first update. It should be equivalent though. There are
  // non-symmetric numerical differences in the q_d matrix. Could this be the
  // cause? TODO: check if the 2 covariance formulations give the same results
  // when q_d is exactly symmetric.
  cov_1_vi = cov_0_vi * f_d.transpose();
  cov_1_vv = cov_0_vv;
}

CoreCovMatrix
Propagator::discreteProcessNoiseCov(const double dt,
                                    const Quaternion& q,
                                    const Vector3& e_w,
                                    const Vector3& e_a,
                                    const double n_w,
                                    const double n_bw,
                                    const double n_a,
                                    const double n_ba) const {

  const double q1 = q.w(), q2 = q.x(), q3 = q.y(), q4 = q.z();
  const double e_w1 = e_w(0), e_w2 = e_w(1), e_w3 = e_w(2);
  const double e_a1 = e_a(0), e_a2 = e_a(1), e_a3 = e_a(2);

  const double t343 = dt * dt;
  const double t348 = q1 * q4 * 2.0;
  const double t349 = q2 * q3 * 2.0;
  const double t344 = t348 - t349;
  const double t356 = q1 * q3 * 2.0;
  const double t357 = q2 * q4 * 2.0;
  const double t345 = t356 + t357;
  const double t350 = q1 * q1;
  const double t351 = q2 * q2;
  const double t352 = q3 * q3;
  const double t353 = q4 * q4;
  const double t346 = t350 + t351 - t352 - t353;
  const double t347 = n_a * n_a;
  const double t354 = n_a * n_a;
  const double t355 = n_a * n_a;
  const double t358 = q1 * q2 * 2.0;
  const double t359 = t344 * t344;
  const double t360 = t345 * t345;
  const double t361 = t346 * t346;
  const double t363 = e_a2 * t345;
  const double t364 = e_a3 * t344;
  const double t362 = t363 + t364;
  const double t365 = t362 * t362;
  const double t366 = t348 + t349;
  const double t367 = t350 - t351 + t352 - t353;
  const double t368 = q3 * q4 * 2.0;
  const double t369 = t356 - t357;
  const double t370 = t350 - t351 - t352 + t353;
  const double t371 = n_w * n_w;
  const double t372 = t358 + t368;
  const double t373 = n_w * n_w;
  const double t374 = n_w * n_w;
  const double t375 = dt * t343 * t346 * t347 * t366 * (1.0 / 3.0);
  const double t376 = t358 - t368;
  const double t377 = t343 * t346 * t347 * t366 * (1.0 / 2.0);
  const double t378 = t366 * t366;
  const double t379 = t376 * t376;
  const double t380 = e_a1 * t367;
  const double t391 = e_a2 * t366;
  const double t381 = t380 - t391;
  const double t382 = e_a3 * t367;
  const double t383 = e_a2 * t376;
  const double t384 = t382 + t383;
  const double t385 = t367 * t367;
  const double t386 = e_a1 * t376;
  const double t387 = e_a3 * t366;
  const double t388 = t386 + t387;
  const double t389 = e_a2 * t370;
  const double t407 = e_a3 * t372;
  const double t390 = t389 - t407;
  const double t392 = e_a1 * t372;
  const double t393 = e_a2 * t369;
  const double t394 = t392 + t393;
  const double t395 = e_a1 * t370;
  const double t396 = e_a3 * t369;
  const double t397 = t395 + t396;
  const double t398 = n_ba * n_ba;
  const double t399 = n_ba * n_ba;
  const double t400 = n_ba * n_ba;
  const double t401 = dt * t343 * t345 * t355 * t370 * (1.0 / 3.0);
  const double t402 = t401 - dt * t343 * t346 * t347 * t369 * (1.0 / 3.0)
    - dt * t343 * t344 * t354 * t372 * (1.0 / 3.0);
  const double t403 = dt * t343 * t354 * t367 * t372 * (1.0 / 3.0);
  const double t404 = t403 - dt * t343 * t347 * t366 * t369 * (1.0 / 3.0)
    - dt * t343 * t355 * t370 * t376 * (1.0 / 3.0);
  const double t405 = t343 * t345 * t355 * t370 * (1.0 / 2.0);
  const double t406 = dt * t343 * t362 * t373 * t397 * (1.0 / 6.0);
  const double t421 = t343 * t346 * t347 * t369 * (1.0 / 2.0);
  const double t422 = dt * t343 * t362 * t371 * t394 * (1.0 / 6.0);
  const double t423 = t343 * t344 * t354 * t372 * (1.0 / 2.0);
  const double t424 = dt * t343 * t362 * t374 * t390 * (1.0 / 6.0);
  const double t408 = t405 + t406 - t421 - t422 - t423 - t424;
  const double t409 = t343 * t354 * t367 * t372 * (1.0 / 2.0);
  const double t410 = dt * t343 * t374 * t384 * t390 * (1.0 / 6.0);
  const double t411 = dt * t343 * t373 * t388 * t397 * (1.0 / 6.0);
  const double t463 = t343 * t355 * t370 * t376 * (1.0 / 2.0);
  const double t464 = t343 * t347 * t366 * t369 * (1.0 / 2.0);
  const double t465 = dt * t343 * t371 * t381 * t394 * (1.0 / 6.0);
  const double t412 = t409 + t410 + t411 - t463 - t464 - t465;
  const double t413 = t369 * t369;
  const double t414 = t372 * t372;
  const double t415 = t370 * t370;
  const double t416 = t343 * t354 * t359 * (1.0 / 2.0);
  const double t417 = t343 * t355 * t360 * (1.0 / 2.0);
  const double t418 = t343 * t347 * t361 * (1.0 / 2.0);
  const double t419 = t416 + t417 + t418 - dt * t343 * t365 * t371 * (1.0 / 6.0)
    - dt * t343 * t365 * t373 * (1.0 / 6.0)
    - dt * t343 * t365 * t374 * (1.0 / 6.0);
  const double t453 = t343 * t344 * t354 * t367 * (1.0 / 2.0);
  const double t454 = t343 * t345 * t355 * t376 * (1.0 / 2.0);
  const double t420 = t377 - t453 - t454;
  const double t426 = e_w2 * t362;
  const double t427 = e_w3 * t362;
  const double t425 = t426 - t427;
  const double t428 = dt * t365;
  const double t429 = e_w1 * e_w1;
  const double t430 = e_w2 * e_w2;
  const double t431 = e_w3 * e_w3;
  const double t432 = t430 + t431;
  const double t433 = t362 * t432;
  const double t434 = e_w1 * t343 * t365;
  const double t435 = t429 + t431;
  const double t436 = t362 * t435;
  const double t443 = e_w2 * e_w3 * t362;
  const double t437 = t433 + t436 - t443;
  const double t438 = e_w1 * t362 * t394;
  const double t511 = e_w1 * t362 * t397;
  const double t439 = t438 - t511;
  const double t440 = t343 * t439 * (1.0 / 2.0);
  const double t441 = t429 + t430;
  const double t442 = t362 * t441;
  const double t444 = t390 * t432;
  const double t445 = e_w2 * t394;
  const double t446 = e_w3 * t397;
  const double t447 = t445 + t446;
  const double t448 = e_w1 * e_w2 * t362;
  const double t449 = e_w1 * e_w3 * t362;
  const double t450 = e_w1 * e_w3 * t362 * (1.0 / 2.0);
  const double t451 = dt * t362;
  const double t452 = e_w1 * t343 * t362 * (1.0 / 2.0);
  const double t455 = dt * t343 * t362 * t374 * t384 * (1.0 / 6.0);
  const double t456 = t343 * t347 * t378 * (1.0 / 2.0);
  const double t457 = t343 * t355 * t379 * (1.0 / 2.0);
  const double t458 = t381 * t381;
  const double t459 = t384 * t384;
  const double t460 = t343 * t354 * t385 * (1.0 / 2.0);
  const double t461 = t388 * t388;
  const double t462 = t456 + t457 + t460 - dt * t343 * t371 * t458 * (1.0 / 6.0)
    - dt * t343 * t374 * t459 * (1.0 / 6.0)
    - dt * t343 * t373 * t461 * (1.0 / 6.0);
  const double t466 = t433 + t442 - t443;
  const double t467 = e_w1 * t362 * t388;
  const double t468 = e_w1 * t362 * t381;
  const double t469 = t467 + t468;
  const double t470 = t343 * t469 * (1.0 / 2.0);
  const double t471 = t384 * t432;
  const double t472 = e_w2 * t381;
  const double t479 = e_w3 * t388;
  const double t473 = t472 - t479;
  const double t474 = -t433 + t448 + t449;
  const double t475 = dt * t343 * t346 * t366 * t398 * (1.0 / 3.0);
  const double t476 = dt * t346 * t347 * t366;
  const double t477 = e_w2 * e_w3 * t381;
  const double t492 = t388 * t435;
  const double t478 = t471 + t477 - t492;
  const double t480 = t472 - t479;
  const double t481 = e_w1 * e_w3 * t381;
  const double t482 = e_w1 * e_w2 * t388;
  const double t483 = t471 + t481 + t482;
  const double t484 = e_w2 * e_w3 * t388;
  const double t486 = t381 * t441;
  const double t485 = t471 + t484 - t486;
  const double t487 = t394 * t441;
  const double t488 = e_w2 * e_w3 * t397;
  const double t489 = t444 + t487 + t488;
  const double t490 = t397 * t435;
  const double t491 = e_w2 * e_w3 * t394;
  const double t493 = e_w1 * t381 * t397;
  const double t541 = e_w1 * t388 * t394;
  const double t494 = t493 - t541;
  const double t495 = t343 * t494 * (1.0 / 2.0);
  const double t496 = e_w1 * e_w2 * t397;
  const double t527 = e_w1 * e_w3 * t394;
  const double t497 = t444 + t496 - t527;
  const double t498 = e_w2 * e_w3 * t381 * (1.0 / 2.0);
  const double t499 = e_w1 * t343 * t381 * (1.0 / 2.0);
  const double t500 = t384 * t432 * (1.0 / 2.0);
  const double t501 = e_w2 * e_w3 * t388 * (1.0 / 2.0);
  const double t502 = n_bw * n_bw;
  const double t503 = n_bw * n_bw;
  const double t504 = t343 * t347 * t413 * (1.0 / 2.0);
  const double t505 = t343 * t354 * t414 * (1.0 / 2.0);
  const double t506 = t397 * t397;
  const double t507 = t390 * t390;
  const double t508 = t343 * t355 * t415 * (1.0 / 2.0);
  const double t509 = t394 * t394;
  const double t510 = t504 + t505 + t508 - dt * t343 * t373 * t506 * (1.0 / 6.0)
    - dt * t343 * t371 * t509 * (1.0 / 6.0)
    - dt * t343 * t374 * t507 * (1.0 / 6.0);
  const double t512 = -t444 + t490 + t491;
  const double t513 = t397 * t437 * (1.0 / 2.0);
  const double t514 = t362 * t394 * t429;
  const double t515 = dt * t362 * t397;
  const double t516 = t362 * t489 * (1.0 / 2.0);
  const double t517 = t394 * t466 * (1.0 / 2.0);
  const double t518 = t362 * t397 * t429;
  const double t519 = t516 + t517 + t518;
  const double t520 = dt * t362 * t394;
  const double t521 = t440 + t520 - dt * t343 * t519 * (1.0 / 3.0);
  const double t522 = t371 * t521;
  const double t523 = t362 * t447;
  const double t524 = t390 * t425;
  const double t525 = t523 + t524;
  const double t526 = t343 * t525 * (1.0 / 2.0);
  const double t528 = t425 * t447;
  const double t529 = t390 * t474 * (1.0 / 2.0);
  const double t530 = t528 + t529 - t362 * t497 * (1.0 / 2.0);
  const double t531 = dt * t343 * t530 * (1.0 / 3.0);
  const double t532 = dt * t362 * t390;
  const double t533 = t526 + t531 + t532;
  const double t534 = t374 * t533;
  const double t535 = dt * t343 * t345 * t370 * t400 * (1.0 / 3.0);
  const double t536 = dt * t345 * t355 * t370;
  const double t537 = t381 * t489 * (1.0 / 2.0);
  const double t538 = t388 * t397 * t429;
  const double t539 = t537 + t538 - t394 * t485 * (1.0 / 2.0);
  const double t540 = dt * t343 * t539 * (1.0 / 3.0);
  const double t542 = t495 + t540 - dt * t381 * t394;
  const double t543 = t388 * t512 * (1.0 / 2.0);
  const double t544 = t381 * t394 * t429;
  const double t545 = t543 + t544 - t397 * t478 * (1.0 / 2.0);
  const double t546 = dt * t343 * t545 * (1.0 / 3.0);
  const double t547 = t495 + t546 - dt * t388 * t397;
  const double t548 = t373 * t547;
  const double t549 = t384 * t447;
  const double t550 = t549 - t390 * t473;
  const double t551 = t343 * t550 * (1.0 / 2.0);
  const double t552 = t384 * t497 * (1.0 / 2.0);
  const double t553 = t390 * t483 * (1.0 / 2.0);
  const double t554 = t447 * t473;
  const double t555 = t552 + t553 + t554;
  const double t556 = dt * t384 * t390;
  const double t557 = t551 + t556 - dt * t343 * t555 * (1.0 / 3.0);
  const double t558 = dt * t343 * t367 * t372 * t399 * (1.0 / 3.0);
  const double t559 = dt * t354 * t367 * t372;
  const double t560 = t548 + t558 + t559 - t371 * t542 - t374 * t557
    - dt * t347 * t366 * t369 - dt * t355 * t370 * t376
    - dt * t343 * t366 * t369 * t398 * (1.0 / 3.0)
    - dt * t343 * t370 * t376 * t400 * (1.0 / 3.0);
  const double t561 = e_w1 * t343 * t394 * t397;
  const double t562 = e_w1 * t343 * t397 * (1.0 / 2.0);
  const double t563 = n_bw * n_bw;
  const double t564 = dt * t343 * t362 * t374 * (1.0 / 6.0);
  const double t565 = dt * t343 * t374 * t390 * (1.0 / 6.0);
  const double t566 = e_w1 * e_w2 * t362 * (1.0 / 2.0);
  const double t567 = -t433 + t450 + t566;
  const double t568 = dt * t343 * t567 * (1.0 / 3.0);
  const double t569 = t343 * t425 * (1.0 / 2.0);
  const double t570 = t451 + t568 + t569;
  const double t571 = dt * t343 * t362 * t373 * t432 * (1.0 / 6.0);
  const double t572 = dt * t343 * t362 * t371 * t432 * (1.0 / 6.0);
  const double t573 = t571 + t572 - t374 * t570;
  const double t574 = e_w1 * e_w2 * t397 * (1.0 / 2.0);
  const double t575 = t444 + t574 - e_w1 * e_w3 * t394 * (1.0 / 2.0);
  const double t576 = t343 * t447 * (1.0 / 2.0);
  const double t577 = dt * t390;
  const double t578 = t576 + t577 - dt * t343 * t575 * (1.0 / 3.0);
  const double t579 = dt * t343 * t371 * t394 * t432 * (1.0 / 6.0);
  const double t580 = t579 - t374 * t578
    - dt * t343 * t373 * t397 * t432 * (1.0 / 6.0);
  const double t581 = dt * t343 * t373 * t388 * (1.0 / 6.0);
  const double t582 = t362 * t432 * (1.0 / 2.0);
  const double t583 = e_w2 * e_w3 * t362 * (1.0 / 2.0);
  const double t584 = t362 * t429;
  const double t585 = t583 + t584;
  const double t586 = e_w3 * t473;
  const double t587 = e_w1 * e_w2 * t384 * (1.0 / 2.0);
  const double t588 = t586 + t587;
  const double t589 = dt * t343 * t588 * (1.0 / 3.0);
  const double t590 = t374 * (t589 - e_w3 * t343 * t384 * (1.0 / 2.0));
  const double t591 = t388 * t429;
  const double t592 = t498 + t591;
  const double t593 = dt * t343 * t592 * (1.0 / 3.0);
  const double t594 = t499 + t593;
  const double t595 = -t492 + t498 + t500;
  const double t596 = dt * t343 * t595 * (1.0 / 3.0);
  const double t597 = dt * t388;
  const double t598 = -t499 + t596 + t597;
  const double t599 = t590 - t371 * t594 - t373 * t598;
  const double t600 = t397 * t429;
  const double t601 = e_w2 * e_w3 * t394 * (1.0 / 2.0);
  const double t602 = e_w1 * t343 * t394 * (1.0 / 2.0);
  const double t603 = e_w3 * t447;
  const double t604 = t603 - e_w1 * e_w2 * t390 * (1.0 / 2.0);
  const double t605 = dt * t343 * t604 * (1.0 / 3.0);
  const double t606 = e_w3 * t343 * t390 * (1.0 / 2.0);
  const double t607 = t605 + t606;
  const double t608 = t374 * t607;
  const double t609 = t390 * t432 * (1.0 / 2.0);
  const double t610 = dt * t397;
  const double t611 = t430 * (1.0 / 2.0);
  const double t612 = t431 * (1.0 / 2.0);
  const double t613 = t611 + t612;
  const double t614 = e_w1 * t343 * (1.0 / 2.0);
  const double t615 = dt * t343 * t362 * t371 * (1.0 / 6.0);
  const double t616 = dt * t343 * t371 * t381 * (1.0 / 6.0);
  const double t617 = dt * t343 * t371 * t394 * (1.0 / 6.0);
  const double t618 = e_w2 * t425;
  const double t619 = t450 + t618;
  const double t620 = dt * t343 * t619 * (1.0 / 3.0);
  const double t621 = e_w2 * t343 * t362 * (1.0 / 2.0);
  const double t622 = t620 + t621;
  const double t623 = dt * t343 * t585 * (1.0 / 3.0);
  const double t624 = t381 * t429;
  const double t625 = t501 + t624;
  const double t626 = dt * t343 * t625 * (1.0 / 3.0);
  const double t627 = e_w1 * t343 * t388 * (1.0 / 2.0);
  const double t628 = e_w2 * t473;
  const double t629 = t628 - e_w1 * e_w3 * t384 * (1.0 / 2.0);
  const double t630 = dt * t343 * t629 * (1.0 / 3.0);
  const double t631 = t630 - e_w2 * t343 * t384 * (1.0 / 2.0);
  const double t632 = -t486 + t500 + t501;
  const double t633 = dt * t343 * t632 * (1.0 / 3.0);
  const double t634 = dt * t381;
  const double t635 = t627 + t633 + t634;
  const double t636 = e_w2 * t447;
  const double t637 = e_w1 * e_w3 * t390 * (1.0 / 2.0);
  const double t638 = t636 + t637;
  const double t639 = dt * t343 * t638 * (1.0 / 3.0);
  const double t640 = e_w2 * t343 * t390 * (1.0 / 2.0);
  const double t641 = t639 + t640;
  const double t642 = t394 * t429;
  const double t643 = e_w2 * e_w3 * t397 * (1.0 / 2.0);
  const double t644 = t487 + t609 + t643;
  const double t645 = dt * t343 * t644 * (1.0 / 3.0);
  const double t646 = t562 + t645 - dt * t394;
  const double t647 = t371 * t646;
  const double t648 = e_w2 * t343 * (1.0 / 2.0);
  const double t649 = dt * e_w1 * e_w3 * t343 * (1.0 / 6.0);
  const double t650 = t648 + t649;
  const double t651 = t374 * t650;
  const double t652 = t651 - dt * t343 * t371 * t613 * (1.0 / 3.0);
  const double t653 = dt * e_w2 * e_w3 * t343 * (1.0 / 6.0);
  const double t654 = t614 + t653;
  const double t655 = t371 * t654;
  const double t656 = dt * t343 * t397 * t563 * (1.0 / 6.0);
  const double t657 = dt * e_w1 * t343 * t563 * (1.0 / 6.0);
  const double t658 = dt * t343 * t369 * t398 * (1.0 / 6.0);
  const double t659 = t343 * t369 * t398 * (1.0 / 2.0);
  const double t660 = dt * t343 * t344 * t399 * (1.0 / 6.0);
  const double t661 = t343 * t344 * t399 * (1.0 / 2.0);
  const double t662 = dt * t343 * t376 * t400 * (1.0 / 6.0);
  const double t663 = t343 * t376 * t400 * (1.0 / 2.0);

  CoreCovMatrix q_d = CoreCovMatrix::Zero();
  q_d(kIdxP + 0, kIdxP + 0) = dt * t343 * t347 * t361
    * (1.0 / 3.0) + dt * t343 * t354 * t359 * (1.0 / 3.0)
    + dt * t343 * t355 * t360 * (1.0 / 3.0);
  q_d(kIdxP + 0, kIdxP + 1) = t375
    - dt * t343 * t345 * t355 * (t358 - q3 * q4 * 2.0) * (1.0 / 3.0)
    - dt * t343 * t344 * t354 * t367 * (1.0 / 3.0);
  q_d(kIdxP + 0, kIdxP + 2) = t402;
  q_d(kIdxP + 0, kIdxV + 0) = t419;
  q_d(kIdxP + 0, kIdxV + 1) = t420;
  q_d(kIdxP + 0, kIdxV + 2) = t408;
  q_d(kIdxP + 0, kIdxQ + 0) = t564;
  q_d(kIdxP + 0, kIdxQ + 2) = t615;
  q_d(kIdxP + 0, kIdxBa + 0) = dt * t343 * t346 * t398
    * (-1.0 / 6.0);
  q_d(kIdxP + 0, kIdxBa + 1) = t660;
  q_d(kIdxP + 0, kIdxBa + 2) = dt * t343 * t345 * t400
    * (-1.0 / 6.0);
  q_d(kIdxP + 1, kIdxP + 0) = t375
    - dt * t343 * t344 * t354 * t367 * (1.0 / 3.0)
    - dt * t343 * t345 * t355 * t376 * (1.0 / 3.0);
  q_d(kIdxP + 1, kIdxP + 1) = dt * t343 * t347 * t378
    * (1.0 / 3.0) + dt * t343 * t355 * t379 * (1.0 / 3.0)
    + dt * t343 * t354 * t385 * (1.0 / 3.0);
  q_d(kIdxP + 1, kIdxP + 2) = t404;
  q_d(kIdxP + 1, kIdxV + 0) = t377 + t455
    - t343 * t344 * t354 * t367 * (1.0 / 2.0)
    - t343 * t345 * t355 * t376 * (1.0 / 2.0)
    - dt * t343 * t362 * t371 * t381 * (1.0 / 6.0)
    - dt * t343 * t362 * t373 * t388 * (1.0 / 6.0);
  q_d(kIdxP + 1, kIdxV + 1) = t462;
  q_d(kIdxP + 1, kIdxV + 2) = t412;
  q_d(kIdxP + 1, kIdxQ + 0) = dt * t343 * t374 * t384
    * (-1.0 / 6.0);
  q_d(kIdxP + 1, kIdxQ + 1) = t581;
  q_d(kIdxP + 1, kIdxQ + 2) = t616;
  q_d(kIdxP + 1, kIdxBa + 0) = dt * t343 * t366 * t398
    * (-1.0 / 6.0);
  q_d(kIdxP + 1, kIdxBa + 1) = dt * t343 * t367 * t399
    * (-1.0 / 6.0);
  q_d(kIdxP + 1, kIdxBa + 2) = t662;
  q_d(kIdxP + 2, kIdxP + 0) = t402;
  q_d(kIdxP + 2, kIdxP + 1) = t404;
  q_d(kIdxP + 2, kIdxP + 2) = dt * t343 * t347 * t413
    * (1.0 / 3.0) + dt * t343 * t354 * t414 * (1.0 / 3.0)
    + dt * t343 * t355 * t415 * (1.0 / 3.0);
  q_d(kIdxP + 2, kIdxV + 0) = t408;
  q_d(kIdxP + 2, kIdxV + 1) = t412;
  q_d(kIdxP + 2, kIdxV + 2) = t510;
  q_d(kIdxP + 2, kIdxQ + 0) = t565;
  q_d(kIdxP + 2, kIdxQ + 1) = dt * t343 * t373 * t397
    * (-1.0 / 6.0);
  q_d(kIdxP + 2, kIdxQ + 2) = t617;
  q_d(kIdxP + 2, kIdxBa + 0) = t658;
  q_d(kIdxP + 2, kIdxBa + 1) = dt * t343 * t372 * t399
    * (-1.0 / 6.0);
  q_d(kIdxP + 2, kIdxBa + 2) = dt * t343 * t370 * t400
    * (-1.0 / 6.0);
  q_d(kIdxV + 0, kIdxP + 0) = t419;
  q_d(kIdxV + 0, kIdxP + 1) = t420;
  q_d(kIdxV + 0, kIdxP + 2) = t408;
  q_d(kIdxV + 0, kIdxV + 0) =
    t374
    * (t428 + t343 * t362 * t425
        + dt * t343 * (t362 * (t448 + t449 - t362 * t432) + t425 * t425)
        * (1.0 / 3.0))
    + t373
    * (t428 - t434
        + dt * t343 * (t365 * t429 - t362 * t437) * (1.0 / 3.0))
    + t371
    * (t428 + t434
        + dt * t343
        * (t365 * t429 - t362 * (t433 + t442 - e_w2 * e_w3 * t362))
        * (1.0 / 3.0)) + dt * t347 * t361 + dt * t354 * t359
    + dt * t355 * t360 + dt * t343 * t359 * t399 * (1.0 / 3.0)
    + dt * t343 * t361 * t398 * (1.0 / 3.0)
    + dt * t343 * t360 * t400 * (1.0 / 3.0);
  q_d(kIdxV + 0, kIdxV + 1) = t475 + t476
    - dt * t344 * t354 * t367 - dt * t345 * t355 * t376
    - dt * t343 * t344 * t367 * t399 * (1.0 / 3.0)
    - dt * t343 * t345 * t376 * t400 * (1.0 / 3.0);
  q_d(kIdxV + 0, kIdxV + 2) = t522 + t534 + t535 + t536
    - t373
    * (t440 + t515
        - dt * t343
        * (t513 + t514
          + t362 * (t490 + t491 - t390 * t432) * (1.0 / 2.0))
        * (1.0 / 3.0)) - dt * t346 * t347 * t369
    - dt * t344 * t354 * t372 - dt * t343 * t346 * t369 * t398 * (1.0 / 3.0)
    - dt * t343 * t344 * t372 * t399 * (1.0 / 3.0);
  q_d(kIdxV + 0, kIdxQ + 0) = t573;
  q_d(kIdxV + 0, kIdxQ + 2) = -t371
    * (t451 + t452
        - dt * t343 * (t442 + t582 - e_w2 * e_w3 * t362 * (1.0 / 2.0))
        * (1.0 / 3.0)) - t374 * t622
    + t373 * (t452 - dt * t343 * t585 * (1.0 / 3.0));
  q_d(kIdxV + 0, kIdxBw + 0) = dt * t343 * t362 * t502
    * (-1.0 / 6.0);
  q_d(kIdxV + 0, kIdxBw + 2) = dt * t343 * t362 * t503
    * (-1.0 / 6.0);
  q_d(kIdxV + 0, kIdxBa + 0) = t343 * t346 * t398
    * (-1.0 / 2.0);
  q_d(kIdxV + 0, kIdxBa + 1) = t661;
  q_d(kIdxV + 0, kIdxBa + 2) = t343 * t345 * t400
    * (-1.0 / 2.0);
  q_d(kIdxV + 1, kIdxP + 0) = t377 - t453 - t454 + t455
    - dt * t343 * t362 * t371 * t381 * (1.0 / 6.0)
    - dt * t343 * t362 * t373 * t388 * (1.0 / 6.0);
  q_d(kIdxV + 1, kIdxP + 1) = t462;
  q_d(kIdxV + 1, kIdxP + 2) = t412;
  q_d(kIdxV + 1, kIdxV + 0) = t475 + t476
    - t374
    * (t343 * (t384 * t425 - t362 * t473) * (1.0 / 2.0)
        - dt * t343
        * (t362 * t483 * (1.0 / 2.0) - t384 * t474 * (1.0 / 2.0)
          + t425 * t473) * (1.0 / 3.0) + dt * t362 * t384)
    + t371
    * (t470 + dt * t362 * t381
        + dt * t343
        * (t362 * t485 * (1.0 / 2.0) - t381 * t466 * (1.0 / 2.0)
          + t362 * t388 * t429) * (1.0 / 3.0))
    + t373
    * (-t470 + dt * t362 * t388
        + dt * t343
        * (t388 * t437 * (-1.0 / 2.0) + t362 * t478 * (1.0 / 2.0)
          + t362 * t381 * t429) * (1.0 / 3.0))
    - dt * t344 * t354 * t367 - dt * t345 * t355 * t376
    - dt * t343 * t344 * t367 * t399 * (1.0 / 3.0)
    - dt * t343 * t345 * t376 * t400 * (1.0 / 3.0);
  q_d(kIdxV + 1, kIdxV + 1) = -t374
    * (-dt * t459 + dt * t343 * (t384 * t483 - t480 * t480) * (1.0 / 3.0)
        + t343 * t384 * t473)
    + t373
    * (dt * t461 + dt * t343 * (t388 * t478 + t429 * t458) * (1.0 / 3.0)
        - e_w1 * t343 * t381 * t388)
    + t371
    * (dt * t458 + dt * t343 * (t381 * t485 + t429 * t461) * (1.0 / 3.0)
        + e_w1 * t343 * t381 * t388) + dt * t347 * t378 + dt * t355 * t379
    + dt * t354 * t385 + dt * t343 * t378 * t398 * (1.0 / 3.0)
    + dt * t343 * t385 * t399 * (1.0 / 3.0)
    + dt * t343 * t379 * t400 * (1.0 / 3.0);
  q_d(kIdxV + 1, kIdxV + 2) = t560;
  q_d(kIdxV + 1, kIdxQ + 0) = -t374
    * (-dt * t384 + t343 * t473 * (1.0 / 2.0)
        + dt * t343
        * (t471 + e_w1 * e_w2 * t388 * (1.0 / 2.0)
          + e_w1 * e_w3 * t381 * (1.0 / 2.0)) * (1.0 / 3.0))
    + dt * t343 * t371 * t381 * t432 * (1.0 / 6.0)
    + dt * t343 * t373 * t388 * t432 * (1.0 / 6.0);
  q_d(kIdxV + 1, kIdxQ + 1) = t599;
  q_d(kIdxV + 1, kIdxQ + 2) = -t374 * t631 - t371 * t635
    - t373 * (t626 - e_w1 * t343 * t388 * (1.0 / 2.0));
  q_d(kIdxV + 1, kIdxBw + 0) = dt * t343 * t384 * t502
    * (1.0 / 6.0);
  q_d(kIdxV + 1, kIdxBw + 1) = dt * t343 * t388 * t563
    * (-1.0 / 6.0);
  q_d(kIdxV + 1, kIdxBw + 2) = dt * t343 * t381 * t503
    * (-1.0 / 6.0);
  q_d(kIdxV + 1, kIdxBa + 0) = t343 * t366 * t398
    * (-1.0 / 2.0);
  q_d(kIdxV + 1, kIdxBa + 1) = t343 * t367 * t399
    * (-1.0 / 2.0);
  q_d(kIdxV + 1, kIdxBa + 2) = t663;
  q_d(kIdxV + 2, kIdxP + 0) = t408;
  q_d(kIdxV + 2, kIdxP + 1) = t412;
  q_d(kIdxV + 2, kIdxP + 2) = t510;
  q_d(kIdxV + 2, kIdxV + 0) = t522 + t534 + t535 + t536
    - t373
    * (t440 + t515
        - dt * t343 * (t513 + t514 + t362 * t512 * (1.0 / 2.0))
        * (1.0 / 3.0)) - dt * t346 * t347 * t369
    - dt * t344 * t354 * t372 - dt * t343 * t346 * t369 * t398 * (1.0 / 3.0)
    - dt * t343 * t344 * t372 * t399 * (1.0 / 3.0);
  q_d(kIdxV + 2, kIdxV + 1) = t560;
  q_d(kIdxV + 2, kIdxV + 2) = -t371
    * (t561 - dt * t509
        + dt * t343 * (t394 * t489 - t429 * t506) * (1.0 / 3.0))
    + t373
    * (t561 + dt * t506
        - dt * t343 * (t397 * t512 - t429 * t509) * (1.0 / 3.0))
    + t374
    * (dt * t507 - dt * t343 * (t390 * t497 - t447 * t447) * (1.0 / 3.0)
        + t343 * t390 * t447) + dt * t347 * t413 + dt * t354 * t414
    + dt * t355 * t415 + dt * t343 * t398 * t413 * (1.0 / 3.0)
    + dt * t343 * t399 * t414 * (1.0 / 3.0)
    + dt * t343 * t400 * t415 * (1.0 / 3.0);
  q_d(kIdxV + 2, kIdxQ + 0) = t580;
  q_d(kIdxV + 2, kIdxQ + 1) = t608
    + t371
    * (dt * t343 * (t600 - e_w2 * e_w3 * t394 * (1.0 / 2.0)) * (1.0 / 3.0)
        - e_w1 * t343 * t394 * (1.0 / 2.0))
    + t373
    * (t602 + t610
        - dt * t343 * (t490 + t601 - t390 * t432 * (1.0 / 2.0))
        * (1.0 / 3.0));
  q_d(kIdxV + 2, kIdxQ + 2) = t647 - t374 * t641
    - t373
    * (t562
        + dt * t343 * (t642 - e_w2 * e_w3 * t397 * (1.0 / 2.0))
        * (1.0 / 3.0));
  q_d(kIdxV + 2, kIdxBw + 0) = dt * t343 * t390 * t502
    * (-1.0 / 6.0);
  q_d(kIdxV + 2, kIdxBw + 1) = t656;
  q_d(kIdxV + 2, kIdxBw + 2) = dt * t343 * t394 * t503
    * (-1.0 / 6.0);
  q_d(kIdxV + 2, kIdxBa + 0) = t659;
  q_d(kIdxV + 2, kIdxBa + 1) = t343 * t372 * t399
    * (-1.0 / 2.0);
  q_d(kIdxV + 2, kIdxBa + 2) = t343 * t370 * t400
    * (-1.0 / 2.0);
  q_d(kIdxQ + 0, kIdxP + 0) = t564;
  q_d(kIdxQ + 0, kIdxP + 2) = t565;
  q_d(kIdxQ + 0, kIdxV + 0) = t573;
  q_d(kIdxQ + 0, kIdxV + 2) = t580;
  q_d(kIdxQ + 0, kIdxQ + 0) = t374
    * (dt - dt * t343 * t432 * (1.0 / 3.0)) + dt * t343 * t502 * (1.0 / 3.0);
  q_d(kIdxQ + 0, kIdxQ + 2) = t652;
  q_d(kIdxQ + 0, kIdxBw + 0) = t343 * t502 * (-1.0 / 2.0);
  q_d(kIdxQ + 1, kIdxP + 0) = dt * t343 * t362 * t373
    * (1.0 / 6.0);
  q_d(kIdxQ + 1, kIdxP + 1) = t581;
  q_d(kIdxQ + 1, kIdxP + 2) = dt * t343 * t373 * t397
    * (-1.0 / 6.0);
  q_d(kIdxQ + 1, kIdxV + 0) = -t371 * (t452 + t623)
    - t374
    * (dt * t343 * (t566 - e_w3 * t425) * (1.0 / 3.0)
        - e_w3 * t343 * t362 * (1.0 / 2.0))
    + t373 * (-t451 + t452 + dt * t343 * (t436 + t582 - t583) * (1.0 / 3.0));
  q_d(kIdxQ + 1, kIdxV + 1) = t599;
  q_d(kIdxQ + 1, kIdxV + 2) = t608
    + t373 * (t602 + t610 - dt * t343 * (t490 + t601 - t609) * (1.0 / 3.0))
    - t371 * (t602 - dt * t343 * (t600 - t601) * (1.0 / 3.0));
  q_d(kIdxQ + 1, kIdxQ + 0) = -t374
    * (e_w3 * t343 * (1.0 / 2.0) - dt * e_w1 * e_w2 * t343 * (1.0 / 6.0))
    - dt * t343 * t373 * t613 * (1.0 / 3.0);
  q_d(kIdxQ + 1, kIdxQ + 1) = t373
    * (dt - dt * t343 * t435 * (1.0 / 3.0)) + dt * t343 * t563 * (1.0 / 3.0)
    + dt * t343 * t371 * t429 * (1.0 / 3.0)
    + dt * t343 * t374 * t431 * (1.0 / 3.0);
  q_d(kIdxQ + 1, kIdxQ + 2) = t655
    - t373 * (t614 - dt * e_w2 * e_w3 * t343 * (1.0 / 6.0))
    - dt * e_w2 * e_w3 * t343 * t374 * (1.0 / 3.0);
  q_d(kIdxQ + 1, kIdxBw + 0) = dt * e_w3 * t343 * t502
    * (1.0 / 6.0);
  q_d(kIdxQ + 1, kIdxBw + 1) = t343 * t563 * (-1.0 / 2.0);
  q_d(kIdxQ + 1, kIdxBw + 2) = dt * e_w1 * t343 * t503
    * (-1.0 / 6.0);
  q_d(kIdxQ + 2, kIdxP + 0) = t615;
  q_d(kIdxQ + 2, kIdxP + 1) = t616;
  q_d(kIdxQ + 2, kIdxP + 2) = t617;
  q_d(kIdxQ + 2, kIdxV + 0) = -t374 * t622
    - t371 * (t451 + t452 - dt * t343 * (t442 + t582 - t583) * (1.0 / 3.0))
    + t373 * (t452 - t623);
  q_d(kIdxQ + 2, kIdxV + 1) = -t374 * t631 - t371 * t635
    - t373 * (t626 - t627);
  q_d(kIdxQ + 2, kIdxV + 2) = t647 - t374 * t641
    - t373 * (t562 + dt * t343 * (t642 - t643) * (1.0 / 3.0));
  q_d(kIdxQ + 2, kIdxQ + 0) = t652;
  q_d(kIdxQ + 2, kIdxQ + 1) = t655 - t373 * (t614 - t653)
    - dt * e_w2 * e_w3 * t343 * t374 * (1.0 / 3.0);
  q_d(kIdxQ + 2, kIdxQ + 2) = t371
    * (dt - dt * t343 * t441 * (1.0 / 3.0)) + dt * t343 * t503 * (1.0 / 3.0)
    + dt * t343 * t373 * t429 * (1.0 / 3.0)
    + dt * t343 * t374 * t430 * (1.0 / 3.0);
  q_d(kIdxQ + 2, kIdxBw + 0) = dt * e_w2 * t343 * t502
    * (-1.0 / 6.0);
  q_d(kIdxQ + 2, kIdxBw + 1) = t657;
  q_d(kIdxQ + 2, kIdxBw + 2) = t343 * t503 * (-1.0 / 2.0);
  q_d(kIdxBw + 0, kIdxV + 0) = dt * t343 * t362 * t502
    * (-1.0 / 6.0);
  q_d(kIdxBw + 0, kIdxV + 2) = dt * t343 * t390 * t502
    * (-1.0 / 6.0);
  q_d(kIdxBw + 0, kIdxQ + 0) = t343 * t502 * (-1.0 / 2.0);
  q_d(kIdxBw + 0, kIdxQ + 2) = dt * e_w2 * t343 * t502
    * (-1.0 / 6.0);
  q_d(kIdxBw + 0, kIdxBw + 0) = dt * t502;
  q_d(kIdxBw + 1, kIdxV + 0) = dt * t343 * t362 * t563
    * (-1.0 / 6.0);
  q_d(kIdxBw + 1, kIdxV + 1) = dt * t343 * t388 * t563
    * (-1.0 / 6.0);
  q_d(kIdxBw + 1, kIdxV + 2) = t656;
  q_d(kIdxBw + 1, kIdxQ + 1) = t343 * t563 * (-1.0 / 2.0);
  q_d(kIdxBw + 1, kIdxQ + 2) = t657;
  q_d(kIdxBw + 1, kIdxBw + 1) = dt * t563;
  q_d(kIdxBw + 2, kIdxV + 0) = dt * t343 * t362 * t503
    * (-1.0 / 6.0);
  q_d(kIdxBw + 2, kIdxV + 1) = dt * t343 * t381 * t503
    * (-1.0 / 6.0);
  q_d(kIdxBw + 2, kIdxV + 2) = dt * t343 * t394 * t503
    * (-1.0 / 6.0);
  q_d(kIdxBw + 2, kIdxQ + 1) = dt * e_w1 * t343 * t503
    * (-1.0 / 6.0);
  q_d(kIdxBw + 2, kIdxQ + 2) = t343 * t503 * (-1.0 / 2.0);
  q_d(kIdxBw + 2, kIdxBw + 2) = dt * t503;
  q_d(kIdxBa + 0, kIdxP + 0) = dt * t343 * t346 * t398
    * (-1.0 / 6.0);
  q_d(kIdxBa + 0, kIdxP + 1) = dt * t343 * t366 * t398
    * (-1.0 / 6.0);
  q_d(kIdxBa + 0, kIdxP + 2) = t658;
  q_d(kIdxBa + 0, kIdxV + 0) = t343 * t346 * t398
    * (-1.0 / 2.0);
  q_d(kIdxBa + 0, kIdxV + 1) = t343 * t366 * t398
    * (-1.0 / 2.0);
  q_d(kIdxBa + 0, kIdxV + 2) = t659;
  q_d(kIdxBa + 0, kIdxBa + 0) = dt * t398;
  q_d(kIdxBa + 1, kIdxP + 0) = t660;
  q_d(kIdxBa + 1, kIdxP + 1) = dt * t343 * t367 * t399
    * (-1.0 / 6.0);
  q_d(kIdxBa + 1, kIdxP + 2) = dt * t343 * t372 * t399
    * (-1.0 / 6.0);
  q_d(kIdxBa + 1, kIdxV + 0) = t661;
  q_d(kIdxBa + 1, kIdxV + 1) = t343 * t367 * t399
    * (-1.0 / 2.0);
  q_d(kIdxBa + 1, kIdxV + 2) = t343 * t372 * t399
    * (-1.0 / 2.0);
  q_d(kIdxBa + 1, kIdxBa + 1) = dt * t399;
  q_d(kIdxBa + 2, kIdxP + 0) = dt * t343 * t345 * t400
    * (-1.0 / 6.0);
  q_d(kIdxBa + 2, kIdxP + 1) = t662;
  q_d(kIdxBa + 2, kIdxP + 2) = dt * t343 * t370 * t400
    * (-1.0 / 6.0);
  q_d(kIdxBa + 2, kIdxV + 0) = t343 * t345 * t400
    * (-1.0 / 2.0);
  q_d(kIdxBa + 2, kIdxV + 1) = t663;
  q_d(kIdxBa + 2, kIdxV + 2) = t343 * t370 * t400
    * (-1.0 / 2.0);
  q_d(kIdxBa + 2, kIdxBa + 2) = dt * t400;

  return q_d;
}
