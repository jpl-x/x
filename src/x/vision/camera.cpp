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

#include <iostream>
#include <x/vision/camera.h>
using namespace x;

Camera::Camera() {}

Camera::Camera(double fx, double fy, double cx, double cy, double s,
               unsigned int img_width, unsigned int img_height)
    : s_(s), img_width_(img_width), img_height_(img_height) {
  fx_ = img_width * fx;
  fy_ = img_height * fy;
  cx_ = img_width * cx;
  cy_ = img_height * cy;

  inv_fx_ = 1.0 / fx_;
  inv_fy_ = 1.0 / fy_;
  cx_n_ = cx_ * inv_fx_;
  cy_n_ = cy_ * inv_fy_;
  s_term_ = 1.0 / (2.0 * std::tan(s / 2.0));
  // populateLUT();
}

void Camera::populateLUT() {
  LUT_distortion_.reserve(img_width_ * img_height_);
  for (size_t i = 0; i < img_width_; i++) {
    for (size_t j = 0; j < img_height_; j++) {
      const double cam_dist_x = i * inv_fx_ - cx_n_;
      const double cam_dist_y = j * inv_fy_ - cy_n_;

      const double dist_r =
          sqrt(cam_dist_x * cam_dist_x + cam_dist_y * cam_dist_y);

      double distortion_factor = 1.0;
      if (dist_r > 0.01)
        distortion_factor = inverseTf(dist_r) / dist_r;

      const double xn = distortion_factor * cam_dist_x;
      const double yn = distortion_factor * cam_dist_y;

      LUT_distortion_.emplace_back(cv::Point2d(xn * fx_ + cx_, yn * fy_ + cy_));
    }
  }
}

unsigned int Camera::getWidth() const { return img_width_; }

unsigned int Camera::getHeight() const { return img_height_; }

double Camera::getInvFx() const { return inv_fx_; }

double Camera::getInvFy() const { return inv_fy_; }

double Camera::getCxN() const { return cx_n_; }

double Camera::getCyN() const { return cy_n_; }

void Camera::undistort(FeatureList& features) const
{
  // Undistort each point in the input vector
  for(unsigned int i = 0; i < features.size(); i++)
    undistort(features[i]);
}

void Camera::undistort(Feature& feature) const
{
  const double cam_dist_x = feature.getXDist() * inv_fx_ - cx_n_;
  const double cam_dist_y = feature.getYDist() * inv_fy_ - cy_n_;

  const double dist_r = sqrt(cam_dist_x * cam_dist_x + cam_dist_y * cam_dist_y);

  double distortion_factor = 1.0;
  if(dist_r > 0.01)
    distortion_factor = inverseTf(dist_r) / dist_r;

  const double xn = distortion_factor * cam_dist_x;
  const double yn = distortion_factor * cam_dist_y;

  feature.setX(xn * fx_ + cx_);
  feature.setY(yn * fy_ + cy_);
}

Feature Camera::normalize(const Feature &feature) const {
  Feature normalized_feature(feature.getTimestamp(),
                             feature.getX() * inv_fx_ - cx_n_,
                             feature.getY() * inv_fy_ - cy_n_);
  normalized_feature.setXDist(feature.getXDist() * inv_fx_ - cx_n_);
  normalized_feature.setYDist(feature.getYDist() * inv_fx_ - cx_n_);

  return normalized_feature;
}

/** \brief Normalized the image coordinates for all features in the input track
 *  \param track Track to normalized
 *  \param max_size Max size of output track, cropping from the end (default 0:
 * no cropping) \return Normalized track
 */
Track Camera::normalize(const Track &track, const size_t max_size) const {
  // Determine output track size
  const size_t track_size = track.size();
  size_t size_out;
  if (max_size)
    size_out = std::min(max_size, track_size);
  else
    size_out = track_size;

  Track normalized_track(size_out, Feature());
  const size_t start_idx(track_size - size_out);
  for (size_t j = start_idx; j < track_size; ++j)
    normalized_track[j - start_idx] = normalize(track[j]);

  return normalized_track;
}

/** \brief Normalized the image coordinates for all features in the input track
 * list \param tracks List of tracks \param max_size Max size of output tracks,
 * cropping from the end (default 0: no cropping) \return Normalized list of
 * tracks
 */
TrackList Camera::normalize(const TrackList &tracks,
                            const size_t max_size) const {
  const size_t n = tracks.size();
  TrackList normalized_tracks(n, Track());

  for (size_t i = 0; i < n; ++i)
    normalized_tracks[i] = normalize(tracks[i], max_size);

  return normalized_tracks;
}

double Camera::inverseTf(const double dist) const {
  if (s_ == 0.0)
    return dist;
  else
    return std::tan(dist * s_) * s_term_;
}
