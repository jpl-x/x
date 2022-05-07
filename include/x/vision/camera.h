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

#ifndef JPL_VPP_CAMERA_H_
#define JPL_VPP_CAMERA_H_

#include <x/vision/types.h>
#include <x/vision/track.h>
#include "opencv2/highgui/highgui.hpp"

namespace x
{
class Camera
{
public:
  Camera();
  Camera(double fx,
         double fy,
         double cx,
         double cy,
         double s,
         unsigned int img_width,
         unsigned int img_height);
  unsigned int getWidth() const;
  unsigned int getHeight() const;
  double getInvFx() const;
  double getInvFy() const;
  double getCxN() const;
  double getCyN() const;
  void undistort(FeatureList& features) const;
  void undistort(Feature& feature) const;
  // Returns image coordinates in normal plane
  Feature normalize(const Feature& feature) const;
  Track normalize(const Track& track, const size_t max_size = 0) const;
  TrackList normalize(const TrackList& tracks, const size_t max_size = 0) const;
private:
  double fx_ = 0.0; // Focal length
  double fy_ = 0.0;
  double cx_ = 0.0; // Principal point
  double cy_ = 0.0;
  double s_ = 0.0;  // Distortion
  unsigned int img_width_ = 0.0;
  unsigned int img_height_ = 0.0;
  // Distortion terms we only want to compute once
  double inv_fx_ = -1; // Inverse focal length
  double inv_fy_ = -1;
  double cx_n_   = -1; // Principal point in normalized coordinates
  double cy_n_   = -1;
  double s_term_ = -1; // Distortion term
  // Inverse FOV distortion transformation
  double inverseTf(const double dist) const;
};
}

#endif
