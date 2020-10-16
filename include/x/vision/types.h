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

#ifndef X_HAT_TYPES_H
#define X_HAT_TYPES_H

#include <x/common/types.h>
#include <x/vision/feature.h>
#include <x/vision/tiled_image.h>

#include <vector>

#define JPL_VPP_WARN_STREAM(x) std::cerr<<"\033[0;33m[WARN] "<<x<<"\033[0;0m"<<std::endl; 

namespace x {

  using Matrix3 = Eigen::Matrix3d;
  using QuaternionArray = std::vector<Quaternion>;
  using Vector3Array = std::vector<Vector3>;
  // Feature match
  struct Match
  {
    Feature previous;
    Feature current;
  };
  // Pixel coordinates
  struct PixelCoor
  {
    int x; // x pixel coordinate
    int y; // y pixel coordinate
  };
  typedef std::vector<Feature> FeatureList;
  typedef std::vector<Feature> Track;
  typedef std::vector<FeatureList> TrackList;
  typedef std::vector<Match> MatchList;
  typedef std::vector<TiledImage> ImagePyramid;
  typedef std::vector<cv::KeyPoint> Keypoints;

  //TODO(jeff) Get rid of this. Use Eigen::Quaternion instead.
  struct Attitude
  {
    double ax = 0;  //< quat x
    double ay = 0;  //< quat y
    double az = 0;  //< quat z
    double aw = 0;  //< quat orientation
    Attitude(double ax, double ay, double az, double aw)
      : ax(ax), ay(ay), az(az), aw(aw) {}
    Attitude() = default;
  };

  //TODO(jeff) Get rid of this. Use Vector3 instead.
  struct Translation
  {
    double tx = 0;  //< cam position x
    double ty = 0;  //< cam position y
    double tz = 0;  //< cam position z
    Translation(double tx, double ty, double tz) : tx(tx), ty(ty), tz(tz) {}
    Translation(void) = default;
  };

  //TODO(jeff) Get rid of below and use the above
  using AttitudeList = std::vector<Attitude>;
  using TranslationList = std::vector<Translation>;

} // namespace x

#endif
