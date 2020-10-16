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

#ifndef JPL_VPP_TILED_IMAGE_H_
#define JPL_VPP_TILED_IMAGE_H_

#include <x/vision/feature.h>

#include <opencv2/highgui/highgui.hpp>

namespace x
{
class TiledImage : public cv::Mat
{
public:
  TiledImage();
  TiledImage(const Mat& cv_img);
  TiledImage(const TiledImage& img);
  TiledImage(const Mat& img,
             double timestamp,
             unsigned int frame_number,
             unsigned int n_tiles_h,
             unsigned int n_tiles_w,
             unsigned int max_feat_per_tile);
  TiledImage& operator=(const TiledImage& other);
  TiledImage clone() const;
  void setTileSize(const double tile_height, const double tile_width);

  void setTileParams(unsigned int n_tiles_h,
                     unsigned int n_tiles_w,
                     unsigned int max_feat_per_tile);

  double getTimestamp() const;
  unsigned int getFrameNumber() const;
  double getTileHeight() const;
  double getTileWidth() const;
  unsigned int getNTilesH() const;
  unsigned int getNTilesW() const;
  unsigned int getMaxFeatPerTile() const;

  // Get the feature count for a specific tile
  unsigned int getFeatureCountAtTile(unsigned int row, unsigned int col) const;

  // Increment feature count for a specific tile
  void incrementFeatureCountAtTile(unsigned int row, unsigned int col);
  void resetFeatureCounts();
  void setTileForFeature(Feature& feature) const;
  // Plot one feature in the image
  void plotFeature(Feature& feature, cv::Scalar color);
  // Plot tiles on the image
  void plotTiles();

private:
  double timestamp_ = 0.0;
  unsigned int frame_number_ = 0;
  double tile_height_ = 0.0;
  double tile_width_  = 0.0;
  unsigned int n_tiles_h_ = 1;          // number of tiles along height
  unsigned int n_tiles_w_ = 1;          // number of tiles along width (1 <=> no tiling)
  unsigned int max_feat_per_tile_ = 40; // maximum count of features per image tile
  unsigned int n_tiles_ = 1;            // total number of image tiles (n_tiles_w_ * n_tiles_h)
  std::vector<unsigned int> tiles_;     // n_tiles_w_ x n_tiles_h 1D array that holds count
                                        // of features per tile (row, col)
};
}

#endif
