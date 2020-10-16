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

#include <x/vision/tiled_image.h>

#include <opencv2/imgproc.hpp>

using namespace x;

TiledImage::TiledImage() : cv::Mat()
{
}

TiledImage::TiledImage(const Mat& cv_img) : cv::Mat(cv_img)
{
}

TiledImage::TiledImage(const TiledImage& img)
  : cv::Mat(img)
  , timestamp_(img.timestamp_)
  , frame_number_(img.frame_number_)
  , n_tiles_h_(img.n_tiles_h_)
  , n_tiles_w_(img.n_tiles_w_)
  , max_feat_per_tile_(img.max_feat_per_tile_)
{
  n_tiles_     = img.n_tiles_;
  tile_height_ = img.tile_height_;
  tile_width_  = img.tile_width_;
  tiles_       = img.tiles_;

}

TiledImage::TiledImage(const Mat& img, double timestamp, unsigned int frame_number, unsigned int n_tiles_h,
                       unsigned int n_tiles_w, unsigned int max_feat_per_tile)
  : cv::Mat(img)
  , timestamp_(timestamp)
  , frame_number_(frame_number)
  , n_tiles_h_(n_tiles_h)
  , n_tiles_w_(n_tiles_w)
  , max_feat_per_tile_(max_feat_per_tile)
{
  n_tiles_ = n_tiles_h * n_tiles_w;
  tile_height_ = (double)img.rows / n_tiles_h;  // (rows, cols) are int in OpenCV!
  tile_width_ = (double)img.cols / n_tiles_w;
  tiles_ = std::vector<unsigned int>(n_tiles_, 0);
}

/** \brief Copy assignment operator
 */
TiledImage& TiledImage::operator=(const TiledImage& other)
{
  if (&other == this)
    return *this;

  cv::Mat::operator=(other);
  timestamp_ = other.timestamp_;
  frame_number_ = other.frame_number_;
  tile_height_ = other.tile_height_;
  tile_width_ = other.tile_width_;
  n_tiles_h_ = other.n_tiles_h_;
  n_tiles_w_ = other.n_tiles_w_;
  max_feat_per_tile_ = other.max_feat_per_tile_;
  n_tiles_ = other.n_tiles_;
  tiles_ = other.tiles_;

  return *this;
}

TiledImage TiledImage::clone() const
{
  cv::Mat base(*this);
  TiledImage clone(base.clone());

  clone.timestamp_ = timestamp_;
  clone.frame_number_ = frame_number_;
  clone.tile_height_ = tile_height_;
  clone.tile_width_ = tile_width_;
  clone.n_tiles_h_ = n_tiles_h_;
  clone.n_tiles_w_ = n_tiles_w_;
  clone.max_feat_per_tile_ = max_feat_per_tile_;
  clone.n_tiles_ = n_tiles_;
  clone.tiles_ = tiles_;

  return clone;
}

void TiledImage::setTileSize(const double tile_height, const double tile_width)
{
  tile_height_ = tile_height;
  tile_width_ = tile_width;
}

void TiledImage::setTileParams(unsigned int n_tiles_h, unsigned int n_tiles_w, unsigned int max_feat_per_tile)
{
  n_tiles_h_ = n_tiles_h;
  n_tiles_w_ = n_tiles_w;
  max_feat_per_tile_ = max_feat_per_tile;
  n_tiles_ = n_tiles_h * n_tiles_w;
  tile_height_ = (double)rows / n_tiles_h;  // (rows, cols) are int in OpenCV!
  tile_width_ = (double)cols / n_tiles_w;
  tiles_ = std::vector<unsigned int>(n_tiles_, 0);
}

double TiledImage::getTimestamp() const
{
  return timestamp_;
}

unsigned int TiledImage::getFrameNumber() const
{
  return frame_number_;
}

double TiledImage::getTileHeight() const
{
  return tile_height_;
}

double TiledImage::getTileWidth() const
{
  return tile_width_;
}

unsigned int TiledImage::getNTilesH() const
{
  return n_tiles_h_;
}

unsigned int TiledImage::getNTilesW() const
{
  return n_tiles_w_;
}

unsigned int TiledImage::getMaxFeatPerTile() const
{
  return max_feat_per_tile_;
}

unsigned int TiledImage::getFeatureCountAtTile(unsigned int row, unsigned int col) const
{
  return tiles_[row * n_tiles_w_ + col];
}

void TiledImage::incrementFeatureCountAtTile(unsigned int row, unsigned int col)
{
  tiles_[row * n_tiles_w_ + col] += 1;
}

void TiledImage::resetFeatureCounts()
{
  std::fill(tiles_.begin(), tiles_.end(), 0);
}

void TiledImage::setTileForFeature(Feature& feature) const
{
  // Find the tile column
  double c = feature.getXDist() - tile_width_ - 0.5;
  unsigned int col = 0;
  while (c > 0)
  {
    col += 1;
    c -= tile_width_;
  }

  // Find the tile row
  double r = rows - feature.getYDist() - 0.5;
  unsigned int row = n_tiles_h_ - 1;
  while (r > tile_height_)
  {
    row -= 1;
    r -= tile_height_;
  }

  // Store these in feature
  feature.setTile(row, col);
}

void TiledImage::plotFeature(Feature& feature, cv::Scalar color)
{
  // Display feature point
  const double l = 2;  // half length of feature display square
  cv::rectangle(*this, cv::Point(feature.getXDist() - l, feature.getYDist() - l),
                cv::Point(feature.getXDist() + l, feature.getYDist() + l), color, -1, 8);
}

void TiledImage::plotTiles()
{
  const cv::Scalar color = cv::Scalar(0, 0, 0);
  const int thickness = 1;

  // draw the column dividers
  for (unsigned int c = 1; c < n_tiles_w_; ++c)
  {
    const double xPos = (cols / n_tiles_w_) * c;
    // Note: top-left corner is (-0.5,-0.5) in OpenCV
    cv::line(*this, cv::Point(xPos, -0.5), cv::Point(xPos, rows - 0.5), color, thickness);
  }

  // draw the row dividers
  for (unsigned int r = 1; r < n_tiles_h_; ++r)
  {
    const double yPos = (rows / n_tiles_h_) * r;
    cv::line(*this, cv::Point(-0.5, yPos), cv::Point(cols - 0.5, yPos), color, thickness);
  }
}
