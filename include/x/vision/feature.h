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

#ifndef X_FEATURE_H_
#define X_FEATURE_H_

namespace x
{
class Feature
{
public:

  /************************** Constructors **************************/

  Feature();

  Feature(double timestamp,
          double x,
          double y);

  Feature(double timestamp,
          unsigned int frame_number,
          double x,
          double y,
          double x_dist,
          double y_dist);

  Feature(double timestamp,
          unsigned int frame_number,
          double x_dist,
          double y_dist,
          unsigned int pyramid_level,
          float fast_score);

  /**************************** Setters *****************************/

  void setX(const double x) { x_ = x; };

  void setY(const double y) { y_ = y; };

  void setXDist(const double x_dist) { x_dist_ = x_dist; };

  void setYDist(const double y_dist) { y_dist_ = y_dist; };
  
  void setTile(int row, int col) { tile_row_ = row; tile_col_ = col; };

  /**************************** Getters *****************************/

  double getTimestamp() const {return timestamp_; };

  double getX() const { return x_; };

  double getY() const { return y_; }

  double getXDist() const { return x_dist_; };

  double getYDist() const { return y_dist_; };

  unsigned int getPyramidLevel() const { return pyramid_level_; };

  float getFastScore() const { return fast_score_; };

  int getTileRow() const { return tile_row_; };

  int getTileCol() const { return tile_col_; };
  
private:
  /**
   * Image timestamp [s]
   */
  double timestamp_ = 0.0;

  /**
   * Image ID
   */
  unsigned int frame_number_ = 0;

  /**
   * (Undistorted) X pixel coordinate
   */
  double x_ = 0.0;

  /**
   * (Undistorted) Y pixel coordinate
   */
  double y_ = 0.0;

  /**
   * Distorted X pixel coordinate
   */
  double x_dist_ = 0.0;

  /**
   * Distorted Y pixel coordinate
   */
  double y_dist_ = 0.0;

  /**
   * Pyramid level in which the feature was detected
   */
  unsigned int pyramid_level_ = 0;
  
  /**
   * FAST score
   */
  float fast_score_ = 0.0;

  /**
   * Row index of the image tile in which the feature is located
   */
  int tile_row_ = -1;

  /**
   * Column index of the image tile in which the feature is located
   */
  int tile_col_ = -1;
};
}

#endif // X_FEATURE
