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

#ifndef JPL_VPP_TRACKER_H_
#define JPL_VPP_TRACKER_H_

#include <x/vision/camera.h>
#include <x/vision/types.h>

namespace x
{
class Tracker
{
public:

  Tracker();
  Tracker(const Camera& cam,
          const unsigned int fast_detection_delta,
          const bool non_max_supp,
          const unsigned int block_half_length,
          unsigned int margin,
          unsigned int n_feat_min,
          int outlier_method,
          double outlier_param1,
          double outlier_param2);
  void setParams(const Camera& cam,
                 const unsigned int fast_detection_delta,
                 const bool non_max_supp,
                 const unsigned int block_half_length,
                 unsigned int margin,
                 unsigned int n_feat_min,
                 int outlier_method,
                 double outlier_param1,
                 double outlier_param2);
  // Get latest matches
  MatchList getMatches() const;
  // Get ID of last image processed
  int getImgId() const;
  // Tracks features from last image into the current one
  void track(TiledImage& current_img,
             double timestamp,
             unsigned int frame_number);
  bool checkMatches();
  // Plots matches in the image
  static void plotMatches(MatchList matches, TiledImage& img);
private:
  MatchList matches_;
  int img_id_ = 0;
  FeatureList previous_features_;
  double previous_timestamp_ = 0.0;
  TiledImage previous_img_;
  // Tracker params
  Camera camera_;
  unsigned int fast_detection_delta_ = 9; // the intensity difference threshold for the FAST feature detector
  bool non_max_supp_ = true;             // if true, non-maximum suppression is applied to detected corners (keypoints)
  unsigned int block_half_length_ = 20; // the blocked region for other features to occupy will be of size (2 * block_half_length_ + 1)^2 [px]
  unsigned int margin_ = 20;    // the margin from the edge of the image within which all detected features must fall into
  unsigned int n_feat_min_ = 40;
  int outlier_method_ = 8;
  double outlier_param1_ = 0.3;
  double outlier_param2_ = 0.99;
  // Placeholder members from an older version to enable multi-scale pyramid
  // detection and tracking
  // TODO(Jeff or Roland): implement based on best Mars Heli test results
  ImagePyramid img_pyramid_;
  unsigned int pyramid_depth_ = 1; // depth of the half-sampling pyramid
  // Feature detection. One can avoid detection in the neighborhood of features
  // already tracked and specified in the optional argument old_points.
  void featureDetection(const TiledImage& img,
                        FeatureList& new_pts,
                        double timestamp,
                        unsigned int frame_number,
                        FeatureList& old_pts);
  // Get the image pyramid given the highest resolution image using down-sampling
  void getImagePyramid (const TiledImage& img, ImagePyramid& pyramid);
  // Get FAST features for an image pyramid at all levels. Optionally avoids
  // detection on the neighborhood of old features.
  void getFASTFeaturesPyramid(ImagePyramid& pyramid,
                              double timestamp,
                              unsigned int frame_number,
                              FeatureList& features,
                              FeatureList& old_features);
  // Get fast features for a single image pyramid level. Avoids detection
  // on the neighborhood of old features, if applicable.
  void getFASTFeaturesImage(TiledImage& img,
                            double timestamp,
                            unsigned int frame_number,
                            unsigned int pyramid_level,
                            FeatureList& features,
                            FeatureList& old_features);
  // Computes the neighborhood mask for the feature list at input.
  void computeNeighborhoodMask(const FeatureList& features,
                               const cv::Mat& img,
                               cv::Mat& mask);
  // Check whether a feature falls within a certain border at a certain pyramid level
  bool isFeatureInsideBorder(Feature& feature,
                             const TiledImage& img,
                             unsigned int margin,
                             unsigned int pyramid_level) const;
  // Get scaled image coordinate for a feature
  void getScaledPixelCoordinate(const Feature& feature,
                                PixelCoor& scaled_coord) const;

  // Compare the score of two FAST features (return true if
  // feature1.fastScore > feature2.fastScore)
  bool compareFASTFeaturesScore(const Feature& feature1,
                                const Feature& feature2) const;
  // Appends candidate features which are far enough from existing features. The
  // neighborhood mask needs to be provided beforehands. The order of the
  // features matters.
  void appendNonNeighborFeatures(TiledImage& img,
                                 FeatureList& features,
                                 FeatureList& candidate_features,
                                 cv::Mat& mask);
  void removeOverflowFeatures(TiledImage& img1, 
                              TiledImage& img2,
                              FeatureList& features1,
                              FeatureList& features2) const;
  // Feature tracking using KLT
  void featureTracking(const cv::Mat& img1,
                       const cv::Mat& img2,
                       FeatureList& pts1,
                       FeatureList& pts2,
                       const double timestamp2,
                       const unsigned int frame_number2) const;
};
}
#endif
