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

#include <x/vision/tracker.h>
#include <x/vision/timing.h>

#include <math.h>
#include <iostream>
#include <boost/bind.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace x;

Tracker::Tracker()
{}

Tracker::Tracker(const Camera& cam,
                 const unsigned int fast_detection_delta,
                 const bool non_max_supp,
                 const unsigned int block_half_length,
                 unsigned int margin,
                 unsigned int n_feat_min,
                 int outlier_method,
                 double outlier_param1,
                 double outlier_param2)
: camera_(cam)
, fast_detection_delta_(fast_detection_delta)
, non_max_supp_(non_max_supp)
, block_half_length_(block_half_length)
, margin_(margin)
, n_feat_min_(n_feat_min)
, outlier_method_(outlier_method)
, outlier_param1_(outlier_param1)
, outlier_param2_(outlier_param2)
{}

void Tracker::setParams(const Camera& cam,
                        const unsigned int fast_detection_delta,
                        const bool non_max_supp,
                        const unsigned int block_half_length,
                        unsigned int margin,
                        unsigned int n_feat_min,
                        int outlier_method,
                        double outlier_param1,
                        double outlier_param2)
{
  camera_ = cam;
  fast_detection_delta_ = fast_detection_delta;
  non_max_supp_ = non_max_supp;
  block_half_length_ = block_half_length;
  margin_ = margin;
  n_feat_min_ = n_feat_min;
  outlier_method_ = outlier_method;
  outlier_param1_ = outlier_param1;
  outlier_param2_ = outlier_param2;
}

MatchList Tracker::getMatches() const
{
  return matches_;
}

int Tracker::getImgId() const
{
  return img_id_;
}

void Tracker::track(TiledImage& current_img,
                    double timestamp,
                    unsigned int frame_number)
{
  // Increment the current image number
  img_id_++;

  #ifdef TIMING
  // Set up and start internal timing
  clock_t clock1, clock2, clock3, clock4, clock5, clock6, clock7;
  clock1 = clock();
  #endif

  // Build image pyramid
  getImagePyramid(current_img, img_pyramid_);

  #ifdef TIMING
  clock2 = clock();
  #endif

  //============================================================================
  // Feature detection and tracking
  //============================================================================

  FeatureList current_features;
  MatchList matches;

  // If first image
  if(img_id_ == 1)
  {
    // Detect features in first image
    FeatureList old_features;
    featureDetection(current_img, current_features, timestamp, frame_number, old_features);

    // Undistortion of current features
    camera_.undistort(current_features);

    #ifdef TIMING
    clock3 = clock();

    // Deal with unused timers
    clock4 = clock3;
    clock5 = clock4;
    clock6 = clock5;
    #endif
  }
  else // not first image
  {
    #ifdef TIMING
    // End detection timer even if it did not occur
    clock3 = clock();
    #endif

    // Track features
    featureTracking(previous_img_, current_img,
                    previous_features_, current_features,
                    timestamp, frame_number);

    #ifdef TIMING
    clock4 = clock();
    #endif

    removeOverflowFeatures(previous_img_, current_img, previous_features_, current_features);

    // Refresh features if needed
    if(current_features.size() < n_feat_min_)
    {
      #ifdef DEBUG
      std::cout << "Number of tracked features reduced to "
                << current_features.size()
                << std::endl;
      std::cout << "Triggering re-detection"
                << std::endl;
      #endif

      // Detect and track new features outside of the neighborhood of
      // previously-tracked features
      FeatureList previous_features_new, current_features_new;
      featureDetection(previous_img_,
                       previous_features_new,
                       previous_img_.getTimestamp(),
                       previous_img_.getFrameNumber(),
                       previous_features_);
      featureTracking(previous_img_,
                      current_img,
                      previous_features_new,
                      current_features_new,
                      timestamp,
                      frame_number);

      // Concatenate previously-tracked and newly-tracked features
      previous_features_.insert(previous_features_.end(),
                                          previous_features_new.begin(),
                                          previous_features_new.end());

      current_features.insert(current_features.end(),
                              current_features_new.begin(),
                              current_features_new.end());
    }

    #ifdef TIMING
    clock5 = clock();
    #endif

    // Undistortion of current features
    camera_.undistort(current_features);

    // Undistort features in the previous image
    // TODO(jeff) only do it for new features newly detected
    camera_.undistort(previous_features_);

    //==========================================================================
    // Outlier removal
    //==========================================================================

    unsigned int n_matches = previous_features_.size();
    if (n_matches)
    {
      //TODO(jeff) Store coordinates as cv::Point2f in Feature class to avoid
      // theses conversions
      // Convert features to OpenCV keypoints
      std::vector<cv::Point2f> pts1, pts2;
      pts1.resize(n_matches);
      pts2.resize(n_matches);
      for(unsigned int i = 0; i < pts1.size(); i++)
      {
        pts1[i].x = previous_features_[i].getX();
        pts1[i].y = previous_features_[i].getY();
        pts2[i].x = current_features[i].getX();
        pts2[i].y = current_features[i].getY();
      }

      std::vector<uchar> mask;
      cv::findFundamentalMat(pts1,
                             pts2,
                             outlier_method_,
                             outlier_param1_,
                             outlier_param2_,
                             mask);
      /*cv::findHomography(pts1,
                           current_features,
                           outlier_method_,
                           outlier_param1_,
                           mask,
                           2000,
                           0.995);*/

      FeatureList current_features_refined, previous_features_refined;
      for(size_t i = 0; i < mask.size(); ++i)
      {
        if(mask[i] != 0)
        {
          previous_features_refined.push_back(previous_features_[i]);
          current_features_refined.push_back(current_features[i]);
        }
      }

      std::swap(previous_features_refined, previous_features_);
      std::swap(current_features_refined, current_features);
    }
    #ifdef TIMING
    clock6 = clock();
    #endif

    //==========================================================================
    // Match construction
    //==========================================================================

    // Reset match list
    n_matches = previous_features_.size();
    matches.resize(n_matches);

    // Store the message array into a match list
    for(unsigned int i = 0; i < n_matches; ++i)
    {
      Match current_match;
      current_match.previous = previous_features_[i];
      current_match.current  = current_features[i];

      // Add match to list
      matches[i] = current_match;
    }
  }

  // Store results so it can queried outside the class
  matches_ = matches;

  // Store info to match at next image
  previous_features_           = current_features;
  previous_timestamp_          = timestamp;
  previous_img_                = current_img;

  // Exit the function if this is the first image
  if(img_id_ < 2)
    return;

  #ifdef TIMING
  clock7 = clock();
  #endif

  // Print TIMING info
#ifdef TIMING
  std::cout << "Tracking Info ====================================" << std::endl;
  std::cout << "Number of matches:                   " << matches.size() << std::endl;
  std::cout << "==================================================" << std::endl;
  std::cout <<
      "Times ====================================" << std::endl <<
      "Build image pyramid:               " << (double) (clock2  - clock1)/CLOCKS_PER_SEC*1000 << " ms" << std::endl <<
      "Feature detection:                 " << (double) (clock3  - clock2)/CLOCKS_PER_SEC*1000 << " ms" << std::endl <<
      "Feature tracking:                  " << (double) (clock4  - clock3)/CLOCKS_PER_SEC*1000 << " ms" << std::endl <<
      "Feature re-detection and tracking: " << (double) (clock5  - clock4)/CLOCKS_PER_SEC*1000 << " ms" << std::endl <<
      "Outlier check:                     " << (double) (clock6  - clock5)/CLOCKS_PER_SEC*1000 << " ms" << std::endl <<
      "Feature conversion:                " << (double) (clock7  - clock6)/CLOCKS_PER_SEC*1000 << " ms" << std::endl <<
      "------------------------------------------" << std::endl <<
      "TOTAL:                             " << (double) (clock7 - clock1)/CLOCKS_PER_SEC*1000 << " ms" << std::endl <<
      "==========================================" << std::endl;
#endif

  // Plot matches in the image
  plotMatches(matches_, current_img);
}

/** \brief True if matches are chronological order
 *  @todo Make Match struct a class with this function in it.
 */
bool Tracker::checkMatches()
{
  if (matches_.size())
  {
    Match match(matches_[0]);
    if (match.previous.getTimestamp() < match.current.getTimestamp())
    {
      return true;
    }
  }
  return false;
}

void Tracker::plotMatches(MatchList matches, TiledImage& img)
{
  // Convert grayscale image to color image
#if CV_MAJOR_VERSION == 4
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
#else
    cv::cvtColor(img, img, CV_GRAY2BGR);
#endif

  // Plot tiles
  img.plotTiles();

  // Plot matches
  const cv::Scalar green(100,255,100);
  const unsigned int n = matches.size();
  for(unsigned int ii = 0; ii < n; ii++)
    img.plotFeature(matches[ii].current, green);

  std::string str = std::string("Matches: ") + std::to_string(n);

  cv::putText(img, str, cv::Point((int) 10, (int) img.rows-10), cv::FONT_HERSHEY_PLAIN,
                1.0, green, 1.5, 8, false);
}

void Tracker::featureDetection(const TiledImage& img,
                               FeatureList& new_feature_list,
                               double timestamp,
                               unsigned int frame_number,
                               FeatureList& old_feature_list)
{
  // Pyramidal OpenCV FAST
  ImagePyramid pyr;
  getImagePyramid(img, pyr);

  // Extract new features
  FeatureList feature_list;
  getFASTFeaturesPyramid(pyr, timestamp, frame_number, new_feature_list, old_feature_list);
}

void Tracker::getImagePyramid (const TiledImage& img,
                               ImagePyramid& pyramid)
{
  pyramid.clear();                                  // clear the vector of images
  pyramid.push_back(img);                           // the first pyramid level is the highest resolution image
  for(unsigned int i = 1; i < pyramid_depth_; i++)  // for all pyramid depths
  {
    TiledImage downsampledImage;                       // create a down-sampled image
    downsampledImage.setTileSize(img.getTileHeight(), img.getTileWidth());
    cv::pyrDown(pyramid[i-1],downsampledImage);     // reduce the image size by a factor of 2
    pyramid.push_back(downsampledImage);            // add the down-sampled image to the pyramid
  }
}

void Tracker::getFASTFeaturesPyramid(ImagePyramid& pyramid,
                                     double timestamp,
                                     unsigned int frame_number,
                                     FeatureList& features,
                                     FeatureList& old_features)
{
  // Loop through the image pyramid levels and get the fast features
  for(unsigned int l = 0; l < pyramid.size(); l++)
    getFASTFeaturesImage(pyramid[l], timestamp, frame_number, l, features, old_features);
}

void Tracker::getFASTFeaturesImage(TiledImage& img,
                                   double timestamp,
                                   unsigned int frame_number,
                                   unsigned int pyramid_level,
                                   FeatureList& features,
                                   FeatureList& old_features)
{
  Keypoints keypoints; // vector of cv::Keypoint
  // scaling for points dependent on pyramid level to get their position
  // (x, y) = (x*2^pyramid_level,y*2^pyramid_level)
  double scale_factor = pow(2,(double) pyramid_level);

  // Get FAST features
  cv::FAST(img, keypoints, fast_detection_delta_, non_max_supp_);

  // Create a blocked mask which is 1 in the neighborhood of old features, and 0
  // elsewhere.
  cv::Mat blocked_img = cv::Mat::zeros(img.rows, img.cols, CV_8U);
  computeNeighborhoodMask(old_features, img, blocked_img);

  // For all keypoints, extract the relevant information and put it into the
  // vector of Feature structs
  FeatureList candidate_features;
  for(unsigned int i = 0; i < keypoints.size(); i++)
  {
    Feature feature(timestamp,
                    frame_number,
                    ((double) keypoints[i].pt.x) * scale_factor,
                    ((double) keypoints[i].pt.y) * scale_factor,
                    pyramid_level,
                    keypoints[i].response);

    // Check whether features fall in the bounds of allowed features; if yes,
    // add it to candidate features and store its designated bucket information
    if(isFeatureInsideBorder(feature,
                             img,
                             margin_,
                             feature.getPyramidLevel()))
    {
      candidate_features.push_back(feature);
    }
  }

  // Sort the candidate features by FAST score (highest first)
  std::sort(candidate_features.begin(), candidate_features.end(),
            boost::bind(&Tracker::compareFASTFeaturesScore, this, _1, _2));

  // Append candidate features which are not neighbors of tracked features.
  // Note: old features are not supposed to be in here, see. updateTrailLists()
  appendNonNeighborFeatures(img, features, candidate_features, blocked_img);
}

void Tracker::computeNeighborhoodMask(const FeatureList& features,
                                      const cv::Mat& img,
                                      cv::Mat& mask)
{
  // Loop through candidate features.
  // Create submask for each feature
  for(int i = 0; i < (int) features.size(); i++)
  {
    // Rounded feature coordinates
    double x = std::round(features[i].getXDist());
    double y = std::round(features[i].getYDist());

    // Submask top-left corner and width/height determination.
    int x_tl, y_tl, w, h;
    // Width
    if(x - block_half_length_ < 0)
    {
      x_tl = 0;
      w    = block_half_length_ + 1 + x;
    }
    else if(x + block_half_length_ > img.cols - 1)
    {
      x_tl = x - block_half_length_;
      w    = block_half_length_ + img.cols - x;
    }
    else
    {
      x_tl = x - block_half_length_;
      w    = 2*block_half_length_ + 1;
    }
    // Height
    if(y - block_half_length_ < 0)
    {
      y_tl = 0;
      h    = block_half_length_ + 1 + y;
    }
    else if(y + block_half_length_ > img.rows - 1)
    {
      y_tl = y - block_half_length_;
      h    = block_half_length_ + img.rows - y;
    }
    else
    {
      y_tl = y - block_half_length_;
      h    = 2*block_half_length_ + 1;
    }

    // Submask application
    cv::Mat blocked_box = mask(cv::Rect(x_tl, y_tl, w, h)); // box to block off in the image
    blocked_box.setTo(cv::Scalar(1)); // block out the surrounding area by setting the mask to 1
  }
}

bool Tracker::isFeatureInsideBorder(Feature& feature,
                                    const TiledImage& img,
                                    unsigned int margin,
                                    unsigned int pyramid_level) const
{
  // Return true if the feature falls within the border at the specified pyramid level
  int xMin = (int) margin;
  int xMax = (int) floor(img.cols/pow(2,(double) pyramid_level) - margin) - 1;
  int yMin = (int) margin;
  int yMax = (int) floor(img.rows/pow(2,(double) pyramid_level) - margin) - 1;
  PixelCoor feature_coord = {0, 0};
  getScaledPixelCoordinate(feature, feature_coord);
  return ( (feature_coord.x >= xMin) && (feature_coord.x <= xMax) &&
           (feature_coord.y >= yMin) && (feature_coord.y <= yMax));
}

void Tracker::getScaledPixelCoordinate(const Feature& feature,
                                       PixelCoor& scaled_coord) const
{
  // Transform the feature into scaled coordinates
  scaled_coord.x = (int) std::round(feature.getXDist() / pow(2, (double) feature.getPyramidLevel()));
  scaled_coord.y = (int) std::round(feature.getYDist() / pow(2, (double) feature.getPyramidLevel()));
}

bool Tracker::compareFASTFeaturesScore(const Feature& feature1,
                                       const Feature& feature2) const
{
  return feature1.getFastScore() > feature2.getFastScore();
}

void Tracker::appendNonNeighborFeatures(TiledImage& img,
                                        FeatureList& features,
                                        FeatureList& candidate_features,
                                        cv::Mat& mask)
{
  // Loop through candidate features.
  // If the patch is not blocked and there is room in the bucket, add it to features
  for(unsigned int i = 0; i < candidate_features.size(); i++)
  {
    double x = std::round(candidate_features[i].getXDist()); // approximate pixel coords (integer)
    double y = std::round(candidate_features[i].getYDist()); // approximate pixel coords (integer)

    // if the point is not blocked out
    if(mask.at<unsigned char>(y, x) == 0)
    {
      // if there is room in the bucket
      Feature& candidate_feature = candidate_features[i];
      img.setTileForFeature(candidate_feature);
      
	  // add it to the features and update the bucket count
      features.push_back(candidate_features[i]);

      //block out the surrounding area by setting the mask to 1
      cv::Mat blocked_box = mask(cv::Rect(x - block_half_length_, y - block_half_length_,
                                            2*block_half_length_ + 1, 2*block_half_length_ + 1));
      blocked_box.setTo(cv::Scalar(1));
    }
  }
}

void Tracker::removeOverflowFeatures(TiledImage& img1,
                                      TiledImage& img2,
                                      FeatureList& features1,
                                      FeatureList& features2) const
{
  img1.resetFeatureCounts();

  // Loop through current features to update their tile location 
  for(int i = features2.size(); i >= 1; i--)
  {
    img1.setTileForFeature(features1[i-1]);
    img2.setTileForFeature(features2[i-1]);
    
    img1.incrementFeatureCountAtTile(features1[i-1].getTileRow(), features1[i-1].getTileCol());
    img2.incrementFeatureCountAtTile(features2[i-1].getTileRow(), features2[i-1].getTileCol());
  }
  // Loop through current features to check if there are more than the max
  // allowed per tile and delete all features from this tile if so.
  // They are sorted by score so start at the bottom.
  for(int i = features2.size()-1; i >= 1; i--)
  {
    const unsigned int feat2_row = features2[i-1].getTileRow();
    const unsigned int feat2_col = features2[i-1].getTileCol();
    const unsigned int count = img2.getFeatureCountAtTile(feat2_row, feat2_col);
    if(count > img1.getMaxFeatPerTile())
    {
      //if the bucket is overflowing, remove the feature.
      features1.erase(features1.begin() + i-1);
      features2.erase(features2.begin() + i-1);
    }
  }
}

//this function automatically gets rid of points for which tracking fails
void Tracker::featureTracking(const cv::Mat& img1,
                              const cv::Mat& img2,
                              FeatureList& features1,
                              FeatureList& features2,
                              const double timestamp2,
                              const unsigned int frame_number2) const
{
  // Convert features to OpenCV keypoints
  //TODO(jeff) Store coordinates as cv::Point2f in Feature class to avoid
  // theses conversions
  std::vector<cv::Point2f> pts1;
  pts1.resize(features1.size());
  for(unsigned int i = 0; i < pts1.size(); i++)
  {
    pts1[i].x = features1[i].getXDist();
    pts1[i].y = features1[i].getYDist();
  }

  std::vector<uchar> status;
  std::vector<float> err;
  cv::Size win_size = cv::Size(31,31);
  cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

  // Prevents calcOpticalFlowPyrLK crash
  std::vector<cv::Point2f> pts2;
  if(pts1.size() > 0)
  {
    cv::calcOpticalFlowPyrLK(img1, img2,
                             pts1, pts2,
                             status, err, win_size, 2, term_crit,
                             cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 0.0); //0.001
  }

  // Output features which KLT tracked and stayed inside the frame
  int index_correction = 0;
  for(unsigned int i = 0; i < status.size(); i++)
  {
    const cv::Point2f pt = pts2.at(i);
    if(status.at(i) != 0
       && pt.x >= -0.5 && pt.y >= -0.5
       && pt.x <= img2.cols-0.5 && pt.y <= img2.rows-0.5)
    {
      Feature new_feature(timestamp2,
                          frame_number2,
                          pt.x,
                          pt.y,
                          features1[i].getPyramidLevel(),
                          features1[i].getFastScore());

      features2.push_back(new_feature);
    }
    else
    {
      features1.erase(features1.begin() + (i - index_correction));
      index_correction++;
    }
  }

  // Sanity check
  assert(features1.size() == features2.size());
}
