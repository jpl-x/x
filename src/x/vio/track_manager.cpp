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

#include <x/vio/track_manager.h>
#include <x/vio/tools.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace x;

TrackManager::TrackManager() {}

TrackManager::TrackManager(const Camera& camera, const double min_baseline_n)
: camera_(camera)
, min_baseline_n_(min_baseline_n)
{}

void TrackManager::setCamera(Camera camera) {
  camera_ = camera;
}

TrackList TrackManager::normalizeSlamTracks(const int size_out) const {
  return camera_.normalize(slam_trks_, size_out);
}

TrackList TrackManager::getMsckfTracks() const {
  return msckf_trks_n_;
}

TrackList TrackManager::getNewSlamStdTracks() const {
  return new_slam_std_trks_n_;
}

TrackList TrackManager::getNewSlamMsckfTracks() const {
  return new_slam_msckf_trks_n_;
}

void TrackManager::clear() {
  slam_trks_.clear();
  new_slam_trks_.clear();
  lost_slam_idxs_.clear();
  opp_trks_.clear();
  msckf_trks_n_.clear();
}

void TrackManager::removePersistentTracksAtIndex(const unsigned int idx) {
  slam_trks_.erase(slam_trks_.begin() + idx);
}

void TrackManager::removeNewPersistentTracksAtIndexes(
    const std::vector<unsigned int> invalid_tracks_idx) {
  // Note: Assumes indexes are in increasing order

  // For each feature to remove
  size_t i = invalid_tracks_idx.size();
  while(i)
  {
    new_slam_trks_.erase(new_slam_trks_.begin() + invalid_tracks_idx[i-1]);
    i--;
  }
}

std::vector<unsigned int> TrackManager::getLostSlamTrackIndexes() const {
  return lost_slam_idxs_;
}

void TrackManager::manageTracks(MatchList& matches,
                                const AttitudeList cam_rots,
                                const int n_poses_max,
                                const int n_slam_features_max,
                                const int min_track_length,
                                TiledImage& img) {
  // Append the new persistent tracks from last image to the persistent track
  // list and clear the list for new ones
  slam_trks_.insert(slam_trks_.end(),
                   new_slam_trks_.begin(),
                   new_slam_trks_.end());
  new_slam_trks_.clear();
  
  // Array storing the new and current persistent track indexes per bin in
  // preparation to spread them out through all the bins. The indexes assume
  // the new persistent track list is appended after the persistent track one.
  const unsigned int n_bins = img.getNTilesH() * img.getNTilesW();
  std::vector<unsigned int> bin_track_idx[n_bins];

  // Similar array storing only current persistent tracks per bin, but not
  // updating the indexes after deletion so those can be used to remove the
  // features from the state later on.
  std::vector<unsigned int> bin_track_idx_per[n_bins];

  // Index of bin with max number of features
  unsigned int idx_bin_max_per = 0;

  // Loop through all persistent tracks to determine if they are still active.
  unsigned int t = 0;
  unsigned int lost_per_count = 0;
  lost_slam_idxs_.clear();
  while (t < slam_trks_.size())
  {
    bool isTrackLost = true;
    unsigned int m = 0;
    while (m < matches.size())
    {
      // Check if the last feature of the persistent track is the previous match
      if (slam_trks_[t].back().getX() == matches[m].previous.getX()
       && slam_trks_[t].back().getY() == matches[m].previous.getY())
      {
        isTrackLost = false; // we found the track, so it is not lost

        // Find and store the current bin for this feature
        Feature& feature = matches[m].current;
        img.setTileForFeature(feature);
        const unsigned int bin_nbr =
            matches[m].current.getTileRow() * img.getNTilesW() + matches[m].current.getTileCol();
        bin_track_idx[bin_nbr].push_back(t);
        bin_track_idx_per[bin_nbr].push_back(t + lost_per_count);

        // Update index of bin with max number of features if necessary
        if (bin_track_idx[bin_nbr].size() >
        bin_track_idx[idx_bin_max_per].size())
        {
          idx_bin_max_per = bin_nbr;
        }

        // Add the matched feature to the track
        slam_trks_[t].push_back(matches[m].current);
        matches.erase(matches.begin() + m); // erase it from the match list

        break; // leave the while loop on the matches
      }
      m++; // increment the match iterator
    }

      if(isTrackLost)
    {
      // Add its index to lost tracks so it is deleted from the filter state and
      // covariance matrix
      lost_slam_idxs_.push_back(t + lost_per_count);

      // Erase the persistence feature track
      slam_trks_.erase(slam_trks_.begin() + t);

      // Increment counter
      lost_per_count++;

      continue; // continue to avoid incrementing track iterator t
    }
    t++; // increment the track iterator
  }

  // Loop through all remaining matches:
  // 1/ Extend current opportunistic tracks
  // 2/ Convert full MSCKF tracks to persistent tracks if possible
  // 3/ Fill up open persistent states with remaining matches
  msckf_trks_n_.clear();
  TrackList previous_opp_trks = opp_trks_;
  opp_trks_.clear();
  unsigned int m = 0;
  while (m < matches.size()) // Matches are in descending FAST order
  {
    t = 0;
    bool isTrackLost = true;
    while (t < previous_opp_trks.size())
    {
      // Check if the last feature of the opportunistic track is the previous match
      if (previous_opp_trks[t].back().getX() == matches[m].previous.getX()
       && previous_opp_trks[t].back().getY() == matches[m].previous.getY())
      {
        isTrackLost = false; // we found the track, so it is not lost
        // Add the matched feature to the track
        previous_opp_trks[t].push_back(matches[m].current);
        break; // leave the while loop on the opportunistic tracks
      }
      t++; // increment the opportunistic track iterator
    }

    if(!isTrackLost) // if the track is not lost
    {
      // Push it to the opportunistic tracks
      opp_trks_.push_back(previous_opp_trks[t]);
      // Remove from the previous opportunistic list
      previous_opp_trks.erase(previous_opp_trks.begin() + t);
    }
    else // New opportunistic track!
    {
      Track opp_track;
      opp_track.push_back(matches[m].previous);
      opp_track.push_back(matches[m].current);
      opp_trks_.push_back(opp_track);
    }
    m++;
  }
      
  // Sort remaining tracks by length, longest to shortest
  std::sort(opp_trks_.begin(), opp_trks_.end(), [](const Track & a, const Track & b){ return a.size() > b.size(); });
  t = 0;
  while (t < opp_trks_.size()) // Tracks are sorted by length, longest to shortest
  { 
    // Find and store current bin for this feature
    Feature& feature = opp_trks_[t].back(); 
    img.setTileForFeature(feature);
    const unsigned int bin_nbr =
      feature.getTileRow() * img.getNTilesW() + feature.getTileCol();

    // If track is long enough (quality metric)
    if (opp_trks_[t].size() > min_track_length - 1)
    {
      // If persistent features spots are available
      if((slam_trks_.size() + new_slam_trks_.size()) < n_slam_features_max)
      {
        // Push the track to the persistent list
        new_slam_trks_.push_back(opp_trks_[t]);
        opp_trks_.erase(opp_trks_.begin() + t);
        bin_track_idx[bin_nbr].push_back(slam_trks_.size() + new_slam_trks_.size() - 1);

        // Update index of bin with max number of features if necessary
        if (bin_track_idx[bin_nbr].size() >
        bin_track_idx[idx_bin_max_per].size())
        {
          idx_bin_max_per = bin_nbr;
        }
      }
      else // determine if another persistent feature should be removed to
        // spread out their distribution
      {
        // If there is a difference of more than one feature between the
        // count of the current bin and the one with the maximum count
        if (bin_track_idx[idx_bin_max_per].size() >
        bin_track_idx[bin_nbr].size() + 1)
        {
          // Remove youngest track from the max bin
          // If it is a new persistent feature
          if (bin_track_idx[idx_bin_max_per].back() >= slam_trks_.size())
          {
            new_slam_trks_.erase(new_slam_trks_.begin() + bin_track_idx[idx_bin_max_per].back() - slam_trks_.size());
          }
          else
          {
            //TODO(jeff): This persistent track should still be processed and be removed after
            // the update
            lost_slam_idxs_.push_back(bin_track_idx_per[idx_bin_max_per].back());
            bin_track_idx_per[idx_bin_max_per].pop_back();
            slam_trks_.erase(slam_trks_.begin() + bin_track_idx[idx_bin_max_per].back());
          }

          // Adjust indexes
          for (unsigned int i = 0; i < n_bins; i++)
          {
            for (unsigned int j = 0; j < bin_track_idx[i].size(); j++)
            {
              if (bin_track_idx[i][j] > bin_track_idx[idx_bin_max_per].back())
              {
                bin_track_idx[i][j]--;
              }
            }
          }

          // Remove index
          bin_track_idx[idx_bin_max_per].pop_back();
          
	      // Add current track
          new_slam_trks_.push_back(opp_trks_[t]);
          opp_trks_.erase(opp_trks_.begin() + t);
          bin_track_idx[bin_nbr].push_back(slam_trks_.size() + new_slam_trks_.size() - 1);

          // Update index of bin with max number of features
          for (unsigned int i = 0; i < n_bins; i++)
          {
            if (bin_track_idx[i].size() >
                bin_track_idx[idx_bin_max_per].size())
            {
              idx_bin_max_per = i;
            }
          }
        }
	      else
        {
          // If the track is too long
          if (opp_trks_[t].size() > n_poses_max - 1)
          {
            // Normalize track (and crop it if it longer than the attitude list)
            Track normalized_track = camera_.normalize(opp_trks_[t], cam_rots.size());
            if(checkBaseline(normalized_track, cam_rots))
              msckf_trks_n_.push_back(normalized_track);

            opp_trks_.erase(opp_trks_.begin() + t); 
          }
          else
     	      t++; // increment the match iterator
        }
      }
    }
    else
      t++; // increment the match iterator
  }

  
  // Sort new SLAM tracks:
  // - semi-infinite depth uncertainty (standard SLAM) or,
  // - MSCKF-SLAM
  new_slam_std_trks_n_.clear(); // clean last image's data
  new_slam_msckf_trks_n_.clear();

  TrackList new_slam_std_trks, new_slam_msckf_trks; // see use below

  for(size_t i=0; i<new_slam_trks_.size(); i++)
  {
    // Normalize coordinates
    const Track trk = new_slam_trks_[i];
    // Track cannot be longer than the attitude list
    const Track normalized_trk = camera_.normalize(trk, cam_rots.size());

    //Check baseline and sort
    if(checkBaseline(normalized_trk, cam_rots))
    {
      new_slam_msckf_trks_n_.push_back(normalized_trk);
      new_slam_msckf_trks.push_back(trk);
    }
    else
    {
      new_slam_std_trks_n_.push_back(normalized_trk);
      new_slam_std_trks.push_back(trk);
    }
  }

  // Reorder new SLAM tracklist with MSCKF-SLAM first so it matches
  // the order in which the states will be inserted in Update
  // TODO(jeff): Unsafe. Assumes MSCKF-SLAM happens before standard
  // SLAM in Update.
  new_slam_trks_.clear();
  new_slam_trks_.insert( new_slam_trks_.end(),
                         new_slam_msckf_trks.begin(),
                         new_slam_msckf_trks.end() );
  new_slam_trks_.insert( new_slam_trks_.end(),
                         new_slam_std_trks.begin(),
                         new_slam_std_trks.end() );

  // Debug image
  plotFeatures(img, min_track_length);
}



/** \brief Returns a triangle of SLAM features around input image coordinates.
 *  \param lrf_img_pt Image coordinates of the LRF impact point
 *  \param img Image plot
 *  \return A vector with the ID of the SLAM features in the triangle
 */
std::vector<int> 
TrackManager::featureTriangleAtPoint(const Feature& lrf_img_pt,
                                     TiledImage& img) const {
  /*******************************************
   * Delaunay triangulation on SLAM features *
   *******************************************/

  // Rectangle to be used with Subdiv2D (account +/- 0.5 offset on coordinates at edges)
  const cv::Size size = img.size();
  const cv::Rect rect(-1, -1, size.width+2, size.height+2);

  // Create an instance of Subdiv2D
  cv::Subdiv2D subdiv(rect);
  // Create a vector of point to store SLAM features
  std::vector<cv::Point2d> points;

  // Add SLAM feature to Delaune triangulation
  for (unsigned int i = 0; i < slam_trks_.size(); i++)
  {
    const Feature feature(slam_trks_[i].back());
    points.push_back(cv::Point2d(feature.getXDist(),feature.getYDist()));
    cv::Point2d pt(feature.getXDist(),feature.getYDist());
	subdiv.insert(pt);
  }

  // Define colors for drawing.
  const cv::Scalar delaunay_color(255,255,255);

  // Draw delaunay triangles
  draw_delaunay( img, subdiv, delaunay_color );
  
  // Define color for drawing the LRF triangle
  const cv::Scalar delaunay_color_selected(0,0,255);

  // Find triangle edge to the right of the LRF image point
  int edge, vertex;
  cv::Point2d lrf_cv_pt(lrf_img_pt.getXDist(), lrf_img_pt.getYDist());
  const int status = subdiv.locate(lrf_cv_pt, edge, vertex); 
  
  //Plot LRF image point
  Feature lrf_img_feature;
  lrf_img_feature.setXDist(lrf_cv_pt.x);
  lrf_img_feature.setYDist(lrf_cv_pt.y); 
  img.plotFeature(lrf_img_feature, delaunay_color_selected);

  /******************************************
   * Find SLAM feature IDs for the triangle *
   ******************************************/
  //Note: The first 4 points of the subdivision correspond to the edges of the image. If they are one of the triangle's
  // vertex, that triangle is not valid.

  std::vector<int> tr_feat_ids = {-1,-1,-1};
  
  // If lrf falls inside or on the edge of a triangle
  if(status == cv::Subdiv2D::PTLOC_INSIDE || status == cv::Subdiv2D::PTLOC_ON_EDGE)
  {
    // Origin vertex of the edge
    tr_feat_ids[0] = subdiv.edgeOrg(edge) - 4;
    // Destination vertex of the edge 
    tr_feat_ids[1] = subdiv.edgeDst(edge) - 4;
    // Destination vertex of the next edge => Triangle is complete
    tr_feat_ids[2] = subdiv.edgeDst(subdiv.nextEdge(edge)) - 4;
  }
  // else if it falls on one of the vertices
  else if(status == cv::Subdiv2D::PTLOC_VERTEX)
  {
    // LRF vertex
    tr_feat_ids[0] = vertex - 4;
    // Destination vertex of the first edge in the list of the current vertex
    subdiv.getVertex(vertex, &edge);
    tr_feat_ids[1] = subdiv.edgeDst(edge) - 4;
    // Destination vertex of the next edge => Triangle is complete
    tr_feat_ids[2] = subdiv.edgeDst(subdiv.nextEdge(edge)) - 4;
  }
  // else there was an input error
  else
  {
    tr_feat_ids.clear();
    return tr_feat_ids;
  }
  
  // Plot selected triangle
  subdiv = cv::Subdiv2D(rect);

  for (unsigned int i = 0; i < 3; i++)
  {
    // The first 4 points of the subdivision correspond to the edges of the image. If they are one of the triangle's
    // vertex, that triangle is not valid.
    if( tr_feat_ids[i] > -1)
    {
      const Feature feature(slam_trks_[tr_feat_ids[i]].back());
      points.push_back(cv::Point2d(feature.getXDist(),feature.getYDist()));
      subdiv.insert(cv::Point2d(feature.getXDist(),feature.getYDist()));
    }
    else // invalid triangle 
    {
      tr_feat_ids.clear();
      return tr_feat_ids;
    }
  }

  // Plot triangle
  draw_delaunay( img, subdiv, delaunay_color_selected );

  return tr_feat_ids;
} 
   
// Draw delaunay triangles
void TrackManager::draw_delaunay(cv::Mat& img,
                                 cv::Subdiv2D& subdiv,
                                 cv::Scalar delaunay_color ) const {
  std::vector<cv::Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  std::vector<cv::Point> pt(3);
  cv::Size size = img.size();
  cv::Rect rect(0,0, size.width, size.height);

  for( size_t i = 0; i < triangleList.size(); i++ )
  {
    cv::Vec6f t = triangleList[i];
    pt[0] = cv::Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = cv::Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = cv::Point(cvRound(t[4]), cvRound(t[5]));

    // Draw rectangles completely inside the image.
    if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
    {
#if CV_MAJOR_VERSION == 4
        line(img, pt[0], pt[1], delaunay_color, 1, cv::LINE_AA, 0);
        line(img, pt[1], pt[2], delaunay_color, 1, cv::LINE_AA, 0);
        line(img, pt[2], pt[0], delaunay_color, 1, cv::LINE_AA, 0);
#else
        line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
        line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
        line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
#endif
    }
  }
}

bool TrackManager::checkBaseline(const Track& track,
                                 const AttitudeList& C_q_G) const {
  const int n_quats = C_q_G.size();
  const int n_obs = track.size();
  
  // Index of the first and last observation of that feature in the
  // pose vector
  const int i_last = n_quats - 1;
  const int i_first = i_last - n_obs + 1;

  // There must be at least 2 observations in the track to check the
  // baseline AND the attitude list size must be greater or equal
  // than the track's
  assert(n_obs > 1 && n_quats >= n_obs);

  // Init
  double min_feat_x = track[n_obs - 1].getX();
  double max_feat_x = track[n_obs - 1].getX();
  double min_feat_y = track[n_obs - 1].getY();
  double max_feat_y = track[n_obs - 1].getY();
  Quatern attitude_to_quaternion;
  const Eigen::Quaterniond Cn_q_G = attitude_to_quaternion(C_q_G[i_last]);

  // For each observation, going backwards in time
  // TODO(jeff): Make consistent wrt linear triangulation, e.g.
  // for (int i = 0; i > -1; --i)
  for (int i = i_first; i <= i_last; ++i)
  {
    // Rotate 3d ray coordinates in last camera frame
    const Eigen::Quaterniond Ci_q_G = attitude_to_quaternion(C_q_G[i]);
    const int i_trk = i - i_first;
    const Eigen::Quaterniond Cn_q_Ci = Ci_q_G.normalized().conjugate() * Cn_q_G.normalized();
    const Eigen::Quaterniond q_ray_Ci(0.0, track[i_trk].getX(), track[i_trk].getY(), 1.0);
    Eigen::Quaterniond q_ray_Cn = Cn_q_Ci.conjugate() * q_ray_Ci * Cn_q_Ci;

    // Normalized coordinates
    const double feat_x = q_ray_Cn.x() / q_ray_Cn.z();
    const double feat_y = q_ray_Cn.y() / q_ray_Cn.z();

    // Update min/max in X dimension
    if (feat_x < min_feat_x)
      min_feat_x = feat_x;
    else if (feat_x > max_feat_x)
      max_feat_x = feat_x;

    // Update min/max in Y dimension
    if (feat_y < min_feat_y)
      min_feat_y = feat_y;
    else if (feat_y > max_feat_y)
      max_feat_y = feat_y;
  }

  const double delta_feat_x = max_feat_x - min_feat_x;
  const double delta_feat_y = max_feat_y - min_feat_y;

  if (delta_feat_x > min_baseline_n_ || delta_feat_y > min_baseline_n_)
    return true;
  else
    return false;
}

void TrackManager::plotFeatures(TiledImage& img,
                                const int min_track_length) {
  // Convert image in a color image
#if CV_MAJOR_VERSION == 4
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
#else
    cv::cvtColor(img, img, CV_GRAY2BGR);
#endif

  // Plot tiles
  img.plotTiles();

  cv::Scalar green(100,255,100);
  cv::Scalar orange(0,204,249);
  cv::Scalar blue(249,144,24);

  // Draw SLAM features
  const size_t n_slam = slam_trks_.size();
  for (size_t ii = 0; ii < n_slam; ii++)
    img.plotFeature(slam_trks_[ii][slam_trks_[ii].size() - 1], green);
  const size_t n_new_slam = new_slam_trks_.size();
  for (size_t ii = 0; ii < n_new_slam; ii++)
    img.plotFeature(new_slam_trks_[ii][new_slam_trks_[ii].size() - 1], green);

  // Draw opportunistic features
  const size_t n_opp = opp_trks_.size();
  unsigned int count_pot = 0, count_opp = 0;
  for (size_t ii = 0; ii < n_opp; ii++)
  {
    // Use track length to differentiate potential tracks, which are not long
    // enough to be processed as an MSCKF measurement if lost.
    const size_t track_sz = opp_trks_[ii].size();

    if( track_sz >= min_track_length - 1)
    {
      count_opp++;
      img.plotFeature(opp_trks_[ii].back(), orange);
    }
    else
    {
      count_pot++;
      img.plotFeature(opp_trks_[ii].back(), blue);
    }
  }

  std::string slam_str     = std::string("SLAM: ") +
      std::to_string(n_slam + n_new_slam) +
      std::string(" - ");
  std::string oppStr     = std::string("Opp: ") +
      std::to_string(count_opp) +
      std::string(" - MSCKF: ") +
      std::to_string(msckf_trks_n_.size());
  std::string potStr = std::string(" - Pot: ") +
      std::to_string(count_pot);

  cv::putText(img, slam_str, cv::Point((int) 10, (int) camera_.getHeight()-10), cv::FONT_HERSHEY_PLAIN,
              1.0, green, 1.5, 8, false);

  int baseline = 0;
  cv::Size textSize = cv::getTextSize(slam_str, cv::FONT_HERSHEY_PLAIN, 1.0, 1.5, &baseline);
  int offset = textSize.width;
  cv::putText(img, oppStr, cv::Point((int) 10 + offset, (int) camera_.getHeight()-10), cv::FONT_HERSHEY_PLAIN,
              1.0, orange, 1.5, 8, false);
  textSize = cv::getTextSize(oppStr, cv::FONT_HERSHEY_PLAIN, 1.0, 1.5, &baseline);
  offset += textSize.width;
  cv::putText(img, potStr, cv::Point((int) 10 + offset, (int) camera_.getHeight()-10), cv::FONT_HERSHEY_PLAIN,
              1.0, blue, 1.5, 8, false);
}
