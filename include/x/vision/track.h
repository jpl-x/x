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

#ifndef X_VISION_TRACK_H_
#define X_VISION_TRACK_H_

#include <x/vision/feature.h>
#include <vector>

namespace x
{

  /**
   * A feature track class.
   *
   * This class is a vector of features (i.e. 2D image coordinates) ordered
   * chronologically.
   */
  class Track : public std::vector<Feature>
  {
    public:
      /**
       * Default constructor.
       */
      Track()
        : std::vector<Feature>()
      {};

      /**
       * Constructs a Track object of a given size, filled a with a given
       * feature object.
       *
       * @param[in] count Track size.
       * @param[in] feature Feature to initialize each track entry at.
       */
      Track(const size_type count, const Feature& feature)
        : std::vector<Feature>(count, feature)
      {};
  };
} // namespace x

#endif  // X_VISION_TRACK_H_
