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

#include <x/vision/feature.h>

using namespace x;

Feature::Feature()
{}

Feature::Feature(double timestamp,
                 double x,
                 double y)
: timestamp_(timestamp)
, x_(x)
, y_(y)
{}

Feature::Feature(double timestamp,
                 unsigned int frame_number,
                 double x,
                 double y,
                 double x_dist,
                 double y_dist)
: timestamp_(timestamp)
, frame_number_(frame_number)
, x_(x)
, y_(y)
, x_dist_(x_dist)
, y_dist_(y_dist)
{}

Feature::Feature(double timestamp,
                 unsigned int frame_number,
                 double x_dist,
                 double y_dist,
                 unsigned int pyramid_level,
                 float fast_score)
: timestamp_(timestamp)
, frame_number_(frame_number)
, x_dist_(x_dist)
, y_dist_(y_dist)
, pyramid_level_(pyramid_level)
, fast_score_(fast_score)
{}
