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

#ifndef JPL_VPP_TIMING_H
#define JPL_VPP_TIMING_H

#ifdef TIMING

#include <iostream>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <map>

#include <x/vision/types.h>

namespace x
{
// If not defined, timing output is only printed at program end.
#define TIMER_TO_COUT

struct TimeSum {
  using Dur = int;
  using Unit = std::chrono::microseconds;
  int runs;
  Dur total_time;
  TimeSum(Dur time) : runs(1), total_time(time) {}
  TimeSum(void) : runs(0), total_time(0) {}
};

static inline void FinishTimer(int signal);
class Timer {
public:
  using Id = std::string;
  using Time = std::chrono::time_point<std::chrono::system_clock>;
  using Dur = TimeSum::Dur;
  using Unit = TimeSum::Unit;
  static std::string DurAsString;
  static void Start(Id const id);
  static void Stop(Id const id);
  static void Finish(void);

  static inline Time Now() { return std::chrono::system_clock::now(); }

private:
  using ClockMap = std::map<Id, Time>;
  using Iterator = std::map<Id, Time>::const_iterator;
  using ClockSums = std::map<Id, TimeSum>;

  static Timer* timer_;
  ClockSums sums_;
  ClockMap clocks_;

#ifndef TIMER_TO_COUT
  std::stringstream stream_;
#endif

  /// Default Constructor.  This installs an anonymous signalhandler
  /// to be executed on ctrl-c-events and call the Timer::Finish()
  /// method.
  Timer(void) { signal(SIGINT, FinishTimer); }
  inline Iterator GetClock(Timer::Id id) {
    auto it = clocks_.find(id);
    return it;
  }

  inline void RemoveClock(Timer::Iterator it) { clocks_.erase(it); }

  inline Timer::Iterator AddClock(Id id) { clocks_.insert({id, Now()}); }

  inline bool Valid(Timer::Iterator it) const { return it != clocks_.end(); }
  Iterator GetClock(Id const id) const;
  void StopClock(Iterator start, Timer::Id id);

  void Sum(Timer::Id id, Timer::Dur secs);

  std::ostream& Stream(void)
  {
#ifdef TIMER_TO_COUT
    return std::cout;
#else
    return stream_;
#endif
  }
};

static inline void FinishTimer(int signal)
{
  (void)signal;
  Timer::Finish();
  exit(0);
}

#define TIME_ON(id)  Timer::Start(id);
#define TIME_OFF(id) Timer::Stop(id);
}
#else // #ifdef TIMING

namespace x 
{
inline static void nothing(void) {}
#define TIME_ON(id) nothing();
#define TIME_OFF(id) nothing();
}
#endif  // TIMING

#endif
