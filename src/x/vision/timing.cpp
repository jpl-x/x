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

#ifdef TIMING
#include <x/vision/timing.h>

using namespace x;

void operator+=(TimeSum& ts, TimeSum::Dur time) {
  ts.total_time += time;
  ts.runs++;
}

Timer* Timer::timer_ = nullptr;
std::string Timer::DurAsString = "us";

void Timer::Start(Timer::Id const id) {
  static Timer timer;      // singleton
  if (not timer.timer_) {  // make singleton available for Stop too
    Timer::timer_ = &timer;
  }
  if (timer.Valid(timer.GetClock(id))) {
    timer.Stream() << "Rejecting start of clock that's already running: " << id
                   << std::endl;
  } else {
    timer.AddClock(id);
    timer.Stream() << ">>>> Clock Start: " << id << std::endl;
  }
}

void Timer::Stop(Timer::Id const id) {
  if (not Timer::timer_) {
    std::cout << "Trying to stop a clock in uninitialized object: " << id
              << std::endl;
  } else {
    timer_->StopClock(timer_->GetClock(id), id);
  }
}

void Timer::StopClock(Timer::Iterator start, Timer::Id id) {
  if (not Valid(start)) {
    Stream() << "Trying to stop a clock that hasn't been started: " << id
             << std::endl;
  } else {
    auto dur = Timer::Now() - start->second;
    Dur elapsed = std::chrono::duration_cast<Timer::Unit> (dur).count();
    Stream() << "<<<< Clock Stop: " << start->first << ": " << elapsed << " "
             << Timer::DurAsString << std::endl;
    Sum(start->first, elapsed);
    // Sum(start->first, duration);
    RemoveClock(start);
  }
}

void Timer::Finish(void) {
  if (not Timer::timer_) return;
  for (auto clock = timer_->clocks_.begin(); clock != timer_->clocks_.end();
       ++clock) {
    Timer::timer_->StopClock(clock, clock->first);
  }

  Timer::timer_->Stream() << std::setw(65) << std::setfill('-') << "" << std::endl
                          << std::setfill(' ');
  Timer::timer_->Stream() << "Timing finished. Average times:" << std::endl;
  for (auto sum = timer_->sums_.begin(); sum != timer_->sums_.end(); ++sum) {
    Timer::timer_->Stream() << std::setw(30) << sum->first << ": ";
    Timer::timer_->Stream() << std::setw(15)
                            << std::to_string(sum->second.total_time /
                                              sum->second.runs) << " "
                            << Timer::DurAsString << " / "
                            << std::to_string(1.0 / (sum->second.total_time /
                                                     sum->second.runs * std::pow(10, -6))) << " Hz";
    Timer::timer_->Stream() << std::setw(20) << " ("
                            << std::to_string(sum->second.runs) << " executions)"
                            << std::endl;
  }
  Timer::timer_->Stream() << std::setw(65) << std::setfill('-') << "" << std::endl
                          << std::setfill(' ');

#ifndef TIMER_TO_COUT
  std::cout << timer_->stream_.str() << std::endl;
#endif
}

void Timer::Sum(Timer::Id id, Timer::Dur secs) {
  auto it = sums_.find(id);
  if (it == sums_.end()) {
    sums_.insert({id, secs});
  } else {
    it->second += secs;
  }
}

#endif  // TIMING
