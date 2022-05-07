cmake_minimum_required(VERSION 3.16)
project(x VERSION 1.1.0)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

#################################################################################
# User build settings

OPTION(MULTI_THREAD "Multi thread" ON) # Set true to process image and inertial data on different
		      # threads

option(VERBOSE "Publish std out and other data" ON) # Set false to disable all publishing and standard output
				  # stream, except pose at update rate. That will improve runtime.

option(TIMING "Publish std out and other data" OFF) # Set true to enable timers

option(PROFILING "Publish std out and other data" OFF) # Set true to disable compiler flags which are not
		      # compatible with Callgrind profiling tool.

option(UNIT_TESTS "Publish std out and other data" OFF) # Set true to enable unit tests

#################################################################################

# Set definitions
if(MULTI_THREAD)
  message("Multi-thread: ON")
  add_definitions(-DMULTI_THREAD)
endif()
if(VERBOSE)
  message("Verbose: ON")
  add_definitions(-DVERBOSE)
endif()
if(TIMING)
  message("Timing: ON")
  add_definitions(-DTIMING)
endif()
if(UNIT_TESTS)
  message("Unit Tests: ON")
  add_definitions(-DRUN_UNIT_TESTS)
endif()

add_definitions(-D_LINUX -D_REENTRANT)

# Look for OpenCV >= 3.3.1
find_package(OpenCV 4 QUIET)
if(NOT ${OpenCV_FOUND})
  find_package(OpenCV 3.3.1 QUIET)
  if(NOT ${OpenCV_FOUND})
    message(FATAL_ERROR "OpenCV >= 3.3.1 required.")
  endif()
endif()

# Eigen plugin
add_definitions(-DEIGEN_MATRIXBASE_PLUGIN=<x/common/eigen_matrix_base_plugin.h>)
find_package(Eigen3 REQUIRED)

# Set build flags, depending on the architecture

if (CMAKE_BUILD_TYPE MATCHES Release)
  message("Release Mode")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64") # tested on Jetson TX2
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+crypto -mcpu=cortex-a57+crypto -flto -ffast-math -fvect-cost-model=unlimited")
  #elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch32") # uncomment with correct check for Snapdragon Flight Pro
  #  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -flto -ffast-math -fvect-cost-model=unlimited")
  endif()
  
  if (${PROFILING} MATCHES false)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-loop-optimizations -fsee -funroll-loops -fno-math-errno -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros")
  endif()
  
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
  message("Debug Mode")
  add_definitions(-DDEBUG -DDEBUGMSF)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("Release with Debug Info Mode")
  # Enable asserts
  add_definitions(-UNDEBUG)
endif()

set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
set(EIGEN3_LIBRARIES ${EIGEN3_LIBRARIES})

include_directories (include)

include_directories (SYSTEM
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

set (SOURCE
  src/x/ekf/ekf.cpp
  src/x/ekf/propagator.cpp
  src/x/ekf/state.cpp
  src/x/ekf/state_buffer.cpp
  src/x/ekf/updater.cpp
  src/x/vio/vio.cpp
  src/x/vio/vio_updater.cpp
  src/x/vio/state_manager.cpp
  src/x/vio/track_manager.cpp
  src/x/vio/msckf_update.cpp
  src/x/vio/msckf_slam_update.cpp
  src/x/vio/slam_update.cpp
  src/x/vio/range_update.cpp
  src/x/vio/solar_update.cpp
  src/x/vision/camera.cpp
  src/x/vision/feature.cpp
  src/x/vision/tiled_image.cpp
  src/x/vision/timing.cpp
  src/x/vision/tracker.cpp
  src/x/vision/triangulation.cpp
)

add_library(x ${SOURCE})

# Additional libraries to link against
target_link_libraries(x
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

# Uninstall Target 
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/xUninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/xUninstall.cmake" IMMEDIATE @ONLY)
add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/xUninstall.cmake")

set(CMAKE_INSTALL_LIBDIR lib)

# Generate pkg-config file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.pc.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Cmake find_package() support.
set(CMAKE_EXPORT_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
include(CMakePackageConfigHelpers)

install(TARGETS x EXPORT xTargets)
install(EXPORT xTargets
    DESTINATION ${CMAKE_EXPORT_DESTINATION}
)
install(DIRECTORY include/ DESTINATION include)
configure_package_config_file(xConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/xConfig.cmake
    INSTALL_DESTINATION ${CMAKE_EXPORT_DESTINATION}
)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/xConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/xConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/xConfigVersion.cmake
    DESTINATION ${CMAKE_EXPORT_DESTINATION}
)

# Support automatic RPM/DEB generation via CPack
SET(CPACK_CMAKE_GENERATOR ${CMAKE_GENERATOR})
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "polivicio@gmail.com")
set(CPACK_PACKAGE_VENDOR "JPL")
set(CPACK_PACKAGE_DESCRIPTION "Generic C++ library for vision-based navigation, with multi-sensor fusion capabilities for thermal, range, solar and GPS measurements.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${CPACK_PACKAGE_DESCRIPTION})
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}.${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_GENERATOR "RPM" "DEB")
set(CPACK_RPM_PACKAGE_AUTOREQ 1)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS 1)

include(CPack)
