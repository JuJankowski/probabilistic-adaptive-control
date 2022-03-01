cmake_minimum_required(VERSION 3.4)
project(probabilistic_trajectory_tracking)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  visualization_msgs
  tf
  message_generation
)

find_package(Eigen3 REQUIRED)

add_message_files(
  FILES
  Task.msg
  Context.msg
  ContextArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES probabilistic_trajectory_tracking
  CATKIN_DEPENDS
    roscpp
    std_msgs
    geometry_msgs
    message_runtime
)

add_library(probabilistic_trajectory_tracking
  src/probabilistic_adaptive_control.cpp
  src/probabilistic_trajectory_tracking.cpp
  src/reference_handler.cpp
)

target_include_directories(probabilistic_trajectory_tracking SYSTEM PUBLIC
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  include
)

add_executable(pac_node
  src/pac_node.cpp
)

target_link_libraries(pac_node
 ${catkin_LIBRARIES} 
 probabilistic_trajectory_tracking
)

add_dependencies(probabilistic_trajectory_tracking ${${PROJECT_NAME}_EXPORTED_TARGETS})

## Installation
install(TARGETS probabilistic_trajectory_tracking
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(probabilistic_trajectory_tracking FILES ${SOURCES} ${HEADERS})
  add_tidy_target(probabilistic_trajectory_tracking
    FILES ${SOURCES}
    DEPENDS probabilistic_trajectory_tracking
  )
endif()
