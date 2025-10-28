cmake_minimum_required(VERSION 3.8)
project(visualization)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -pedantic)
  add_compile_options(-Werror)
  add_compile_options(-Werror=return-type)
  add_compile_options(-Werror=address)
  add_compile_options(-Werror=sequence-point)
  add_compile_options(-Werror=format)
  add_compile_options(-Werror=array-bounds)
  add_compile_options(-Werror=enum-compare)
  add_compile_options(-Werror=init-self)
  add_compile_options(-Werror=logical-op)
endif()

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};$ENV{LECTRON_DRIVERS};$ENV{LECTRON_LIBS};$ENV{MAVLINK_PATH}")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)

find_package(Eigen3 3.3 REQUIRED)

find_package(time_helper REQUIRED)
find_package(mathlib REQUIRED)
find_package(param_manager REQUIRED)
find_package(logger REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(SRC_FILE ${PROJECT_SOURCE_DIR}/src)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(visualization_node ${SRC_FILE}/visualization.cpp)

ament_target_dependencies(visualization_node
  rclcpp
  px4_msgs
  visualization_msgs
  tf2_ros
  nav_msgs
)

target_link_libraries(visualization_node
  time_helper 
  mathlib 
  param_manager
  logger
  Eigen3::Eigen 
)

install(TARGETS visualization_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
