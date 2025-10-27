#pragma once
#include <deque>
#include <limits>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include "lectron/mathlib/transformation.hpp"
#include "lectron/time_helper/time_count.hpp"
#include "lectron/logger/logger.hpp"

using navPathMsg          = nav_msgs::msg::Path;
using odometryMsg         = nav_msgs::msg::Odometry;
using vehicleOdomMsg      = px4_msgs::msg::VehicleOdometry;
using poseStampedMsg      = geometry_msgs::msg::PoseStamped;
using transformStampedMsg = geometry_msgs::msg::TransformStamped;

struct Publishers {
    rclcpp::Publisher<navPathMsg>::SharedPtr px4_path;
    rclcpp::Publisher<navPathMsg>::SharedPtr vio_path;
};

struct Subscribers {
    rclcpp::Subscription<odometryMsg>::SharedPtr     vio_odom;
    rclcpp::Subscription<vehicleOdomMsg>::SharedPtr  px4_odom;
};

struct CallbackTimers {
    rclcpp::TimerBase::SharedPtr command;
};

struct State {
    float position[3]{0.f, 0.f, 0.f};
    float q[4]{1.f, 0.f, 0.f, 0.f};
    uint64_t timestamp{0};
    int pose_frame{0};
    static constexpr int POSE_FRAME_UNKNOWN = 0;
    static constexpr int POSE_FRAME_NED = 1;
    static constexpr int POSE_FRAME_FRD = 2;
    static constexpr int POSE_FRAME_ENU = 3;
};

class Visualization : public rclcpp::Node {
public:
    Visualization();
    ~Visualization() = default;
    void vehicleOdomCallback(const vehicleOdomMsg::UniquePtr msg);
    void vioOdomImuCallback(const odometryMsg::UniquePtr msg);
    void commandCallback();

private:
    static Eigen::Quaterniond quatFromArrayNormalize(const float q[4]);
    void computeENU(const State &s, Eigen::Vector3d &p_enu, Eigen::Quaterniond &q_enu) const;
    void setupTF(transformStampedMsg &t, const Eigen::Vector3d &p_enu, const Eigen::Quaterniond &q_enu,
                 const std::string &parent_id, const std::string &child_id);
    bool addPoseIfNew(const Eigen::Vector3d &p_enu, navPathMsg &path, Eigen::Vector3d &prev_pose);
    bool isDifferenceInOffset(double a, double b, double off) const;

private:
    CallbackTimers _timer;
    Publishers _pub;
    Subscribers _sub;
    navPathMsg _px4_path;
    navPathMsg _vio_path;
    State _vio_state;
    State _px4_state;
    size_t _pose_size{5000};
    bool _show_vio{true};
    Eigen::Vector3d _prev_pose_px4;
    Eigen::Vector3d _prev_pose_vio;
    bool _traj_have_px4{false};
    bool _traj_have_vio{false};
    Eigen::Vector3d _traj_last_px4;
    Eigen::Vector3d _traj_last_vio;
    double _traj_dist_px4{0.0};
    double _traj_dist_vio{0.0};
    uint64_t _last_report_ms{0};
    std::unique_ptr<tf2_ros::TransformBroadcaster> _tf_vio;
    std::unique_ptr<tf2_ros::TransformBroadcaster> _tf_px4;
};
