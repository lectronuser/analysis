#include "visualization.hpp"
using namespace std::placeholders;

Eigen::Quaterniond Visualization::quatFromArrayNormalize(const float q[4])
{
    Eigen::Quaterniond qq(q[0], q[1], q[2], q[3]);
    if (qq.squaredNorm() == 0.0)
        return Eigen::Quaterniond::Identity();
    qq.normalize();
    return qq;
}

void Visualization::computeENU(const State &s, Eigen::Vector3d &p_enu, Eigen::Quaterniond &q_enu) const
{
    static const Eigen::Matrix3d R_ned2enu = (Eigen::Matrix3d() << 0, 1, 0,
                                              1, 0, 0,
                                              0, 0, -1)
                                                 .finished();
    static const Eigen::Matrix3d R_frd2flu = (Eigen::Matrix3d() << 1, 0, 0,
                                              0, -1, 0,
                                              0, 0, -1)
                                                 .finished();

    Eigen::Quaterniond q = quatFromArrayNormalize(s.q);

    if (s.pose_frame == State::POSE_FRAME_ENU)
    {
        p_enu = Eigen::Vector3d(s.position[0], s.position[1], s.position[2]);
        q_enu = q;
    }
    else
    {
        p_enu = Eigen::Vector3d(s.position[1], s.position[0], -s.position[2]);
        q_enu = Eigen::Quaterniond(R_ned2enu) * q * Eigen::Quaterniond(R_frd2flu);
        q_enu.normalize();
    }
}

void Visualization::setupTF(transformStampedMsg &t,
                            const Eigen::Vector3d &p_enu,
                            const Eigen::Quaterniond &q_enu,
                            const std::string &parent_id,
                            const std::string &child_id)
{
    t.header.frame_id = parent_id;
    t.child_frame_id = child_id;
    t.transform.translation.x = p_enu.x();
    t.transform.translation.y = p_enu.y();
    t.transform.translation.z = p_enu.z();
    t.transform.rotation.w = q_enu.w();
    t.transform.rotation.x = q_enu.x();
    t.transform.rotation.y = q_enu.y();
    t.transform.rotation.z = q_enu.z();
}

bool Visualization::isDifferenceInOffset(double a, double b, double off) const
{
    return std::abs(a - b) > off;
}

bool Visualization::addPoseIfNew(const Eigen::Vector3d &p_enu,
                                 navPathMsg &path,
                                 Eigen::Vector3d &prev_pose)
{
    bool is_new = false;
    if (!std::isfinite(prev_pose[0]))
    {
        is_new = true;
    }
    else
    {
        const bool dx = isDifferenceInOffset(p_enu.x(), prev_pose.x(), 0.1);
        const bool dy = isDifferenceInOffset(p_enu.y(), prev_pose.y(), 0.1);
        const bool dz = isDifferenceInOffset(p_enu.z(), prev_pose.z(), 0.1);
        is_new = (dx || dy || dz);
    }

    if (is_new)
    {
        poseStampedMsg pose_stamped;
        pose_stamped.header.stamp = this->get_clock()->now();
        pose_stamped.header.frame_id = path.header.frame_id;
        pose_stamped.pose.position.x = p_enu.x();
        pose_stamped.pose.position.y = p_enu.y();
        pose_stamped.pose.position.z = p_enu.z();
        path.poses.push_back(pose_stamped);
        if (path.poses.size() > _pose_size)
            path.poses.erase(path.poses.begin());
        prev_pose = p_enu;
        return true;
    }
    return false;
}

Visualization::Visualization()
    : Node("visualization_node"),
      _tf_vio(std::make_unique<tf2_ros::TransformBroadcaster>(this)),
      _tf_px4(std::make_unique<tf2_ros::TransformBroadcaster>(this))
{
    _show_vio = this->declare_parameter<bool>("show_vio", true);
    _prev_pose_px4.setConstant(std::numeric_limits<double>::quiet_NaN());
    _prev_pose_vio.setConstant(std::numeric_limits<double>::quiet_NaN());

    _timer.command = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&Visualization::commandCallback, this));

    _pub.px4_path = this->create_publisher<navPathMsg>("/px4_path", 100);
    _pub.vio_path = this->create_publisher<navPathMsg>("/vio_path", 100);

    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 5), qos_profile);

    _sub.vio_odom = this->create_subscription<odometryMsg>(
        "/ov_msckf/odomimu", 10,
        std::bind(&Visualization::vioOdomImuCallback, this, _1));

    _sub.px4_odom = this->create_subscription<vehicleOdomMsg>(
        "/fmu/out/vehicle_odometry", qos,
        std::bind(&Visualization::vehicleOdomCallback, this, _1));

    logInfo("Visualization started. show_vio=%s", _show_vio ? "true" : "false");
}

void Visualization::vehicleOdomCallback(const vehicleOdomMsg::UniquePtr msg)
{
    _px4_state.position[0] = msg->position[0];
    _px4_state.position[1] = msg->position[1];
    _px4_state.position[2] = msg->position[2];
    _px4_state.q[0] = msg->q[0];
    _px4_state.q[1] = msg->q[1];
    _px4_state.q[2] = msg->q[2];
    _px4_state.q[3] = msg->q[3];
    _px4_state.pose_frame = State::POSE_FRAME_NED;
    _px4_state.timestamp = getCurrentTimeMs();
}

void Visualization::vioOdomImuCallback(const odometryMsg::UniquePtr msg)
{
    if (!_show_vio)
        return;
    _vio_state.position[0] = msg->pose.pose.position.x;
    _vio_state.position[1] = msg->pose.pose.position.y;
    _vio_state.position[2] = msg->pose.pose.position.z;
    _vio_state.q[0] = msg->pose.pose.orientation.w;
    _vio_state.q[1] = msg->pose.pose.orientation.x;
    _vio_state.q[2] = msg->pose.pose.orientation.y;
    _vio_state.q[3] = msg->pose.pose.orientation.z;
    _vio_state.pose_frame = State::POSE_FRAME_ENU;
    _vio_state.timestamp = getCurrentTimeMs();
}

void Visualization::commandCallback()
{
    const rclcpp::Time now = this->get_clock()->now();
    _px4_path.header.frame_id = "global";
    _px4_path.header.stamp = now;
    _vio_path.header.frame_id = "global";
    _vio_path.header.stamp = now;

    {
        Eigen::Vector3d p_px4;
        Eigen::Quaterniond q_px4;
        computeENU(_px4_state, p_px4, q_px4);

        transformStampedMsg t_px4;
        setupTF(t_px4, p_px4, q_px4, "global", "px4_state");
        t_px4.header.stamp = now;
        _tf_px4->sendTransform(t_px4);

        const bool added = addPoseIfNew(p_px4, _px4_path, _prev_pose_px4);
        if (added)
        {
            if (!_traj_have_px4)
            {
                _traj_last_px4 = p_px4;
                _traj_have_px4 = true;
            }
            else
            {
                _traj_dist_px4 += (p_px4 - _traj_last_px4).norm();
                _traj_last_px4 = p_px4;
            }
        }
        _pub.px4_path->publish(_px4_path);
    }

    if (_show_vio && _vio_state.timestamp != 0)
    {
        Eigen::Vector3d p_vio;
        Eigen::Quaterniond q_vio;
        computeENU(_vio_state, p_vio, q_vio);

        transformStampedMsg t_vio;
        setupTF(t_vio, p_vio, q_vio, "global", "vio_state");
        t_vio.header.stamp = now;
        _tf_vio->sendTransform(t_vio);

        const bool added = addPoseIfNew(p_vio, _vio_path, _prev_pose_vio);
        if (added)
        {
            if (!_traj_have_vio)
            {
                _traj_last_vio = p_vio;
                _traj_have_vio = true;
            }
            else
            {
                _traj_dist_vio += (p_vio - _traj_last_vio).norm();
                _traj_last_vio = p_vio;
            }
        }
        _pub.vio_path->publish(_vio_path);
    }

    const uint64_t now_ms = getCurrentTimeMs();
    if (now_ms - _last_report_ms >= 1000)
    {
        if (_traj_have_px4)
        {
            logInfo("[PX4][%5.1f, %5.1f, %3.1f] Toplam Mesafe: %.1f m",
                    _px4_state.position[0], _px4_state.position[1], -_px4_state.position[2], _traj_dist_px4);
        }

        if (_show_vio && _traj_have_vio)
        {
            logInfo("[VIO][%5.1f, %5.1f, %3.1f] Toplam Mesafe: %.1f m", 
                _vio_state.position[0], _vio_state.position[1], _vio_state.position[2], _traj_dist_vio);
        }
        _last_report_ms = now_ms;
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Visualization>());
    if (rclcpp::ok())
        rclcpp::shutdown();
    return 0;
}
