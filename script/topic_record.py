#!/usr/bin/env python3
import os, re, csv, psutil, subprocess, signal, time
from glob import glob

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import SingleThreadedExecutor
from px4_msgs.msg import VehicleStatus

BAG_DIR = os.path.expanduser('~/bag')
BASE_NAME = 'realsense_bag'
TOPICS = [
    '/d456/d456/infra1/image_rect_raw',
    '/d456/d456/infra2/image_rect_raw',
    '/d456/d456/imu',
    '/fmu/out/vehicle_odometry',
]
VEH_STATUS_TOPIC = '/fmu/out/vehicle_status'
SIZE_LIMIT_GB = 20
SIZE_LIMIT_BYTES = SIZE_LIMIT_GB * (1024 ** 3)
CSV_PERIOD_SEC = 1.0
SIZE_CHECK_PERIOD_SEC = 2.0

def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except FileNotFoundError:
                pass
    return total

def next_bag_name(base_dir: str, base_name: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    names = []
    for d in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, d)):
            m = re.fullmatch(rf'{re.escape(base_name)}(?:_(\d+))?', d)
            if m:
                idx = int(m.group(1)) if m and m.group(1) else 0
                names.append(idx)
    if not names:
        candidate = base_name
        if os.path.exists(os.path.join(base_dir, candidate)):
            return f'{base_name}_1'
        return candidate
    n = max(names) + 1 if max(names) > 0 or base_name in os.listdir(base_dir) else 1
    candidate = f'{base_name}_{n}' if n > 0 else base_name
    return candidate

def first_temp_c() -> float:
    try:
        temps = psutil.sensors_temperatures()
        vals = []
        for _, arr in temps.items():
            for t in arr:
                if t.current is not None:
                    vals.append(float(t.current))
        if vals:
            return max(vals)
    except Exception:
        pass
    zs = sorted(glob('/sys/class/thermal/thermal_zone*/temp'))
    vals = []
    for z in zs:
        try:
            with open(z, 'r') as f:
                vals.append(float(f.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(vals) if vals else float('nan')

class BagRecorder(Node):
    def __init__(self):
        super().__init__('bag_recorder_guard')
        os.makedirs(BAG_DIR, exist_ok=True)
        self.proc = None
        self.current_bag_name = None
        self.csv_path = None
        self.csv_ready = False

        self.armed = False
        self.last_armed = False
        self._shutting_down = False

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.sub = self.create_subscription(VehicleStatus, VEH_STATUS_TOPIC, self.on_status, qos)
        self.size_timer = self.create_timer(SIZE_CHECK_PERIOD_SEC, self.check_size)
        self.csv_timer = self.create_timer(CSV_PERIOD_SEC, self.write_metrics)

        self.cpu_count = psutil.cpu_count(logical=True) or 1

    def on_status(self, msg: VehicleStatus):
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        if self.armed and not self.last_armed and self.proc is None:
            if dir_size_bytes(BAG_DIR) >= SIZE_LIMIT_BYTES:
                self.get_logger().warn('Boyut limiti aşıldı, kayıt başlatılmadı.')
            else:
                self.start_record()
        if (not self.armed) and self.last_armed and self.proc is not None:
            self.stop_record('DISARM')
        self.last_armed = self.armed

    def start_record(self):
        self.current_bag_name = next_bag_name(BAG_DIR, BASE_NAME)
        out_path = os.path.join(BAG_DIR, self.current_bag_name)
        self.csv_path = os.path.join(BAG_DIR, f'{self.current_bag_name}.csv')
        try:
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                header = ['ros_time_ns', 'ram_percent'] + [f'cpu{i}_percent' for i in range(self.cpu_count)] + ['temp_c']
                w.writerow(header)
            self.csv_ready = True
            psutil.cpu_percent(interval=None, percpu=True)
        except Exception as e:
            self.get_logger().error(f'CSV açılamadı: {e}')
            self.csv_ready = False

        """#cmd = ['ros2', 'bag', 'record', '-o', out_path] + TOPICS
        #try:
        #    self.proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            self.get_logger().info(f'Kayıt başladı: {self.current_bag_name}')
        except Exception as e:
            self.get_logger().error(f'Kayıt başlatılamadı: {e}')
            self.proc = None
            self.current_bag_name = None
            self.csv_ready = False
            self.csv_path = None"""

    def stop_record(self, reason: str):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
            t0 = time.time()
            while (time.time() - t0) < 8.0:
                rc = self.proc.poll()
                if rc is not None:
                    break
                time.sleep(0.2)
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.get_logger().info(f'Kayıt durdu ({reason}): {self.current_bag_name}')
        except Exception as e:
            self.get_logger().error(f'Kayıt durdurulamadı: {e}')
        finally:
            self.proc = None
            self.current_bag_name = None
            self.csv_ready = False
            self.csv_path = None

    def check_size(self):
        if self.proc is not None and dir_size_bytes(BAG_DIR) >= SIZE_LIMIT_BYTES:
            self.stop_record('Boyut limiti')

    def write_metrics(self):
        if not self.csv_ready or self.proc is None or self.csv_path is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        ram = psutil.virtual_memory().percent
        cpu_list = psutil.cpu_percent(interval=None, percpu=True)
        if not cpu_list or len(cpu_list) != self.cpu_count:
            cpu_list = (cpu_list or []) + [0.0] * (self.cpu_count - len(cpu_list or []))
        tempc = first_temp_c()
        row = [now_ns, ram] + list(cpu_list) + [tempc]
        try:
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            self.get_logger().error(f'CSV yazılamadı: {e}')

    def shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            if self.proc is not None:
                self.stop_record('Çıkış')
            try:
                self.size_timer.cancel()
                self.destroy_timer(self.size_timer)
            except Exception:
                pass
            try:
                self.csv_timer.cancel()
                self.destroy_timer(self.csv_timer)
            except Exception:
                pass
            try:
                self.destroy_subscription(self.sub)
            except Exception:
                pass
        finally:
            try:
                self.destroy_node()
            except Exception:
                pass

def main():
    rclpy.init()
    node = BagRecorder()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C alındı, temiz kapanış...')
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        node.shutdown()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
