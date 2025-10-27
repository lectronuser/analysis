#!/usr/bin/env python3
import time, math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
import psutil

TOPIC = "/ov_msckf/poseimu"
SILENCE_SEC = 1.0
SAMPLE_PERIOD = 0.2

class Monitor(Node):
    def __init__(self):
        super().__init__("raspi_cpu_ram_monitor")
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(PoseWithCovarianceStamped, TOPIC, self.cb, qos)
        self.timer = self.create_timer(SAMPLE_PERIOD, self.tick)

        self.started = False
        self.start_t = None
        self.last_msg_t = None
        self.done = False

        self.cpu_sum = 0.0
        self.cpu_count = 0
        self.cpu_peak = 0.0

        self.core_ready = False
        self.core_sum = []
        self.core_count = []
        self.core_peak = []

        self.ram_sum_pct = 0.0
        self.ram_sum_used = 0.0
        self.ram_count = 0
        self.ram_peak_pct = 0.0
        self.ram_peak_used = 0

        self.temp_sum = {}   # key -> sum
        self.temp_cnt = {}   # key -> count
        self.temp_peak = {}  # key -> peak

    def cb(self, _msg):
        now = time.monotonic()
        if not self.started:
            self.started = True
            self.start_t = now
        self.last_msg_t = now

    def tick(self):
        if not self.started:
            return

        cpu_total = psutil.cpu_percent(interval=None)
        self.cpu_sum += cpu_total
        self.cpu_count += 1
        if cpu_total > self.cpu_peak:
            self.cpu_peak = cpu_total

        per = psutil.cpu_percent(percpu=True, interval=None)
        if not self.core_ready:
            n = len(per)
            self.core_sum = [0.0]*n
            self.core_count = [0]*n
            self.core_peak = [0.0]*n
            self.core_ready = True
        for i, v in enumerate(per):
            self.core_sum[i] += v
            self.core_count[i] += 1
            if v > self.core_peak[i]:
                self.core_peak[i] = v

        mem = psutil.virtual_memory()
        self.ram_sum_pct += mem.percent
        self.ram_sum_used += mem.used
        self.ram_count += 1
        if mem.percent > self.ram_peak_pct:
            self.ram_peak_pct = mem.percent
        if mem.used > self.ram_peak_used:
            self.ram_peak_used = mem.used

        temps = psutil.sensors_temperatures() or {}
        for label, entries in temps.items():
            for e in entries:
                cur = getattr(e, "current", None)
                if cur is None or math.isnan(cur):
                    continue
                key = f"{label}:{getattr(e,'label','') or getattr(e,'device','') or 't'}"
                self.temp_sum[key] = self.temp_sum.get(key, 0.0) + float(cur)
                self.temp_cnt[key] = self.temp_cnt.get(key, 0) + 1
                self.temp_peak[key] = max(self.temp_peak.get(key, float("-inf")), float(cur))

        now = time.monotonic()
        if self.last_msg_t and (now - self.last_msg_t) >= SILENCE_SEC and not self.done:
            self.done = True
            self.report()
            rclpy.shutdown()

    def report(self):
        dur = (self.last_msg_t - self.start_t) if (self.start_t and self.last_msg_t) else 0.0
        cpu_avg = (self.cpu_sum / self.cpu_count) if self.cpu_count else 0.0
        ram_avg_pct = (self.ram_sum_pct / self.ram_count) if self.ram_count else 0.0
        ram_avg_used = (self.ram_sum_used / self.ram_count) if self.ram_count else 0.0

        print(f"Süre: {dur:.2f}s")
        print(f"CPU Ort: {cpu_avg:.2f}% | Max: {self.cpu_peak:.2f}%")
        if self.core_ready:
            for i in range(len(self.core_sum)):
                avg = (self.core_sum[i]/self.core_count[i]) if self.core_count[i] else 0.0
                print(f"CPU[{i}] Ort: {avg:.2f}% | Max: {self.core_peak[i]:.2f}%")
        print(f"RAM Ort: {ram_avg_pct:.2f}% (~{ram_avg_used/1e9:.2f} GB) | Max: {self.ram_peak_pct:.2f}% (~{self.ram_peak_used/1e9:.2f} GB)")
        if self.temp_sum:
            print("Sıcaklık (°C):")
            for k in sorted(self.temp_sum.keys()):
                tavg = self.temp_sum[k] / max(1, self.temp_cnt[k])
                tpk = self.temp_peak.get(k, float("nan"))
                print(f"{k} Ort: {tavg:.1f} | Max: {tpk:.1f}")
        else:
            print("Sıcaklık verisi yok.")

        exit()

def main():
    rclpy.init()
    node = Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.report()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
