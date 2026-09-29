import ipaddress
import math
import os
import re
import struct
import sys
import time

os.environ.setdefault("MAVLINK20", "1")

import serial.tools.list_ports
from pymavlink import mavutil
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

mavlink = mavutil.mavlink

APP_NAME = "Board Test"
BAUD_RATES = ["9600", "57600", "115200", "230400", "460800", "921600"]
DEFAULT_BAUD = "57600"
REFRESH_MS = 250
STALE_S = 2.0
HEARTBEAT_TIMEOUT_S = 3.0
PARAM_RETRY_MS = 2000
PARAM_MAX_TRIES = 3
STATUS_LOG_MAX = 1000
LOG_TABLE_HEIGHT = 320
PWM_MIN, PWM_MAX = 800, 2200

# Board specific wiring
CM5_SERIAL_PORT = 3  # SERIAL3 = TELEM3 (USART2)

C = {
    "bg": "#0f1115",
    "surface": "#171a21",
    "surface2": "#1f232c",
    "row_alt": "#1a1e26",
    "border": "#2a2f3a",
    "text": "#e6e8eb",
    "muted": "#8b93a1",
    "accent": "#ff8a00",
    "ok": "#22c55e",
    "warn": "#f59e0b",
    "err": "#ef4444",
    "off": "#4b5563",
}

STYLE = f"""
QWidget {{ color: {C['text']}; font-family: "Inter", "Segoe UI", "Roboto", sans-serif; font-size: 13px; }}
QMainWindow, #page, #central {{ background: {C['bg']}; }}
QLabel {{ background: transparent; }}
#header {{ background: {C['surface']}; border-bottom: 1px solid {C['border']}; }}
#appTitle {{ font-size: 16px; font-weight: 700; }}
#appDot {{ color: {C['accent']}; font-size: 20px; }}
#card {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 10px; }}
#cardTitle {{ color: {C['muted']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
#badge {{ color: {C['accent']}; border: 1px solid #5a3a10; border-radius: 4px; padding: 1px 6px; font-size: 11px; font-weight: 700; }}
#kvKey {{ color: {C['muted']}; }}
#kvValue {{ font-weight: 600; }}
QPushButton {{ background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {C['accent']}; }}
QPushButton:disabled {{ color: {C['off']}; border-color: {C['border']}; }}
QPushButton#primary {{ background: {C['accent']}; color: #111; border: none; font-weight: 700; padding: 7px 18px; }}
QPushButton#primary:hover {{ background: #ffa033; }}
QPushButton#primary[connected="true"] {{ background: {C['err']}; color: white; }}
QPushButton#nav {{ background: transparent; border: 1px solid transparent; color: {C['muted']}; font-weight: 600; }}
QPushButton#nav:hover {{ color: {C['text']}; }}
QPushButton#nav:checked {{ background: #2a1d0c; color: {C['accent']}; border-color: #5a3a10; }}
QComboBox, QLineEdit {{ background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 5px 8px; }}
QLineEdit:focus {{ border-color: {C['accent']}; }}
QComboBox:disabled, QLineEdit:disabled {{ color: {C['muted']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {C['surface2']}; border: 1px solid {C['border']}; selection-background-color: {C['accent']}; selection-color: #111; }}
QCheckBox {{ spacing: 8px; }}
QTableWidget {{ background: {C['surface']}; alternate-background-color: {C['row_alt']}; border: none; gridline-color: transparent; selection-background-color: #2a3140; selection-color: {C['text']}; }}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{ background: {C['surface']}; color: {C['muted']}; border: none; border-bottom: 1px solid {C['border']}; padding: 8px; font-weight: 700; }}
QStatusBar {{ background: {C['surface']}; color: {C['muted']}; border-top: 1px solid {C['border']}; padding-left: 12px; }}
QStatusBar QLabel {{ padding-left: 12px; }}
QToolTip {{ background: {C['surface2']}; color: {C['text']}; border: 1px solid {C['border']}; padding: 4px; }}
QProgressBar {{ background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 4px; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 3px; }}
QScrollArea {{ border: none; background: {C['bg']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C['border']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""

BUS_TYPES = {0: "Unknown", 1: "I2C", 2: "SPI", 3: "DroneCAN", 4: "SITL", 5: "MSP", 6: "Serial", 7: "QSPI"}

IMU_CHIPS = {
    0x09: "BMI160", 0x10: "L3G4200D", 0x11: "LSM303D", 0x12: "BMA180", 0x13: "MPU6000",
    0x16: "MPU9250", 0x17: "IIS328DQ", 0x18: "LSM9DS1", 0x21: "MPU6000", 0x22: "L3GD20",
    0x24: "MPU9250", 0x25: "I3G4250D", 0x26: "LSM9DS1", 0x27: "ICM20789", 0x28: "ICM20689",
    0x29: "BMI055", 0x2A: "SITL", 0x2B: "BMI088", 0x2C: "ICM20948", 0x2D: "ICM20648",
    0x2E: "ICM20649", 0x2F: "ICM20602", 0x30: "ICM20601", 0x31: "ADIS1647x", 0x32: "Serial",
    0x33: "ICM40609", 0x34: "ICM42688", 0x35: "ICM42605", 0x36: "ICM40605", 0x37: "IIM42652",
    0x38: "BMI270", 0x39: "BMI085", 0x3A: "ICM42670", 0x3B: "ICM45686", 0x3C: "SCHA63T",
    0x3D: "IIM42653",
}

MAG_CHIPS = {
    0x01: "HMC5883 (old)", 0x02: "LSM303D", 0x04: "AK8963", 0x05: "BMM150", 0x06: "LSM9DS1",
    0x07: "HMC5883", 0x08: "LIS3MDL", 0x09: "AK09916", 0x0A: "IST8310", 0x0B: "ICM20948",
    0x0C: "MMC3416", 0x0D: "QMC5883L", 0x0E: "MAG3110", 0x0F: "SITL", 0x10: "IST8308",
    0x11: "RM3100", 0x12: "RM3100", 0x13: "MMC5883", 0x14: "AK09918", 0x15: "AK09915",
    0x16: "QMC5883P", 0x17: "BMM350", 0x18: "IIS2MDC",
}

BARO_CHIPS = {
    0x01: "SITL", 0x02: "BMP085", 0x03: "BMP280", 0x04: "BMP388", 0x05: "DPS280",
    0x06: "DPS310", 0x07: "FBM320", 0x08: "ICM20789", 0x09: "KellerLD", 0x0A: "LPS2XH",
    0x0B: "MS5611", 0x0C: "SPL06", 0x0D: "DroneCAN", 0x0E: "MSP", 0x0F: "ICP101XX",
    0x10: "ICP201XX", 0x11: "MS5607", 0x12: "MS5837", 0x13: "MS5637", 0x14: "BMP390",
    0x15: "BMP581", 0x16: "SPA06", 0x17: "AUAV",
}

CHIP_TABLES = {"accel": IMU_CHIPS, "gyro": IMU_CHIPS, "mag": MAG_CHIPS, "baro": BARO_CHIPS}

GPS_TYPES = {
    0: "None", 1: "Auto", 2: "uBlox", 5: "NMEA", 6: "SiRF", 8: "SwiftNav", 9: "DroneCAN",
    10: "SBF", 11: "GSOF", 13: "ERB", 14: "MAVLink", 15: "NOVA", 16: "Hemisphere NMEA",
    17: "uBlox MB Base", 18: "uBlox MB Rover", 19: "MSP", 20: "AllyStar", 21: "External AHRS",
    22: "DroneCAN MB Base", 23: "DroneCAN MB Rover", 24: "Unicore NMEA",
    25: "Unicore MB NMEA", 26: "SBF Dual Antenna",
}

GPS_FIX = {0: "No GPS", 1: "No Fix", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK Float", 6: "RTK Fixed",
           7: "Static", 8: "PPP"}

SERIAL_BAUDS = {1: 1200, 2: 2400, 4: 4800, 9: 9600, 19: 19200, 38: 38400, 57: 57600,
                111: 111100, 115: 115200, 230: 230400, 256: 256000, 460: 460800,
                500: 500000, 921: 921600, 1500: 1500000, 2000: 2000000}

SERIAL_PROTOCOLS = {-1: "None", 1: "MAVLink1", 2: "MAVLink2", 5: "GPS", 9: "Rangefinder",
                    18: "OpticalFlow", 23: "RCIN", 28: "Scripting", 32: "MSP", 45: "DDS/XRCE"}

SERIAL_PROTOCOL_GPS = 5

AUTOPILOT_NAMES = {3: "ArduPilot", 12: "PX4", 0: "Generic"}

KIND_LABELS = {
    "accel": "Accelerometer", "gyro": "Gyroscope", "mag": "Compass", "baro": "Barometer",
    "gps": "GPS", "airspeed": "Airspeed", "range": "Rangefinder", "flow": "Optical Flow",
    "temp": "IMU Temperature",
}

IMU_PARTS = (("accel", "Accelerometer"), ("gyro", "Gyroscope"), ("temp", "Temperature"))

INT_TYPES = (mavlink.MAV_PARAM_TYPE_INT8, mavlink.MAV_PARAM_TYPE_INT16, mavlink.MAV_PARAM_TYPE_INT32,
             mavlink.MAV_PARAM_TYPE_UINT8, mavlink.MAV_PARAM_TYPE_UINT16, mavlink.MAV_PARAM_TYPE_UINT32)

AP_DEVICE_PARAMS = {
    "accel": ["INS_ACC_ID", "INS_ACC2_ID", "INS_ACC3_ID"],
    "gyro": ["INS_GYR_ID", "INS_GYR2_ID", "INS_GYR3_ID"],
    "mag": ["COMPASS_DEV_ID", "COMPASS_DEV_ID2", "COMPASS_DEV_ID3"],
    "baro": ["BARO1_DEVID", "BARO2_DEVID", "BARO3_DEVID"],
}

PX4_DEVICE_PARAMS = {
    "accel": ["CAL_ACC0_ID", "CAL_ACC1_ID", "CAL_ACC2_ID"],
    "gyro": ["CAL_GYRO0_ID", "CAL_GYRO1_ID", "CAL_GYRO2_ID"],
    "mag": ["CAL_MAG0_ID", "CAL_MAG1_ID", "CAL_MAG2_ID"],
    "baro": ["CAL_BARO0_ID", "CAL_BARO1_ID", "CAL_BARO2_ID"],
}

AP_EXTRA_PARAMS = (
    ["GPS_TYPE", "GPS_TYPE2", "GPS1_TYPE", "GPS2_TYPE", "RNGFND1_TYPE", "FLOW_TYPE", "ARSPD_TYPE",
     "BATT_MONITOR", "BRD_IO_ENABLE"]
    + [f"SERIAL{i}_PROTOCOL" for i in range(1, 9)]
    + [f"SERIAL{i}_BAUD" for i in range(1, 9)]
)

# External peripherals on the setup page. "{port}" is replaced by the selected SERIALn port and
# ("clear_bit", n) means "current value with bit n cleared" (e.g. un-masking a compass driver).
PERIPHERALS = [
    {"name": "HFlow", "iface": "CAN", "detect": "hflow",
     "note": "Optical flow + rangefinder over DroneCAN on CAN1",
     "params": [("CAN_P1_DRIVER", 1), ("CAN_D1_PROTOCOL", 1), ("FLOW_TYPE", 6),
                ("RNGFND2_TYPE", 24), ("RNGFND2_ORIENT", 25)]},
    {"name": "HMC5883 Compass", "iface": "I2C", "detect": "hmc5883",
     "note": "External I2C magnetometer",
     "params": [("COMPASS_ENABLE", 1), ("COMPASS_EXTERNAL", 1), ("COMPASS_TYPEMASK", ("clear_bit", 0))]},
    {"name": "TF-Luna Distance", "iface": "Serial", "port": 4, "detect": "tfluna",
     "note": "Benewake protocol, 115200 baud, facing down",
     "params": [("SERIAL{port}_PROTOCOL", 9), ("SERIAL{port}_BAUD", 115), ("RNGFND1_TYPE", 20),
                ("RNGFND1_ORIENT", 25)]},
    {"name": "Holybro PMW3901", "iface": "Serial", "port": 5, "detect": "pmw3901",
     "note": "Serial optical flow (CXOF protocol, 19200 baud)",
     "params": [("SERIAL{port}_PROTOCOL", 18), ("SERIAL{port}_BAUD", 19), ("FLOW_TYPE", 4)]},
    {"name": "MPU-9250/6500 Sensor Set", "iface": "I2C", "detect": "mpu9250",
     "note": "IMU is probed from hwdef; this enables its AK8963 compass",
     "params": [("COMPASS_TYPEMASK", ("clear_bit", 2))]},
]

NET_IP_PARAMS = [f"NET_IPADDR{i}" for i in range(4)]
NET_GW_PARAMS = [f"NET_GWADDR{i}" for i in range(4)]
NET_MAC_PARAMS = [f"NET_MACADDR{i}" for i in range(6)]
NET_PARAMS = ["NET_ENABLE", "NET_DHCP", "NET_NETMASK"] + NET_IP_PARAMS + NET_GW_PARAMS + NET_MAC_PARAMS

AP_SETUP_PARAMS = [name for p in PERIPHERALS for name, _ in p["params"] if "{port}" not in name] + NET_PARAMS


def enum_name(enum, value, prefix):
    entry = mavlink.enums.get(enum, {}).get(value)
    if not entry:
        return str(value)
    return entry.name.replace(prefix, "").replace("_", " ").title()


def decode_device_id(dev_id, kind, ardupilot):
    bus_type = dev_id & 0x07
    bus_num = (dev_id >> 3) & 0x1F
    address = (dev_id >> 8) & 0xFF
    devtype = (dev_id >> 16) & 0xFF
    bus = BUS_TYPES.get(bus_type, f"Bus {bus_type}")
    if bus_type in (1, 2, 7):
        addr_text = f"0x{address:02X}"
    elif bus_type == 3:
        addr_text = f"Node {address}"
    else:
        addr_text = str(address)
    chip = CHIP_TABLES.get(kind, {}).get(devtype) if ardupilot else None
    chip_text = f"{chip} (0x{devtype:02X})" if chip else f"Type 0x{devtype:02X}"
    return {"bus": f"{bus} {bus_num}", "address": addr_text, "chip": chip_text, "name": chip}


def xyz(x, y, z, unit, fmt="{:.0f}"):
    return f"X {fmt.format(x)}   Y {fmt.format(y)}   Z {fmt.format(z)}  {unit}"


def format_uptime(ms):
    s = int(ms / 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def format_param(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def make_table(headers, stretch_col, fit=False):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeToContents)
    header.setSectionResizeMode(stretch_col, QHeaderView.Stretch)
    header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    header.setHighlightSections(False)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(34)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setFocusPolicy(Qt.NoFocus)
    if fit:
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        fit_table_height(table)
    return table


def fit_table_height(table):
    rows = max(table.rowCount(), 1)
    height = table.horizontalHeader().sizeHint().height() + rows * table.verticalHeader().defaultSectionSize() + 4
    if table.height() != height:
        table.setFixedHeight(height)


def set_cell(table, row, col, text, color=None, bold=False):
    item = table.item(row, col)
    if item is None:
        item = QTableWidgetItem()
        table.setItem(row, col, item)
    if item.text() != text:
        item.setText(text)
        item.setToolTip(text)
    item.setForeground(QColor(color or C["text"]))
    if item.font().bold() != bold:
        font = item.font()
        font.setBold(bold)
        item.setFont(font)


def set_label(label, text, color=None):
    if label.text() != text:
        label.setText(text)
    label.setStyleSheet(f"color: {color};" if color else "")


class MavlinkReader(QThread):
    message = Signal(object)
    error = Signal(str)

    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self._running = True

    def run(self):
        while self._running:
            try:
                msg = self.conn.recv_match(blocking=True, timeout=0.2)
            except Exception as e:
                if self._running:
                    self.error.emit(str(e))
                break
            if msg is not None:
                self.message.emit(msg)

    def stop(self):
        self._running = False


class Card(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 14, 16, 16)
        self.body.setSpacing(10)
        head = QHBoxLayout()
        label = QLabel(title.upper())
        label.setObjectName("cardTitle")
        head.addWidget(label)
        head.addStretch()
        self.head = head
        self.body.addLayout(head)


class KeyValueCard(Card):
    def __init__(self, title, keys):
        super().__init__(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)
        self.values = {}
        for row, key in enumerate(keys):
            k = QLabel(key)
            k.setObjectName("kvKey")
            v = QLabel("-")
            v.setObjectName("kvValue")
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(k, row, 0)
            grid.addWidget(v, row, 1)
            self.values[key] = v
        self.body.addLayout(grid)
        self.body.addStretch()

    def set(self, key, text, color=None):
        set_label(self.values[key], str(text), color)

    def clear(self):
        for key in self.values:
            self.set(key, "-")


class ChannelCard(Card):
    def __init__(self, title, prefix):
        super().__init__(title)
        self.prefix = prefix
        self.info = QLabel("")
        self.info.setObjectName("kvKey")
        self.head.addWidget(self.info)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(1, 1)
        self.empty = QLabel("No data")
        self.empty.setObjectName("kvKey")
        self.grid.addWidget(self.empty, 0, 0)
        self.body.addLayout(self.grid)
        self.body.addStretch()
        self.rows = []

    def rebuild(self, count):
        for name, bar, value in self.rows:
            for w in (name, bar, value):
                self.grid.removeWidget(w)
                w.deleteLater()
        self.rows = []
        self.empty.setVisible(count == 0)
        for i in range(count):
            name = QLabel(f"{self.prefix}{i + 1}")
            name.setObjectName("kvKey")
            bar = QProgressBar()
            bar.setRange(PWM_MIN, PWM_MAX)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            value = QLabel("-")
            value.setObjectName("kvValue")
            value.setMinimumWidth(44)
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.grid.addWidget(name, i + 1, 0)
            self.grid.addWidget(bar, i + 1, 1)
            self.grid.addWidget(value, i + 1, 2)
            self.rows.append((name, bar, value))

    def set_values(self, values, info=""):
        if len(values) != len(self.rows):
            self.rebuild(len(values))
        for (_, bar, label), v in zip(self.rows, values):
            used = 0 < v < 0xFFFF
            bar.setValue(max(PWM_MIN, min(PWM_MAX, v)) if used else PWM_MIN)
            text = str(v) if used else "-"
            if label.text() != text:
                label.setText(text)
        if self.info.text() != info:
            self.info.setText(info)

    def clear(self):
        self.set_values([])


class PeripheralCard(Card):
    """Parameter preset for one external device: current values from the board, editable new values."""

    def __init__(self, spec, on_apply):
        super().__init__(spec["name"])
        self.spec = spec
        badge = QLabel(spec["iface"])
        badge.setObjectName("badge")
        self.head.insertWidget(1, badge)
        self.detect_label = QLabel("-")
        self.detect_label.setObjectName("kvValue")
        self.head.addWidget(self.detect_label)

        note = QLabel(spec["note"])
        note.setObjectName("kvKey")
        note.setWordWrap(True)
        self.body.addWidget(note)

        self.port_box = None
        if "port" in spec:
            row = QHBoxLayout()
            label = QLabel("Serial port")
            label.setObjectName("kvKey")
            self.port_box = QComboBox()
            self.port_box.addItems([f"SERIAL{i}" for i in range(1, 9)])
            self.port_box.setCurrentIndex(spec["port"] - 1)
            self.port_box.currentIndexChanged.connect(lambda _: self.rebuild())
            row.addWidget(label)
            row.addWidget(self.port_box)
            row.addStretch()
            self.body.addLayout(row)

        self.table = make_table(["Parameter", "Current", "New"], 0, fit=True)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
                                   | QAbstractItemView.EditKeyPressed)
        self.table.setFocusPolicy(Qt.StrongFocus)
        self.body.addWidget(self.table)

        foot = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("kvKey")
        self.status.setWordWrap(True)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(lambda: on_apply(self))
        foot.addWidget(self.status, 1)
        foot.addWidget(self.apply_btn)
        self.body.addLayout(foot)
        self.body.addStretch()
        self.rebuild()

    def entries(self):
        port = self.port_box.currentIndex() + 1 if self.port_box else None
        return [(name.format(port=port), value) for name, value in self.spec["params"]]

    def rebuild(self):
        entries = self.entries()
        self.table.setRowCount(len(entries))
        for i, (name, value) in enumerate(entries):
            for col, text in ((0, name), (1, "-"), (2, "" if isinstance(value, tuple) else str(value))):
                item = QTableWidgetItem(text)
                if col != 2:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, col, item)
        fit_table_height(self.table)
        self.status.setText("")

    def refresh(self, lookup, detected):
        for i, (name, value) in enumerate(self.entries()):
            text, color, current = lookup(name)
            set_cell(self.table, i, 1, text, color)
            new = self.table.item(i, 2)
            if isinstance(value, tuple) and not new.text() and current is not None:
                new.setText(str(int(current) & ~(1 << value[1])))
        set_label(self.detect_label, *detected)

    def new_values(self):
        return [(self.table.item(i, 0).text(), self.table.item(i, 2).text().strip())
                for i in range(self.table.rowCount())]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1180, 820)

        self.conn = None
        self.reader = None
        self.reset_state()

        central = QWidget()
        central.setObjectName("central")
        central.setAttribute(Qt.WA_StyledBackground, True)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.build_header())

        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_monitor())
        self.stack.addWidget(self.build_setup())
        root.addWidget(self.stack, 1)
        self.nav_group.idClicked.connect(self.stack.setCurrentIndex)

        self.statusBar().showMessage("Select a port and connect")

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(REFRESH_MS)
        self.refresh_timer.timeout.connect(self.refresh)
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.setInterval(1000)
        self.heartbeat_timer.timeout.connect(self.send_heartbeat)
        self.param_timer = QTimer(self)
        self.param_timer.setInterval(PARAM_RETRY_MS)
        self.param_timer.timeout.connect(self.request_missing_params)

        self.refresh_ports()
        self.set_connected_ui(False)

    def build_header(self):
        header = QWidget()
        header.setObjectName("header")
        header.setAttribute(Qt.WA_StyledBackground, True)
        lay = QHBoxLayout(header)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(10)

        dot = QLabel("●")
        dot.setObjectName("appDot")
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        lay.addWidget(dot)
        lay.addWidget(title)
        lay.addSpacing(16)

        self.nav_group = QButtonGroup(self)
        for i, name in enumerate(("Monitor", "Setup")):
            btn = QPushButton(name)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            self.nav_group.addButton(btn, i)
            lay.addWidget(btn)
        lay.addStretch()

        self.port_box = QComboBox()
        self.port_box.setEditable(True)
        self.port_box.setMinimumWidth(280)
        self.refresh_btn = QPushButton("Refresh")
        self.baud_box = QComboBox()
        self.baud_box.addItems(BAUD_RATES)
        self.baud_box.setCurrentText(DEFAULT_BAUD)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("primary")
        self.connect_btn.setMinimumWidth(120)
        self.status_pill = QLabel()
        self.status_pill.setMinimumWidth(150)
        self.status_pill.setAlignment(Qt.AlignCenter)

        lay.addWidget(self.port_box)
        lay.addWidget(self.refresh_btn)
        lay.addWidget(self.baud_box)
        lay.addWidget(self.connect_btn)
        lay.addSpacing(6)
        lay.addWidget(self.status_pill)

        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.toggle_connection)
        return header

    @staticmethod
    def scroll_page():
        page = QWidget()
        page.setObjectName("page")
        page.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(page)
        return area, lay

    def build_monitor(self):
        area, lay = self.scroll_page()

        self.board_card = KeyValueCard("Board", [
            "Board", "Firmware", "Autopilot", "Vehicle", "Frame", "Flight Mode", "Armed",
            "System State", "System / Component", "Board ID", "UID", "Uptime",
        ])
        self.rc_card = ChannelCard("RC Channels", "CH")
        self.servo_card = ChannelCard("Motor Outputs", "OUT")
        top = QHBoxLayout()
        top.setSpacing(16)
        for card in (self.board_card, self.rc_card, self.servo_card):
            top.addWidget(card, 1)

        live_card = Card("Live Data")
        self.param_label = QLabel("")
        self.param_label.setObjectName("kvKey")
        self.reload_btn = QPushButton("Reload IDs")
        self.reload_btn.clicked.connect(self.start_param_fetch)
        live_card.head.addWidget(self.param_label)
        live_card.head.addSpacing(10)
        live_card.head.addWidget(self.reload_btn)
        self.live_table = make_table(["Item", "Chip", "Bus", "Address / Port", "Value"], 4, fit=True)
        live_card.body.addWidget(self.live_table)

        status_card = Card("Messages")
        self.severity_box = QComboBox()
        self.severity_box.addItem("All messages", mavlink.MAV_SEVERITY_DEBUG)
        self.severity_box.addItem("Warnings and errors", mavlink.MAV_SEVERITY_WARNING)
        self.severity_box.addItem("Errors only", mavlink.MAV_SEVERITY_ERROR)
        self.severity_box.currentIndexChanged.connect(lambda _: self.refresh_status_log())
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_status_log)
        status_card.head.addWidget(self.severity_box)
        status_card.head.addWidget(clear_btn)
        self.status_table = make_table(["Time", "Severity", "Message"], 2)
        self.status_table.setFixedHeight(LOG_TABLE_HEIGHT)
        status_card.body.addWidget(self.status_table)

        lay.addLayout(top)
        lay.addWidget(live_card)
        lay.addWidget(status_card)
        lay.addStretch()
        return area

    def build_setup(self):
        area, lay = self.scroll_page()

        bar = QHBoxLayout()
        self.setup_param_label = QLabel("")
        self.setup_param_label.setObjectName("kvKey")
        self.read_params_btn = QPushButton("Read Parameters")
        self.read_params_btn.clicked.connect(self.start_param_fetch)
        self.reboot_btn = QPushButton("Reboot Board")
        self.reboot_btn.clicked.connect(self.reboot_board)
        bar.addWidget(self.setup_param_label, 1)
        bar.addWidget(self.read_params_btn)
        bar.addWidget(self.reboot_btn)
        lay.addLayout(bar)

        onboard_card = Card("Onboard Hardware")
        self.onboard_table = make_table(["Item", "Interface", "Status", "Details"], 3, fit=True)
        onboard_card.body.addWidget(self.onboard_table)
        lay.addWidget(onboard_card)

        grid = QGridLayout()
        grid.setSpacing(16)
        self.peripheral_cards = []
        for i, spec in enumerate(PERIPHERALS):
            card = PeripheralCard(spec, self.apply_peripheral)
            self.peripheral_cards.append(card)
            grid.addWidget(card, i // 2, i % 2)
        net_card = self.build_network_card()
        grid.addWidget(net_card, len(PERIPHERALS) // 2, len(PERIPHERALS) % 2)
        for col in range(2):
            grid.setColumnStretch(col, 1)
        lay.addLayout(grid)
        lay.addStretch()
        return area

    def build_network_card(self):
        card = Card("Network")
        badge = QLabel("Ethernet")
        badge.setObjectName("badge")
        card.head.insertWidget(1, badge)

        self.net_enable = QCheckBox("Enable networking")
        self.net_dhcp = QCheckBox("Use DHCP")
        self.net_dhcp.toggled.connect(self.update_net_fields)
        self.net_ip = QLineEdit()
        self.net_mask = QLineEdit()
        self.net_gw = QLineEdit()
        self.net_ip.setPlaceholderText("192.168.144.14")
        self.net_mask.setPlaceholderText("255.255.255.0")
        self.net_gw.setPlaceholderText("192.168.144.1")
        self.net_mac = QLabel("-")
        self.net_mac.setObjectName("kvValue")
        self.net_mac.setTextInteractionFlags(Qt.TextSelectableByMouse)

        form = QGridLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        form.setColumnStretch(1, 1)
        form.addWidget(self.net_enable, 0, 0, 1, 2)
        form.addWidget(self.net_dhcp, 1, 0, 1, 2)
        for row, (text, widget) in enumerate((("IP address", self.net_ip), ("Netmask", self.net_mask),
                                              ("Gateway", self.net_gw), ("MAC address", self.net_mac)), 2):
            label = QLabel(text)
            label.setObjectName("kvKey")
            form.addWidget(label, row, 0)
            form.addWidget(widget, row, 1)
        card.body.addLayout(form)

        foot = QHBoxLayout()
        self.net_status = QLabel("")
        self.net_status.setObjectName("kvKey")
        self.net_status.setWordWrap(True)
        self.net_reload_btn = QPushButton("Reload")
        self.net_reload_btn.clicked.connect(self.reload_network)
        self.net_apply_btn = QPushButton("Apply")
        self.net_apply_btn.clicked.connect(self.apply_network)
        foot.addWidget(self.net_status, 1)
        foot.addWidget(self.net_reload_btn)
        foot.addWidget(self.net_apply_btn)
        card.body.addLayout(foot)
        card.body.addStretch()
        return card

    def update_net_fields(self):
        static = not self.net_dhcp.isChecked()
        for w in (self.net_ip, self.net_mask, self.net_gw):
            w.setEnabled(static)

    def clear_status_log(self):
        self.status_log = []
        self.status_seq += 1
        self.refresh_status_log()

    def reset_state(self):
        self.target = None
        self.autopilot = None
        self.last_heartbeat = 0.0
        self.hb = {}
        self.version = {}
        self.params = {}
        self.param_types = {}
        self.param_names = []
        self.param_tries = 0
        self.live = {}
        self.values = {}
        self.attitude = None
        self.power = None
        self.gps_info = {}
        self.sys_status = None
        self.power_status = None
        self.mcu = None
        self.hwstatus = None
        self.iomcu_msg = None
        self.rc = None
        self.servo = None
        self.status_log = []
        self.status_seq = 0
        self.status_shown = None
        self.net_loaded = False

    def notify(self, text, ms=8000):
        self.statusBar().showMessage(text, ms)

    def set_connected_ui(self, connected):
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.connect_btn.setProperty("connected", connected)
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)
        self.port_box.setEnabled(not connected)
        self.baud_box.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)
        for btn in (self.reload_btn, self.read_params_btn, self.reboot_btn, self.net_reload_btn,
                    self.net_apply_btn, *(c.apply_btn for c in self.peripheral_cards)):
            btn.setEnabled(connected)
        if connected:
            self.refresh_timer.start()
            self.heartbeat_timer.start()
        else:
            self.refresh_timer.stop()
            self.heartbeat_timer.stop()
            self.param_timer.stop()
        self.update_status_pill()

    def update_status_pill(self):
        if not self.conn:
            text, color = "Disconnected", C["err"]
        elif self.target is None:
            text, color = "Waiting heartbeat", C["warn"]
        elif time.monotonic() - self.last_heartbeat > HEARTBEAT_TIMEOUT_S:
            text, color = "Link lost", C["err"]
        else:
            text, color = "Connected", C["ok"]
        self.status_pill.setText(f"●  {text}")
        self.status_pill.setStyleSheet(
            f"color: {color}; background: {C['surface2']}; border: 1px solid {color};"
            f"border-radius: 13px; padding: 4px 12px; font-weight: 700;")

    def refresh_ports(self):
        self.port_box.clear()
        ports = sorted(serial.tools.list_ports.comports(), key=lambda p: p.device)
        for p in ports:
            self.port_box.addItem(f"{p.device} - {p.description}", p.device)
        if not ports:
            self.port_box.setEditText("")

    def selected_port(self):
        text = self.port_box.currentText().strip()
        i = self.port_box.findText(text)
        if i >= 0 and self.port_box.itemData(i):
            return self.port_box.itemData(i)
        return text or None

    def toggle_connection(self):
        if self.conn:
            self.disconnect_board()
            return
        port = self.selected_port()
        if not port:
            self.notify("No port selected")
            return
        try:
            self.conn = mavutil.mavlink_connection(
                port, baud=int(self.baud_box.currentText()), source_system=255,
                source_component=mavlink.MAV_COMP_ID_MISSIONPLANNER, autoreconnect=False)
        except Exception as e:
            self.conn = None
            self.notify(f"Could not connect: {e}")
            return

        self.reset_state()
        self.clear_views()
        self.reader = MavlinkReader(self.conn)
        self.reader.message.connect(self.handle_message)
        self.reader.error.connect(self.connection_error)
        self.reader.start()
        self.set_connected_ui(True)
        self.notify(f"{port} opened, waiting for heartbeat...")

    def disconnect_board(self):
        if self.reader:
            self.reader.stop()
            self.reader.wait(1000)
            self.reader = None
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self.notify("Disconnected")
        self.set_connected_ui(False)

    def connection_error(self, text):
        self.disconnect_board()
        self.notify(f"Connection error: {text}", 0)

    def clear_views(self):
        for table in (self.live_table, self.status_table, self.onboard_table):
            table.setRowCount(0)
            fit_table_height(table)
        self.status_table.setFixedHeight(LOG_TABLE_HEIGHT)
        for card in (self.board_card, self.rc_card, self.servo_card):
            card.clear()
        for card in self.peripheral_cards:
            card.rebuild()
        for w in (self.net_ip, self.net_mask, self.net_gw):
            w.clear()
        self.net_mac.setText("-")
        self.net_status.setText("")
        self.param_label.setText("")
        self.setup_param_label.setText("")

    def send(self, func, *args):
        if not self.conn:
            return
        try:
            func(*args)
        except Exception as e:
            self.connection_error(str(e))

    def send_heartbeat(self):
        self.send(self.conn.mav.heartbeat_send, mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def send_command(self, command, *params):
        sysid, compid = self.target
        values = list(params) + [0] * (7 - len(params))
        self.send(self.conn.mav.command_long_send, sysid, compid, command, 0, *values)

    def on_board_found(self):
        sysid, compid = self.target
        self.send(self.conn.mav.request_data_stream_send, sysid, compid, mavlink.MAV_DATA_STREAM_ALL, 4, 1)
        self.send_command(mavlink.MAV_CMD_REQUEST_MESSAGE, mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION)
        if self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            self.send_command(mavlink.MAV_CMD_DO_SEND_BANNER)
        self.start_param_fetch()

    def reboot_board(self):
        if not self.target:
            return
        answer = QMessageBox.question(self, "Reboot Board", "Reboot the flight controller now?")
        if answer == QMessageBox.Yes:
            self.send_command(mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 1)
            self.notify("Reboot command sent")

    # ---- parameters -------------------------------------------------------------------------

    def start_param_fetch(self):
        if not self.conn or not self.target:
            return
        if self.autopilot == mavlink.MAV_AUTOPILOT_PX4:
            names = [n for group in PX4_DEVICE_PARAMS.values() for n in group]
        elif self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            names = [n for group in AP_DEVICE_PARAMS.values() for n in group] + AP_EXTRA_PARAMS + AP_SETUP_PARAMS
        else:
            names = []
        self.param_names = list(dict.fromkeys(names))
        for name in self.param_names:
            self.params.pop(name, None)
        self.net_loaded = False
        self.fetch_params()

    def fetch_params(self):
        self.param_tries = 0
        self.request_missing_params()
        if self.param_names:
            self.param_timer.start()

    def request_missing_params(self):
        missing = [n for n in self.param_names if n not in self.params]
        if not missing or self.param_tries >= PARAM_MAX_TRIES:
            self.param_timer.stop()
            return
        self.param_tries += 1
        sysid, compid = self.target
        for name in missing:
            self.send(self.conn.mav.param_request_read_send, sysid, compid, name.encode(), -1)

    def param_int(self, name):
        value = self.params.get(name)
        return int(value) if value is not None else 0

    def param_missing(self, name):
        return name in self.param_names and name not in self.params and not self.param_timer.isActive()

    def param_lookup(self, name):
        """(display text, color, value or None) for a parameter on the setup page."""
        if name in self.params:
            return format_param(self.params[name]), None, self.params[name]
        if not self.target:
            return "-", C["muted"], None
        if self.param_missing(name):
            return "n/a", C["muted"], None
        return "…", C["muted"], None

    def write_params(self, pairs):
        """Validate then write (name, text) pairs; returns a status text."""
        if not self.target:
            return "Not connected", C["err"]
        changes = []
        for name, text in pairs:
            if not text or self.param_missing(name):
                continue
            try:
                value = float(text)
            except ValueError:
                return f"Invalid value for {name}: {text}", C["err"]
            current = self.params.get(name)
            if current is not None and float(current) == value:
                continue
            changes.append((name, value))
        if not changes:
            return "Nothing to change", C["muted"]
        sysid, compid = self.target
        for name, value in changes:
            ptype = self.param_types.get(name, mavlink.MAV_PARAM_TYPE_REAL32)
            if self.autopilot == mavlink.MAV_AUTOPILOT_PX4 and ptype in INT_TYPES:
                raw = struct.unpack("<f", struct.pack("<i", int(value)))[0]
            else:
                raw = value
            self.send(self.conn.mav.param_set_send, sysid, compid, name.encode(), raw, ptype)
            self.params.pop(name, None)
            if name not in self.param_names:
                self.param_names.append(name)
        self.fetch_params()
        return f"Wrote {len(changes)} parameter(s), reboot to apply", C["ok"]

    def apply_peripheral(self, card):
        set_label(card.status, *self.write_params(card.new_values()))

    def reload_network(self):
        for name in NET_PARAMS:
            self.params.pop(name, None)
        self.net_loaded = False
        self.fetch_params()

    def apply_network(self):
        pairs = [("NET_ENABLE", str(int(self.net_enable.isChecked()))),
                 ("NET_DHCP", str(int(self.net_dhcp.isChecked())))]
        if not self.net_dhcp.isChecked():
            try:
                ip = ipaddress.IPv4Address(self.net_ip.text().strip())
                gw = ipaddress.IPv4Address(self.net_gw.text().strip())
                mask = self.net_mask.text().strip().lstrip("/")
                bits = int(mask) if mask.isdigit() else ipaddress.IPv4Network(f"0.0.0.0/{mask}").prefixlen
                if not 0 <= bits <= 32:
                    raise ValueError(mask)
            except ValueError as e:
                set_label(self.net_status, f"Invalid address: {e}", C["err"])
                return
            pairs += [(n, str(b)) for n, b in zip(NET_IP_PARAMS, ip.packed)]
            pairs += [(n, str(b)) for n, b in zip(NET_GW_PARAMS, gw.packed)]
            pairs.append(("NET_NETMASK", str(bits)))
        set_label(self.net_status, *self.write_params(pairs))

    def load_network(self):
        if self.net_loaded or any(n not in self.params for n in NET_PARAMS[:-len(NET_MAC_PARAMS)]):
            return
        self.net_loaded = True
        self.net_enable.setChecked(bool(self.param_int("NET_ENABLE")))
        self.net_dhcp.setChecked(bool(self.param_int("NET_DHCP")))
        self.net_ip.setText(".".join(str(self.param_int(n)) for n in NET_IP_PARAMS))
        self.net_gw.setText(".".join(str(self.param_int(n)) for n in NET_GW_PARAMS))
        bits = max(0, min(32, self.param_int("NET_NETMASK")))
        self.net_mask.setText(str(ipaddress.IPv4Network(f"0.0.0.0/{bits}").netmask))
        if all(n in self.params for n in NET_MAC_PARAMS):
            self.net_mac.setText(":".join(f"{self.param_int(n):02X}" for n in NET_MAC_PARAMS))
        self.update_net_fields()

    # ---- board info helpers -----------------------------------------------------------------

    def device_id(self, kind, inst):
        table = PX4_DEVICE_PARAMS if self.autopilot == mavlink.MAV_AUTOPILOT_PX4 else AP_DEVICE_PARAMS
        names = table.get(kind, [])
        return self.param_int(names[inst]) if inst < len(names) else 0

    def devices(self, kind):
        ardupilot = self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        return [decode_device_id(dev, kind, ardupilot) for dev in (self.device_id(kind, i) for i in range(3)) if dev]

    def fresh(self, kind):
        now = time.monotonic()
        return any(k == kind and now - t <= STALE_S for (k, _), (_, _, t) in self.live.items())

    def gps_ports(self):
        ports = []
        for i in range(1, 9):
            if self.param_int(f"SERIAL{i}_PROTOCOL") == SERIAL_PROTOCOL_GPS:
                code = self.param_int(f"SERIAL{i}_BAUD")
                baud = SERIAL_BAUDS.get(code, code)
                ports.append(f"SERIAL{i} @ {baud}" if baud else f"SERIAL{i}")
        return ports

    def gps_type(self, inst):
        names = ("GPS1_TYPE", "GPS_TYPE") if inst == 0 else ("GPS2_TYPE", "GPS_TYPE2")
        for name in names:
            if name in self.params:
                return int(self.params[name])
        return None

    def sys_status_state(self, bit):
        """present / enabled / healthy flags for a SYS_STATUS sensor bit, or None before SYS_STATUS."""
        if not self.sys_status:
            return None
        present, enabled, healthy = self.sys_status
        return bool(present & bit), bool(enabled & bit), bool(healthy & bit)

    # ---- incoming messages ------------------------------------------------------------------

    def set_live(self, kind, inst, msg, text):
        self.live[(kind, inst)] = (msg, text, time.monotonic())

    def set_value(self, key, label, text):
        self.values[key] = (label, text, time.monotonic())

    def handle_message(self, msg):
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            return
        if mtype == "HEARTBEAT" and msg.type == mavlink.MAV_TYPE_GCS:
            return
        if self.target and mtype != "HEARTBEAT" and msg.get_srcSystem() != self.target[0]:
            return
        handler = getattr(self, f"on_{mtype}", None)
        if handler:
            handler(msg)

    def on_HEARTBEAT(self, m):
        if m.autopilot == mavlink.MAV_AUTOPILOT_INVALID:
            return
        if self.target and m.get_srcSystem() != self.target[0]:
            return
        first = self.target is None
        self.target = (m.get_srcSystem(), m.get_srcComponent())
        self.autopilot = m.autopilot
        self.last_heartbeat = time.monotonic()
        try:
            mode = mavutil.mode_string_v10(m)
        except Exception:
            mode = str(m.custom_mode)
        self.hb = {
            "autopilot": AUTOPILOT_NAMES.get(m.autopilot) or enum_name("MAV_AUTOPILOT", m.autopilot, "MAV_AUTOPILOT_"),
            "vehicle": enum_name("MAV_TYPE", m.type, "MAV_TYPE_"),
            "state": enum_name("MAV_STATE", m.system_status, "MAV_STATE_"),
            "armed": bool(m.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
            "mode": mode,
        }
        if first:
            self.notify(f"Board found: system {self.target[0]}, {self.hb['autopilot']}")
            self.on_board_found()

    def on_AUTOPILOT_VERSION(self, m):
        v = m.flight_sw_version
        types = {0: "dev", 64: "alpha", 128: "beta", 192: "rc", 255: "official"}
        self.version["fw_number"] = f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF} {types.get(v & 0xFF, '')}".strip()
        self.version["board_id"] = f"{m.board_version >> 16}" if m.board_version else "-"
        uid2 = bytes(getattr(m, "uid2", []) or [])
        if uid2.strip(b"\x00"):
            self.version["uid"] = uid2.rstrip(b"\x00").hex().upper()
        elif m.uid:
            self.version["uid"] = f"{m.uid:016X}"
        if m.vendor_id or m.product_id:
            self.version["usb"] = f"{m.vendor_id:04X}:{m.product_id:04X}"

    def on_STATUSTEXT(self, m):
        text = m.text.strip()
        board = re.match(r"^(\S+)\s+([0-9A-F]{8})\s+([0-9A-F]{8})\s+([0-9A-F]{8})$", text)
        if board:
            self.version["board"] = board.group(1)
            self.version.setdefault("uid", "".join(board.group(2, 3, 4)))
        elif re.match(r"^(Ardu\w+|Blimp|AP_Periph)\s+V\d", text):
            self.version["firmware"] = text
        elif text.startswith("Frame:"):
            self.version["frame"] = text.split(":", 1)[1].strip()
        if "IOMCU" in text.upper():
            self.iomcu_msg = (m.severity, text)
        severity = enum_name("MAV_SEVERITY", m.severity, "MAV_SEVERITY_")
        self.status_log.append((time.strftime("%H:%M:%S"), m.severity, severity, text))
        del self.status_log[:-STATUS_LOG_MAX]
        self.status_seq += 1
        self.notify(f"[{severity}] {text}")

    def on_PARAM_VALUE(self, m):
        name = m.param_id.rstrip("\x00") if isinstance(m.param_id, str) else m.param_id.decode().rstrip("\x00")
        value = m.param_value
        if self.autopilot == mavlink.MAV_AUTOPILOT_PX4 and m.param_type in INT_TYPES:
            value = struct.unpack("<i", struct.pack("<f", value))[0]
        elif m.param_type in INT_TYPES:
            value = int(round(value))
        self.params[name] = value
        self.param_types[name] = m.param_type

    def on_SYS_STATUS(self, m):
        self.sys_status = (m.onboard_control_sensors_present, m.onboard_control_sensors_enabled,
                           m.onboard_control_sensors_health)
        self.power = {
            "voltage": m.voltage_battery / 1000 if m.voltage_battery != 0xFFFF else None,
            "current": m.current_battery / 100 if m.current_battery >= 0 else None,
            "remaining": m.battery_remaining if m.battery_remaining >= 0 else None,
            "load": m.load / 10,
            "drop": m.drop_rate_comm / 100,
            "t": time.monotonic(),
        }

    def on_POWER_STATUS(self, m):
        self.power_status = {"vcc": m.Vcc / 1000, "vservo": m.Vservo / 1000, "flags": m.flags,
                             "t": time.monotonic()}

    def on_MCU_STATUS(self, m):
        self.mcu = {"temp": m.MCU_temperature / 100, "v": m.MCU_voltage / 1000,
                    "vmin": m.MCU_voltage_min / 1000, "vmax": m.MCU_voltage_max / 1000, "t": time.monotonic()}

    def on_HWSTATUS(self, m):
        self.hwstatus = {"vcc": m.Vcc / 1000, "i2cerr": m.I2Cerr, "t": time.monotonic()}

    def on_SYSTEM_TIME(self, m):
        self.version["uptime"] = format_uptime(m.time_boot_ms)

    def imu(self, inst, msg, m, unit_acc, unit_gyro, unit_mag):
        self.set_live("accel", inst, msg, xyz(m.xacc, m.yacc, m.zacc, unit_acc))
        self.set_live("gyro", inst, msg, xyz(m.xgyro, m.ygyro, m.zgyro, unit_gyro))
        if (m.xmag, m.ymag, m.zmag) != (0, 0, 0):
            self.set_live("mag", inst, msg, xyz(m.xmag, m.ymag, m.zmag, unit_mag))
        temp = getattr(m, "temperature", 0)
        if temp:
            self.set_live("temp", inst, msg, f"{temp / 100:.1f} C")

    def on_RAW_IMU(self, m):
        self.imu(getattr(m, "id", 0), "RAW_IMU", m, "raw", "raw", "raw")

    def on_SCALED_IMU(self, m):
        self.imu(0, "SCALED_IMU", m, "mG", "mrad/s", "mGauss")

    def on_SCALED_IMU2(self, m):
        self.imu(1, "SCALED_IMU2", m, "mG", "mrad/s", "mGauss")

    def on_SCALED_IMU3(self, m):
        self.imu(2, "SCALED_IMU3", m, "mG", "mrad/s", "mGauss")

    def on_HIGHRES_IMU(self, m):
        inst = getattr(m, "id", 0)
        self.set_live("accel", inst, "HIGHRES_IMU", xyz(m.xacc, m.yacc, m.zacc, "m/s2", "{:.2f}"))
        self.set_live("gyro", inst, "HIGHRES_IMU", xyz(m.xgyro, m.ygyro, m.zgyro, "rad/s", "{:.3f}"))
        self.set_live("mag", inst, "HIGHRES_IMU", xyz(m.xmag, m.ymag, m.zmag, "Gauss", "{:.3f}"))
        self.set_live("baro", inst, "HIGHRES_IMU", f"{m.abs_pressure:.2f} hPa   {m.temperature:.1f} C")

    def pressure(self, inst, msg, m):
        self.set_live("baro", inst, msg, f"{m.press_abs:.2f} hPa   {m.temperature / 100:.1f} C")
        if m.press_diff:
            self.set_live("airspeed", inst, msg, f"{m.press_diff:.3f} hPa")

    def on_SCALED_PRESSURE(self, m):
        self.pressure(0, "SCALED_PRESSURE", m)

    def on_SCALED_PRESSURE2(self, m):
        self.pressure(1, "SCALED_PRESSURE2", m)

    def on_SCALED_PRESSURE3(self, m):
        self.pressure(2, "SCALED_PRESSURE3", m)

    def gps(self, inst, msg, m):
        self.gps_info[inst] = {
            "fix": GPS_FIX.get(m.fix_type, str(m.fix_type)), "fix_type": m.fix_type,
            "sats": m.satellites_visible if m.satellites_visible != 255 else None,
            "hdop": m.eph / 100 if m.eph != 0xFFFF else None,
            "lat": m.lat / 1e7, "lon": m.lon / 1e7, "alt": m.alt / 1000, "t": time.monotonic(),
        }
        self.set_live("gps", inst, msg, self.gps_info[inst]["fix"])

    def on_GPS_RAW_INT(self, m):
        self.gps(0, "GPS_RAW_INT", m)

    def on_GPS2_RAW(self, m):
        self.gps(1, "GPS2_RAW", m)

    def on_ATTITUDE(self, m):
        self.attitude = (math.degrees(m.roll), math.degrees(m.pitch), math.degrees(m.yaw) % 360, time.monotonic())

    def on_VFR_HUD(self, m):
        self.set_value("hud", "Speed / Altitude",
                       f"Airspeed {m.airspeed:.1f} m/s   Groundspeed {m.groundspeed:.1f} m/s   "
                       f"Alt {m.alt:.1f} m   Heading {m.heading}")

    def on_DISTANCE_SENSOR(self, m):
        self.set_live("range", m.id, "DISTANCE_SENSOR",
                      f"{m.current_distance} cm   (range {m.min_distance}-{m.max_distance} cm)")

    def on_RANGEFINDER(self, m):
        if ("range", 0) not in self.live or self.live[("range", 0)][0] == "RANGEFINDER":
            self.set_live("range", 0, "RANGEFINDER", f"{m.distance * 100:.0f} cm   {m.voltage:.2f} V")

    def on_OPTICAL_FLOW(self, m):
        self.set_live("flow", 0, "OPTICAL_FLOW", f"X {m.flow_x}   Y {m.flow_y}   Quality {m.quality}")

    def on_BATTERY_STATUS(self, m):
        cells = [v for v in m.voltages if v not in (0, 0xFFFF)]
        volt = f"{sum(cells) / 1000:.2f} V" if cells else "-"
        curr = f"{m.current_battery / 100:.2f} A" if m.current_battery >= 0 else "- A"
        self.set_value(f"battery{m.id + 1}", f"Battery {m.id + 1}", f"{volt}   {curr}   {m.battery_remaining} %")

    def on_RC_CHANNELS(self, m):
        chans = [getattr(m, f"chan{i}_raw") for i in range(1, min(m.chancount, 18) + 1)]
        self.rc = {"chans": chans, "rssi": m.rssi, "t": time.monotonic()}

    def on_SERVO_OUTPUT_RAW(self, m):
        if getattr(m, "port", 0) != 0:
            return
        outs = [getattr(m, f"servo{i}_raw", 0) for i in range(1, 17)]
        while len(outs) > 8 and not outs[-1]:
            outs.pop()
        self.servo = outs

    def on_EKF_STATUS_REPORT(self, m):
        self.set_value("ekf", "EKF Variances",
                       f"Vel {m.velocity_variance:.2f}   Pos {m.pos_horiz_variance:.2f}   "
                       f"Alt {m.pos_vert_variance:.2f}   Mag {m.compass_variance:.2f}")

    # ---- refresh ----------------------------------------------------------------------------

    def refresh(self):
        self.update_status_pill()
        self.update_param_labels()
        if self.stack.currentIndex() == 0:
            self.refresh_board()
            self.refresh_live()
            self.refresh_status_log()
        else:
            self.refresh_setup()

    def update_param_labels(self):
        if self.param_names:
            got = sum(1 for n in self.param_names if n in self.params)
            text = (f"Reading parameters {got}/{len(self.param_names)}..." if self.param_timer.isActive()
                    else f"{got} of {len(self.param_names)} parameters found")
        elif self.target:
            text = "Parameters not available for this autopilot"
        else:
            text = ""
        for label in (self.param_label, self.setup_param_label):
            if label.text() != text:
                label.setText(text)

    def refresh_board(self):
        b = self.board_card
        hb = self.hb
        v = self.version
        if hb:
            b.set("Board", v.get("board", "-"))
            b.set("Firmware", v.get("firmware", v.get("fw_number", "-")))
            b.set("Autopilot", hb["autopilot"])
            b.set("Vehicle", hb["vehicle"])
            b.set("Frame", v.get("frame", "-"))
            b.set("Flight Mode", hb["mode"])
            b.set("Armed", "ARMED" if hb["armed"] else "Disarmed", C["err"] if hb["armed"] else C["ok"])
            b.set("System State", hb["state"])
            b.set("System / Component", f"{self.target[0]} / {self.target[1]}")
            b.set("Board ID", v.get("board_id", "-"))
            b.set("UID", v.get("uid", "-"))
            b.set("Uptime", v.get("uptime", "-"))
        if self.rc:
            rssi = self.rc["rssi"]
            self.rc_card.set_values(self.rc["chans"], f"RSSI {rssi}" if rssi != 255 else "")
        if self.servo:
            self.servo_card.set_values(self.servo)

    def live_rows(self):
        """Rows of (label, chip, bus, address, value, value color, style); style is group, child or single."""
        now = time.monotonic()
        ardupilot = self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        empty = {"bus": "-", "address": "-", "chip": "-"}
        rows = []

        def live(key):
            entry = self.live.get(key)
            if not entry:
                return "No data", C["err"]
            return entry[1], C["err"] if now - entry[2] > STALE_S else None

        def timed(t, color=None):
            return C["err"] if now - t > STALE_S else color

        for inst in range(3):
            devs = {kind: self.device_id(kind, inst) for kind in ("accel", "gyro")}
            parts = [(kind, label) for kind, label in IMU_PARTS if devs.get(kind) or (kind, inst) in self.live]
            if not parts:
                continue
            infos = [decode_device_id(dev, kind, ardupilot) for kind, dev in devs.items() if dev] or [empty]
            chip, bus, addr = (" / ".join(dict.fromkeys(i[field] for i in infos))
                               for field in ("chip", "bus", "address"))
            rows.append((f"IMU {inst + 1}", chip, bus, addr, "", None, "group"))
            for kind, label in parts:
                rows.append((label, "", "", "", *live((kind, inst)), "child"))

        for kind in ("mag", "baro"):
            for inst in range(3):
                dev = self.device_id(kind, inst)
                if not dev and (kind, inst) not in self.live:
                    continue
                info = decode_device_id(dev, kind, ardupilot) if dev else empty
                rows.append((f"{KIND_LABELS[kind]} {inst + 1}", info["chip"], info["bus"], info["address"],
                             *live((kind, inst)), "single"))

        ports = self.gps_ports()
        for inst in range(2):
            gtype = self.gps_type(inst)
            info = self.gps_info.get(inst)
            if not gtype and not info:
                continue
            chip = GPS_TYPES.get(gtype, f"Type {gtype}") if gtype is not None else "-"
            if gtype == 9:
                bus, port = "DroneCAN", "-"
            else:
                bus = "Serial" if gtype is not None else "-"
                port = ports[inst] if inst < len(ports) else "-"
            if not info:
                rows.append((f"GPS {inst + 1}", chip, bus, port, "No data", C["err"], "single"))
                continue
            rows.append((f"GPS {inst + 1}", chip, bus, port, "", None, "group"))
            t = info["t"]
            fix_color = C["ok"] if info["fix_type"] >= 3 else C["warn"] if info["fix_type"] == 2 else C["err"]
            rows.append(("Fix", "", "", "", info["fix"], timed(t, fix_color), "child"))
            for label, text in (
                    ("Satellites", str(info["sats"]) if info["sats"] is not None else "-"),
                    ("HDOP", f"{info['hdop']:.2f}" if info["hdop"] is not None else "-"),
                    ("Position", f"{info['lat']:.7f}, {info['lon']:.7f}"),
                    ("Altitude", f"{info['alt']:.1f} m")):
                rows.append((label, "", "", "", text, timed(t), "child"))

        extra_types = {"range": "RNGFND1_TYPE", "flow": "FLOW_TYPE", "airspeed": "ARSPD_TYPE"}
        for kind, param in extra_types.items():
            instances = sorted({i for k, i in self.live if k == kind})
            if not instances and self.param_int(param):
                instances = [0]
            for inst in instances:
                ptype = self.param_int(param) if inst == 0 else 0
                rows.append((f"{KIND_LABELS[kind]} {inst + 1}", f"Type {ptype}" if ptype else "-", "-", "-",
                             *live((kind, inst)), "single"))

        if self.attitude:
            r, p, y, t = self.attitude
            rows.append(("Attitude", "", "", "", "", None, "group"))
            for label, value in (("Roll", f"{r:.1f}°"), ("Pitch", f"{p:.1f}°"), ("Yaw", f"{y:.1f}°")):
                rows.append((label, "", "", "", value, timed(t), "child"))

        pw = self.power
        if pw:
            t = pw["t"]
            rows.append(("Power", "", "", "", "", None, "group"))
            for label, text, color in (
                    ("Voltage", f"{pw['voltage']:.2f} V" if pw["voltage"] is not None else "-", None),
                    ("Current", f"{pw['current']:.2f} A" if pw["current"] is not None else "-", None),
                    ("Remaining", f"{pw['remaining']} %" if pw["remaining"] is not None else "-", None),
                    ("CPU Load", f"{pw['load']:.1f} %", C["warn"] if pw["load"] > 80 else None),
                    ("Link Loss", f"{pw['drop']:.1f} %", C["err"] if pw["drop"] > 5 else None)):
                rows.append((label, "", "", "", text, timed(t, color), "child"))

        for key in sorted(self.values, key=lambda k: self.values[k][0]):
            label, text, t = self.values[key]
            rows.append((label, "", "", "", text, timed(t), "single"))
        return rows

    def refresh_live(self):
        rows = self.live_rows()
        table = self.live_table
        table.setRowCount(len(rows))
        fit_table_height(table)
        for i, (label, chip, bus, addr, value, color, style) in enumerate(rows):
            group = style == "group"
            if style == "child":
                set_cell(table, i, 0, "      " + label, C["muted"])
            else:
                set_cell(table, i, 0, label, bold=group)
            set_cell(table, i, 1, chip, bold=group)
            set_cell(table, i, 2, bus)
            set_cell(table, i, 3, addr)
            set_cell(table, i, 4, value, color)

    def refresh_status_log(self):
        max_severity = self.severity_box.currentData()
        if self.status_shown == (self.status_seq, max_severity):
            return
        self.status_shown = (self.status_seq, max_severity)
        entries = [e for e in self.status_log if e[1] <= max_severity]
        table = self.status_table
        table.setRowCount(len(entries))
        for i in range(len(entries)):
            stamp, level, severity, text = entries[i]
            if level <= mavlink.MAV_SEVERITY_ERROR:
                color = C["err"]
            elif level == mavlink.MAV_SEVERITY_WARNING:
                color = C["warn"]
            elif level == mavlink.MAV_SEVERITY_DEBUG:
                color = C["muted"]
            else:
                color = None
            set_cell(table, i, 0, stamp, C["muted"])
            set_cell(table, i, 1, severity, color)
            set_cell(table, i, 2, text, color)
        table.scrollToBottom()

    # ---- setup page -------------------------------------------------------------------------

    def detect_peripheral(self, key):
        mags = {d["name"] for d in self.devices("mag")}
        imus = {d["name"] for d in self.devices("accel")}
        if key == "hflow":
            found = self.fresh("flow") and self.param_int("FLOW_TYPE") == 6
        elif key == "hmc5883":
            found = any(name and name.startswith("HMC5883") for name in mags)
        elif key == "tfluna":
            found = self.fresh("range") and self.param_int("RNGFND1_TYPE") == 20
        elif key == "pmw3901":
            found = self.fresh("flow") and self.param_int("FLOW_TYPE") == 4
        elif key == "mpu9250":
            found = "MPU9250" in imus or "AK8963" in mags
        else:
            found = False
        if not self.target:
            return "-", C["muted"]
        return ("● Detected", C["ok"]) if found else ("● Not detected", C["muted"])

    def onboard_rows(self):
        """Rows of (item, interface, status, status color, details)."""
        now = time.monotonic()
        ok, warn, err, muted = C["ok"], C["warn"], C["err"], C["muted"]
        rows = []

        def recent(data):
            return data is not None and now - data["t"] <= STALE_S

        def voltage_state(v, low, high):
            return ("OK", ok) if low <= v <= high else ("Out of range", err)

        ps = self.power_status if recent(self.power_status) else None
        vcc = ps["vcc"] if ps else (self.hwstatus["vcc"] if recent(self.hwstatus) else None)
        if vcc:
            rows.append(("Scaled 5V", "ADC", *voltage_state(vcc, 4.5, 5.5), f"{vcc:.2f} V"))
        else:
            rows.append(("Scaled 5V", "ADC", "No data", muted, "Not reported (POWER_STATUS / HWSTATUS)"))

        mcu = self.mcu if recent(self.mcu) else None
        if mcu:
            rows.append(("ADC1 3V3", "ADC", *voltage_state(mcu["v"], 3.1, 3.5),
                         f"{mcu['v']:.2f} V   (min {mcu['vmin']:.2f} V, max {mcu['vmax']:.2f} V)"))
            rows.append(("MCU", "Internal", "OK" if mcu["temp"] < 85 else "Hot", ok if mcu["temp"] < 85 else warn,
                         f"{mcu['temp']:.1f} C"))
        else:
            rows.append(("ADC1 3V3", "ADC", "No data", muted, "Not reported (MCU_STATUS)"))

        if ps:
            flags = ps["flags"]
            oc = flags & mavlink.MAV_POWER_STATUS_PERIPH_OVERCURRENT
            hp_oc = flags & mavlink.MAV_POWER_STATUS_PERIPH_HIPOWER_OVERCURRENT
            status = ("Overcurrent", err) if oc or hp_oc else ("OK", ok)
            details = [f"nOC {'active' if oc else 'clear'}", f"Hi-power nOC {'active' if hp_oc else 'clear'}",
                       f"Brick {'OK' if flags & mavlink.MAV_POWER_STATUS_BRICK_VALID else 'invalid'}",
                       f"Servo rail {ps['vservo']:.2f} V" if flags & mavlink.MAV_POWER_STATUS_SERVO_VALID
                       else "Servo rail not powered",
                       "USB connected" if flags & mavlink.MAV_POWER_STATUS_USB_CONNECTED else "USB not connected"]
            rows.append(("Peripheral power nEN / nOC", "GPIO", *status, "   ".join(details)))
        else:
            rows.append(("Peripheral power nEN / nOC", "GPIO", "No data", muted, "Not reported (POWER_STATUS)"))

        batt = self.param_lookup("BATT_MONITOR")[2]
        pw = self.power
        reading = f"{pw['voltage']:.2f} V   {pw['current']:.2f} A" if pw and pw["voltage"] is not None \
            and pw["current"] is not None else "No reading"
        if batt is None:
            rows.append(("Onboard INA238", "I2C", "Unknown", muted, "BATT_MONITOR not read"))
        elif int(batt) == 21:
            rows.append(("Onboard INA238", "I2C", "Configured", ok, f"BATT_MONITOR = 21 (INA2xx)   {reading}"))
        else:
            rows.append(("Onboard INA238", "I2C", "Not selected", warn, f"BATT_MONITOR = {int(batt)} (INA2xx is 21)"))

        ardupilot = self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        baro_inst = next((i for i in range(3) if self.device_id("baro", i) and
                          decode_device_id(self.device_id("baro", i), "baro", ardupilot)["name"] == "BMP390"), None)
        if baro_inst is not None:
            baro = decode_device_id(self.device_id("baro", baro_inst), "baro", ardupilot)
            value = self.live.get(("baro", baro_inst))
            rows.append(("Onboard Baro BMP390", "I2C2", "Detected", ok,
                         f"{baro['bus']}   {baro['address']}" + (f"   {value[1]}" if value else "")))
        else:
            rows.append(("Onboard Baro BMP390", "I2C2", "Not detected", err if self.devices("baro") else muted,
                         "No BMP390 in BARO*_DEVID"))

        io_enable = self.param_lookup("BRD_IO_ENABLE")[2]
        if io_enable is None:
            io = ("Unknown", muted, "BRD_IO_ENABLE not available")
        elif not int(io_enable):
            io = ("Disabled", muted, "BRD_IO_ENABLE = 0")
        elif self.iomcu_msg and self.iomcu_msg[0] <= mavlink.MAV_SEVERITY_WARNING:
            io = ("Error", err, self.iomcu_msg[1])
        else:
            io = ("Enabled", ok, self.iomcu_msg[1] if self.iomcu_msg else "BRD_IO_ENABLE = 1, no IOMCU errors reported")
        rows.append(("IO Communication", "USART6 / IO_STATUS", *io))

        rc_state = self.sys_status_state(mavlink.MAV_SYS_STATUS_SENSOR_RC_RECEIVER)
        rc = self.rc if recent(self.rc) else None
        if rc and rc["chans"] and (not rc_state or rc_state[2]):
            rssi = f"   RSSI {rc['rssi']}" if rc["rssi"] != 255 else ""
            rows.append(("SBUS RC", "Serial", "Receiving", ok, f"{len(rc['chans'])} channels{rssi}"))
        elif rc_state and rc_state[0]:
            rows.append(("SBUS RC", "Serial", "No signal", err, "RC receiver present but unhealthy"))
        else:
            rows.append(("SBUS RC", "Serial", "No signal", muted, "No RC_CHANNELS data"))

        log_state = self.sys_status_state(mavlink.MAV_SYS_STATUS_LOGGING)
        if log_state is None:
            rows.append(("SD Card", "SDMMC", "Unknown", muted, "Waiting for SYS_STATUS"))
        elif not log_state[0]:
            rows.append(("SD Card", "SDMMC", "Not reported", muted, "Logging disabled"))
        elif log_state[2]:
            rows.append(("SD Card", "SDMMC", "Present", ok, "Logging healthy"))
        else:
            rows.append(("SD Card", "SDMMC", "Missing / error", err, "Logging unhealthy (no card or write error)"))

        proto = self.param_lookup(f"SERIAL{CM5_SERIAL_PORT}_PROTOCOL")[2]
        baud = self.param_lookup(f"SERIAL{CM5_SERIAL_PORT}_BAUD")[2]
        if proto is None:
            rows.append(("CM5 Serial", "USART2 / TELEM3", "Unknown", muted, f"SERIAL{CM5_SERIAL_PORT} not read"))
        else:
            name = SERIAL_PROTOCOLS.get(int(proto), f"Protocol {int(proto)}")
            rate = SERIAL_BAUDS.get(int(baud), baud) if baud is not None else "-"
            good = int(proto) in (1, 2, 45)
            rows.append(("CM5 Serial", "USART2 / TELEM3", "Configured" if good else name, ok if good else warn,
                         f"SERIAL{CM5_SERIAL_PORT}: {name} @ {rate}"))

        params_ok = any(n in self.params for n in self.param_names)
        rows.append(("Onboard EEPROM AT24C02D", "I2C3", "Not reported", muted, "No MAVLink status for this device"))
        rows.append(("FRAM FM25V02A", "SPI5", "Params readable" if params_ok else "Unknown", ok if params_ok else muted,
                     "Parameter storage (indirect check)"))
        return rows

    def refresh_setup(self):
        rows = self.onboard_rows()
        table = self.onboard_table
        table.setRowCount(len(rows))
        fit_table_height(table)
        for i, (item, iface, status, color, details) in enumerate(rows):
            set_cell(table, i, 0, item, bold=True)
            set_cell(table, i, 1, iface, C["muted"])
            set_cell(table, i, 2, status, color)
            set_cell(table, i, 3, details)

        for card in self.peripheral_cards:
            card.refresh(self.param_lookup, self.detect_peripheral(card.spec["detect"]))

        self.load_network()
        if self.param_missing("NET_ENABLE"):
            set_label(self.net_status, "Board has no NET_* parameters (no Ethernet support)", C["muted"])

    def closeEvent(self, event):
        self.disconnect_board()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
