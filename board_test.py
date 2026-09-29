import datetime
import html
import ipaddress
import logging
import math
import os
import re
import signal
import struct
import sys
import time
import traceback

os.environ.setdefault("MAVLINK20", "1")

import serial.tools.list_ports
from pymavlink import mavutil
from PySide6.QtCore import QMarginsF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPageLayout, QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFrame,
    QGridLayout,
    QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QAbstractSpinBox, QScrollArea, QSlider, QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

mavlink = mavutil.mavlink
file_log = logging.getLogger("board_test")

APP_NAME = "Board Test"
BAUD_RATES = ["9600", "57600", "115200", "230400", "460800", "921600"]
DEFAULT_BAUD = "57600"
REFRESH_MS = 250
STALE_S = 2.0
HEARTBEAT_TIMEOUT_S = 3.0
PARAM_RETRY_MS = 2000
PARAM_MAX_TRIES = 3
PORT_SCAN_MS = 1500
STATUS_LOG_MAX = 1000
# Warnings, errors, test results and internal exceptions are also kept on disk (one file per day)
LOG_DIR = os.path.expanduser("~/board_test_logs")
PWM_MIN, PWM_MAX = 800, 2200
READER_STOP_TIMEOUT_MS = 2000
# Earth field strength limits used by ArduPilot's compass arming check (mGauss)
MAG_FIELD_MIN, MAG_FIELD_MAX = 185, 875

# Auto reboot / reconnect after settings are applied
PARAM_CONFIRM_S = 6.0      # wait for PARAM_VALUE echoes before rebooting
REBOOT_GRACE_MS = 800      # let the reboot command leave before closing the port
RECONNECT_DELAY_S = 2.0    # USB drops out during reboot, do not reopen it straight away
RECONNECT_TIMEOUT_S = 45.0
HEARTBEAT_WAIT_S = 8.0     # reopen the port if no heartbeat arrives (e.g. opened in the bootloader)
TEST_TIMEOUT_S = 25.0      # time a port test waits for data after the reboot
PARAM_LIST_IDLE_S = 1.5   # full parameter download: re-request missing indices after this pause
PARAM_LIST_TRIES = 5
PARAM_LIST_BATCH = 40     # missing parameters re-requested per round
PARAM_TABLE_HEIGHT = 520
MOTOR_TEST_MAX_THROTTLE = 40  # % throttle cap for bench tests

# ---- Lectron Pi5-H7 --------------------------------------------------------------------------
# hwdef: SERIAL_ORDER OTG1 UART7 UART5 USART1 UART8 USART2 UART4 USART3, IOMCU on USART6
BOARD_NAME = "Lectron Pi5-H7"
SERIAL_PORTS = {0: ("USB", "OTG1"), 1: ("TELEM1", "UART7"), 2: ("TELEM2", "UART5"), 3: ("GPS1", "USART1"),
                4: ("GPS2", "UART8"), 5: ("TELEM3", "USART2"), 6: ("EXTERNAL", "UART4"), 7: ("DEBUG", "USART3")}
CM5_SERIAL_PORT = 5  # TELEM3, bridged to CM5 UART3
TELEM_DEFAULT = {"PROTOCOL": 2, "BAUD": 57}  # MAVLink2 @ 57600
OUTPUT_NAMES = {i: f"MAIN {i} (IO)" if i <= 8 else f"AUX {i - 8} (FMU)" for i in range(1, 17)}
ONBOARD_IMUS = [("IMU1 ICM-42670-P", "SPI1", "ICM42670"), ("IMU2 ICM-42670-P", "SPI2", "ICM42670"),
                ("IMU3 BMI270", "SPI3", "BMI270")]
ONBOARD_BAROS = [("Baro1 BMP390", "I2C2"), ("Baro2 BMP390", "I2C4")]
BARO_390_NAMES = ("BMP390", "BMP388")  # the BMP388 driver also handles the BMP390

C = {
    "bg": "#0A0A0F",
    "surface": "#13131A",
    "surface2": "#1C1C26",
    "hover": "#24243A",
    "row_alt": "#17171F",
    "border": "rgba(255, 255, 255, 15)",
    "border_hover": "rgba(255, 255, 255, 31)",
    "text": "#F0F0F5",
    "muted": "#8888A0",
    "off": "#555566",
    "accent": "#3B82F6",
    "accent_hover": "#5B9BF7",
    "accent_dim": "rgba(59, 130, 246, 38)",
    "ok": "#22C55E",
    "warn": "#F59E0B",
    "err": "#EF4444",
}

STYLE = f"""
QWidget {{ color: {C['text']}; font-family: "Inter", "Segoe UI", "Roboto", sans-serif; font-size: 13px; }}
QMainWindow, #page, #central {{ background: {C['bg']}; }}
QLabel {{ background: transparent; }}
#topbar {{ background: {C['surface']}; border-bottom: 1px solid {C['border']}; }}
#sidebar {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
#logo {{ background: {C['accent']}; color: white; border-radius: 8px; font-weight: 800; font-size: 12px; }}
#appTitle {{ font-size: 15px; font-weight: 700; }}
#appSub, #pageSubtitle, #kvKey {{ color: {C['muted']}; }}
#pageTitle {{ font-size: 20px; font-weight: 700; }}
#card {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 10px; }}
#cardTitle {{ color: {C['muted']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
#badge {{ color: {C['accent']}; background: {C['accent_dim']}; border-radius: 4px; padding: 1px 6px; font-size: 11px; font-weight: 700; }}
#kvValue {{ font-family: "JetBrains Mono", "Fira Code", monospace; font-weight: 600; }}
#warning {{ color: {C['warn']}; background: rgba(245, 158, 11, 30); border-radius: 6px; padding: 8px 10px; }}
#bigText {{ font-size: 16px; font-weight: 700; }}
#sectionTitle {{ font-size: 17px; font-weight: 700; padding-top: 8px; }}
#sliderValue {{ font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 18px; font-weight: 700; color: {C['accent']}; }}
QSlider::groove:horizontal {{ height: 6px; background: {C['surface2']}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; width: 18px; height: 18px; margin: -7px 0; border-radius: 9px; }}
QSlider::handle:horizontal:disabled {{ background: {C['off']}; }}
QSlider::sub-page:horizontal:disabled {{ background: {C['off']}; }}
QDialog {{ background: {C['surface']}; }}
QPushButton {{ background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ background: {C['hover']}; border-color: {C['border_hover']}; }}
QPushButton:disabled {{ color: {C['off']}; background: {C['surface2']}; border-color: {C['border']}; }}
QPushButton#primary {{ background: {C['accent']}; color: white; border: none; font-weight: 600; padding: 7px 18px; }}
QPushButton#primary:hover {{ background: {C['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {C['surface2']}; color: {C['off']}; }}
QPushButton#primary[connected="true"] {{ background: {C['err']}; }}
QPushButton#danger {{ background: rgba(239, 68, 68, 38); color: {C['err']}; border: 1px solid rgba(239, 68, 68, 90); }}
QPushButton#danger:disabled {{ background: {C['surface2']}; color: {C['off']}; border-color: {C['border']}; }}
QPushButton#pass {{ color: {C['ok']}; }}
QPushButton#fail {{ color: {C['err']}; }}
QPushButton#nav {{ background: transparent; border: none; border-radius: 8px; padding: 0; }}
QPushButton#nav:hover {{ background: {C['hover']}; }}
QPushButton#nav:checked {{ background: {C['accent_dim']}; }}
QPushButton#tab {{ background: transparent; border: none; color: {C['muted']}; font-weight: 600; padding: 7px 16px; }}
QPushButton#tab:hover {{ color: {C['text']}; background: {C['hover']}; }}
QPushButton#tab:checked {{ color: {C['accent']}; background: {C['accent_dim']}; }}
QComboBox, QLineEdit, QSpinBox {{ background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 5px 8px; }}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{ border-color: {C['accent']}; }}
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled {{ color: {C['off']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {C['surface2']}; border: 1px solid {C['border']}; selection-background-color: {C['accent']}; selection-color: white; }}
QCheckBox {{ spacing: 8px; }}
QTableWidget {{ background: {C['surface']}; alternate-background-color: {C['row_alt']}; border: none; gridline-color: transparent; selection-background-color: {C['hover']}; selection-color: {C['text']}; }}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{ background: {C['surface']}; color: {C['muted']}; border: none; border-bottom: 1px solid {C['border']}; padding: 8px; font-weight: 700; }}
QStatusBar {{ background: {C['surface']}; color: {C['muted']}; border-top: 1px solid {C['border']}; padding-left: 12px; }}
QStatusBar QLabel {{ padding-left: 12px; }}
QToolTip {{ background: {C['surface2']}; color: {C['text']}; border: 1px solid {C['border']}; padding: 4px; }}
QProgressBar {{ background: {C['surface2']}; border: none; border-radius: 4px; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 4px; }}
QScrollArea {{ border: none; background: {C['bg']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['hover']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C['hover']}; border-radius: 4px; min-width: 30px; }}
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

SERIAL_PROTOCOLS = {
    -1: "None", 1: "MAVLink1", 2: "MAVLink2", 3: "FrSky D", 4: "FrSky SPort", 5: "GPS", 7: "Alexmos Gimbal",
    8: "Gimbal", 9: "Rangefinder", 10: "FrSky SPort Passthrough", 11: "Lidar360", 13: "Beacon",
    14: "Volz servo", 15: "SBus servo out", 16: "ESC Telemetry", 17: "Devo Telemetry", 18: "OpticalFlow",
    19: "Robotis Servo", 20: "NMEA Output", 21: "WindVane", 22: "SLCAN", 23: "RCIN", 24: "EFI",
    25: "LTM", 26: "RunCam", 27: "HoTT Telemetry", 28: "Scripting", 29: "Crossfire VTX", 30: "Generator",
    31: "Winch", 32: "MSP", 33: "DJI FPV", 34: "AirSpeed", 35: "ADSB", 36: "AHRS", 37: "SmartAudio",
    38: "FETtec OneWire", 39: "Torqeedo", 40: "AIS", 41: "CoDevESC", 42: "DisplayPort",
    43: "MAVLink High Latency", 44: "IRC Tramp", 45: "DDS/XRCE", 46: "IMU Data",
}

SERVO_FUNCTIONS = {-1: "GPIO", 0: "Disabled", 1: "RC Passthru",
                   **{33 + i: f"Motor {i + 1}" for i in range(8)},
                   **{82 + i: f"Motor {i + 9}" for i in range(4)},
                   **{51 + i: f"RCIN {i + 1}" for i in range(16)}}

FRAME_CLASSES = {0: "Undefined", 1: "Quad", 2: "Hexa", 3: "Octa", 4: "OctaQuad", 5: "Y6", 7: "Tri",
                 10: "Single", 11: "Coax", 12: "BiCopter", 13: "Heli Dual", 14: "DodecaHexa", 15: "HeliQuad",
                 16: "Deca", 17: "Scripting Matrix", 18: "6DoF Scripting", 19: "Dynamic Scripting Matrix"}
FRAME_TYPES = {0: "Plus", 1: "X", 2: "V", 3: "H", 4: "V-Tail", 5: "A-Tail", 10: "Y6B", 11: "Y6F",
               12: "BetaFlightX", 13: "DJIX", 14: "Clockwise X", 15: "I", 18: "BetaFlightX Reversed"}
MOT_PWM_TYPES = {0: "Normal PWM", 1: "OneShot", 2: "OneShot125", 3: "Brushed", 4: "DShot150", 5: "DShot300",
                 6: "DShot600", 7: "DShot1200", 8: "PWM Range"}
NET_PORT_TYPES = {0: "Disabled", 1: "UDP Client", 2: "UDP Server", 3: "TCP Client", 4: "TCP Server"}

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

# Devices for the board specific port tests. "check" is how the device is proven to work.
TEST_DEVICES = {
    "pmw3901": {"name": "PMW3901 optical flow", "protocol": 18, "baud": 19, "check": "flow",
                "params": [("FLOW_TYPE", 4)]},
    "tfluna": {"name": "TF-Luna rangefinder", "protocol": 9, "baud": 115, "check": "range",
               "params": [("RNGFND1_TYPE", 20), ("RNGFND1_ORIENT", 25)]},
    "hflow": {"name": "HFlow (DroneCAN)", "check": "flow",
              "params": [("CAN_P1_DRIVER", 1), ("CAN_D1_PROTOCOL", 1), ("FLOW_TYPE", 6),
                         ("RNGFND2_TYPE", 24), ("RNGFND2_ORIENT", 25)]},
    # GPS1_TYPE / GPS2_TYPE (4.6+) or GPS_TYPE / GPS_TYPE2 (older); whichever is missing is skipped
    "gps1": {"name": "GPS", "protocol": 5, "check": "gps1", "params": [("GPS1_TYPE", 1), ("GPS_TYPE", 1)]},
    "gps2": {"name": "GPS", "protocol": 5, "check": "gps2", "params": [("GPS2_TYPE", 1), ("GPS_TYPE2", 1)]},
    # external I2C compasses are only probed at boot, so this test always reboots
    "i2c_mag": {"name": "I2C compass", "check": "i2c_mag", "reboot": True, "params": []},
    "rc": {"name": "SBUS / PPM receiver", "check": "rc", "params": []},
    "usb": {"name": "USB cable", "check": "usb", "params": []},
    "sd": {"name": "microSD card", "check": "sd", "params": []},
    "eth": {"name": "Ethernet cable to this PC", "check": "eth", "params": []},
    "special": {"name": "Special module (manual check)", "check": None, "params": []},
    "manual": {"name": "Manual check", "check": None, "params": []},
}
TELEM_TEST_DEVICES = ["pmw3901", "tfluna"]

# FMU connectors of the Lectron Pi5-H7: designator -> (tag name, connector, [(pin, signal, voltage)])
CONNECTORS = {
    "PX_CAN": ("Pixhawk CAN", "BM04B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "CAN HIGH", "+3.3V"), (3, "CAN LOW", "+3.3V"), (4, "GROUND", "GND")]),
    "SBUS1": ("Pixhawk SBUS", "BM05B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "PPM INPUT", "+3.3V"), (3, "NC", "---"), (4, "RSSI IN", "+3.3V"),
        (5, "GROUND", "GND")]),
    "UART/I2C1": ("Pixhawk I2C3/UART4", "BM06B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "UART4 TX", "+3.3V"), (3, "UART4 RX", "+3.3V"), (4, "I2C3 SCL", "+3.3V"),
        (5, "I2C3 SDA", "+3.3V"), (6, "GROUND", "GND")]),
    "PX_SPI1": ("Pixhawk SPI6", "BM11B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "SPI6 SCK", "+3.3V"), (3, "SPI6 MISO(RX)", "+3.3V"),
        (4, "SPI6 MOSI(TX)", "+3.3V"), (5, "SPI6 NCS-1", "+3.3V"), (6, "SPI6 NCS-2", "+3.3V"),
        (7, "SPIX SYNC(A*)", "---"), (8, "SPI6 DRDY-1", "+3.3V"), (9, "SPI6 DRDY-2", "+3.3V"),
        (10, "SPI6 NRST", "+3.3V"), (11, "GROUND", "GND")]),
    "IO_PWM1": ("Pixhawk IO PWM (MAIN)", "BM10B-GHS", [
        (1, "VDD SERVO SENS", "0-16V"), *[(i + 2, f"IO PWM CH{i + 1}", "+3.3V") for i in range(8)],
        (10, "GROUND", "GND")]),
    "FMU_PWM1": ("Pixhawk FMU PWM (AUX)", "BM10B-GHS", [
        (1, "VDD SERVO SENS", "0-16V"), *[(i + 2, f"FMU PWM CH{i + 1}", "+3.3V") for i in range(8)],
        (10, "GROUND", "GND")]),
    "FMU_DEBUG1": ("Pixhawk FMU Debug", "SM10B-SRSS", [
        (1, "FMU VDD 3.3V", "+3.3V"), (2, "USART3_TX_DEBUG", "+3.3V"), (3, "USART3_RX_DEBUG", "+3.3V"),
        (4, "FMU_SWDIO", "+3.3V"), (5, "FMU_SWCLK", "+3.3V"), (6, "SPI6_SCK_EXTERNAL1", "+3.3V"),
        (7, "NFC_GPIO", "+3.3V"), (8, "PH11", "+3.3V"), (9, "FMU_NRST", "+3.3V"), (10, "GROUND", "GND")]),
    "IO_DEBUG1": ("Pixhawk IO Debug", "SM10B-SRSS", [
        (1, "IO VDD 3.3V", "+3.3V"), (2, "IO_USART1_TX_DEBUG", "+3.3V"), (3, "NC", "---"), (4, "IO_SWDIO", "+3.3V"),
        (5, "IO_SWCLK", "+3.3V"), (6, "IO_SWO", "+3.3V"), (7, "IO_SPARE_GPIO1", "+3.3V"),
        (8, "IO_SPARE_GPIO2", "+3.3V"), (9, "IO_NRST", "+3.3V"), (10, "GROUND", "GND")]),
    "PX_USB1": ("FMU USB", "USB 2.0 Type-C", [("-", "USB 2.0", "+5V")]),
    "CARD1": ("FMU SD Card", "TF SD card", [("-", "microSD (SDMMC2)", "+3.3V")]),
    "PX_I2C1": ("PX I2C2", "BM04B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "I2C2 SCL", "+3.3V"), (3, "I2C2 SDA", "+3.3V"), (4, "GROUND", "GND")]),
    "PX_TELEM2": ("Pixhawk Telemetry 2", "BM06B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "UART5 TX", "+3.3V"), (3, "UART5 RX", "+3.3V"), (4, "UART5 CTS", "+3.3V"),
        (5, "UART5 RTS", "+3.3V"), (6, "GROUND", "GND")]),
    "PX_TELEM1": ("Pixhawk Telemetry 1", "BM06B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "UART7 TX", "+3.3V"), (3, "UART7 RX", "+3.3V"), (4, "UART7 CTS", "+3.3V"),
        (5, "UART7 RTS", "+3.3V"), (6, "GROUND", "GND")]),
    "GPS1": ("Pixhawk GPS-1 (Full)", "BM10B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "USART1 TX", "+3.3V"), (3, "USART1 RX", "+3.3V"), (4, "I2C1 SCL", "+3.3V"),
        (5, "I2C1 SDA", "+3.3V"), (6, "SAFETY SWITCH IN", "+3.3V"), (7, "SAFETY LED OUT", "+3.3V"),
        (8, "FMU 3.3V", "+3.3V"), (9, "BUZZER", "+3.3V"), (10, "GROUND", "GND")]),
    "GPS2": ("Pixhawk GPS-2 (Basic)", "BM06B-GHS", [
        (1, "PERIPHERAL 5V", "+5V"), (2, "UART8 TX", "+3.3V"), (3, "UART8 RX", "+3.3V"), (4, "I2C2 SCL", "+3.3V"),
        (5, "I2C2 SDA", "+3.3V"), (6, "GROUND", "GND")]),
    "U42": ("Pixhawk Ethernet", "BM04B-GHS", [
        (1, "ETH TX-N", "---"), (2, "ETH TX-P", "---"), (3, "ETH RX-N", "---"), (4, "ETH RX-P", "---")]),
    "PWR_DATA1": ("Pixhawk External Power Monitoring", "BM04B-GHS", [
        (1, "SYSTEM 5V", "+5V"), (2, "I2C1 SCL", "+3.3V"), (3, "I2C1 SDA", "+3.3V"), (4, "GROUND", "GND")]),
    "TELEM3": ("CM5 bridge (no FMU connector)", "internal", [("-", "USART2 ↔ CM5 UART3", "+3.3V")]),
}

# One row per connector function. "serial" = SERIALn under test, "i2c_bus" = ArduPilot I2C bus index
# (I2C_ORDER I2C1 I2C2 I2C3 I2C4 -> 0..3), "hint" = what the operator checks for manual rows.
PORT_TESTS = [
    {"key": "telem1", "connector": "PX_TELEM1", "function": "UART7 · SERIAL1", "serial": 1, "devices": "telem"},
    {"key": "telem2", "connector": "PX_TELEM2", "function": "UART5 · SERIAL2", "serial": 2, "devices": "telem"},
    {"key": "uart4", "connector": "UART/I2C1", "function": "UART4 · SERIAL6", "serial": 6, "devices": "telem"},
    {"key": "i2c3", "connector": "UART/I2C1", "function": "I2C3", "i2c_bus": 2, "devices": ["i2c_mag", "manual"]},
    {"key": "telem3", "connector": "TELEM3", "function": "USART2 · SERIAL5", "serial": 5, "devices": "telem",
     "optional": True},
    {"key": "gps1", "connector": "GPS1", "function": "USART1 · SERIAL3", "serial": 3, "devices": ["gps1"]},
    {"key": "gps1_i2c", "connector": "GPS1", "function": "I2C1", "i2c_bus": 0, "devices": ["i2c_mag", "manual"]},
    {"key": "gps1_safety", "connector": "GPS1", "function": "Safety · LED · buzzer", "devices": ["manual"],
     "hint": "Press the safety switch: LED must change and the buzzer must beep"},
    {"key": "gps2", "connector": "GPS2", "function": "UART8 · SERIAL4", "serial": 4,
     "devices": ["gps2", "special"]},
    {"key": "gps2_i2c", "connector": "GPS2", "function": "I2C2", "i2c_bus": 1, "devices": ["i2c_mag", "manual"]},
    {"key": "px_i2c", "connector": "PX_I2C1", "function": "I2C2", "i2c_bus": 1, "devices": ["i2c_mag", "manual"]},
    {"key": "pwr_data", "connector": "PWR_DATA1", "function": "I2C1 · power monitor", "devices": ["manual"],
     "hint": "Connect the external I2C power module and check its voltage"},
    {"key": "can1", "connector": "PX_CAN", "function": "CAN1", "devices": ["hflow", "manual"]},
    {"key": "sbus", "connector": "SBUS1", "function": "PPM / SBUS in", "devices": ["rc", "manual"]},
    {"key": "spi6", "connector": "PX_SPI1", "function": "SPI6", "devices": ["manual"],
     "hint": "Connect the SPI6 test device and check it on the scope / its own output"},
    {"key": "io_pwm", "connector": "IO_PWM1", "function": "MAIN 1-8 (IO)", "devices": ["manual"],
     "hint": "Drive the outputs from the Motors page (Motor Test) or a servo tester"},
    {"key": "fmu_pwm", "connector": "FMU_PWM1", "function": "AUX 1-8 (FMU)", "devices": ["manual"],
     "hint": "Drive the outputs from the Motors page (Motor Test) or a servo tester"},
    {"key": "fmu_debug", "connector": "FMU_DEBUG1", "function": "USART3 · SWD", "devices": ["manual"],
     "hint": "Attach the debug probe and read the FMU (SWD) / debug console"},
    {"key": "io_debug", "connector": "IO_DEBUG1", "function": "IO USART1 · SWD", "devices": ["manual"],
     "hint": "Attach the debug probe and read the IO co-processor (SWD)"},
    {"key": "usb", "connector": "PX_USB1", "function": "USB 2.0", "devices": ["usb", "manual"]},
    {"key": "sd", "connector": "CARD1", "function": "SDMMC2", "devices": ["sd", "manual"]},
    {"key": "eth", "connector": "U42", "function": "Ethernet", "devices": ["eth", "manual"]},
]
TESTS_BY_KEY = {t["key"]: t for t in PORT_TESTS}

NET_IP_PARAMS = [f"NET_IPADDR{i}" for i in range(4)]
NET_GW_PARAMS = [f"NET_GWADDR{i}" for i in range(4)]
NET_MAC_PARAMS = [f"NET_MACADDR{i}" for i in range(6)]
NET_P1_IP_PARAMS = [f"NET_P1_IP{i}" for i in range(4)]
NET_CONFIG_PARAMS = ["NET_ENABLE", "NET_DHCP", "NET_NETMASK"] + NET_IP_PARAMS + NET_GW_PARAMS
NET_PORT_PARAMS = ["NET_P1_TYPE", "NET_P1_PROTOCOL", "NET_P1_PORT"] + NET_P1_IP_PARAMS
NET_PARAMS = NET_CONFIG_PARAMS + NET_MAC_PARAMS + NET_PORT_PARAMS

SERIAL_PARAMS = [f"SERIAL{i}_{field}" for i in SERIAL_PORTS for field in ("PROTOCOL", "BAUD")]
SERVO_PARAMS = [f"SERVO{i}_{field}" for i in range(1, 17) for field in ("FUNCTION", "MIN", "MAX")]
MOTOR_PARAMS = ["FRAME_CLASS", "FRAME_TYPE", "MOT_PWM_TYPE"]

AP_PARAMS = (
    ["GPS_TYPE", "GPS_TYPE2", "GPS1_TYPE", "GPS2_TYPE", "RNGFND1_TYPE", "FLOW_TYPE", "ARSPD_TYPE",
     "BATT_MONITOR", "BRD_IO_ENABLE"]
    + SERIAL_PARAMS + NET_PARAMS + SERVO_PARAMS + MOTOR_PARAMS
    + [name for dev in TEST_DEVICES.values() for name, _ in dev["params"]]
)


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
    table.verticalHeader().setDefaultSectionSize(36)
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
    style = f"color: {color};" if color else ""
    if label.styleSheet() != style:
        label.setStyleSheet(style)


def make_combo(options):
    combo = QComboBox()
    for value, name in options.items():
        combo.addItem(name, value)
    return combo


def set_combo_value(combo, value, fmt="Value {}"):
    value = int(value)
    index = combo.findData(value)
    if index < 0:
        combo.addItem(fmt.format(value), value)
        index = combo.count() - 1
    combo.setCurrentIndex(index)


def make_spin(low, high, value, suffix=""):
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    spin.setSuffix(suffix)
    spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
    return spin


def fixed_columns(table, widths):
    header = table.horizontalHeader()
    for col, width in widths.items():
        header.setSectionResizeMode(col, QHeaderView.Fixed)
        header.resizeSection(col, width)


def muted_label(text="", wrap=False):
    label = QLabel(text)
    label.setObjectName("kvKey")
    label.setWordWrap(wrap)
    return label


def available_ports():
    """Real serial ports only (Linux lists every /dev/ttyS* even without hardware)."""
    ports = [p for p in serial.tools.list_ports.comports() if p.hwid and p.hwid != "n/a"]
    return sorted(ports, key=lambda p: (p.vid is None, p.device))


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


class UdpProbe(QThread):
    """Ethernet port test: listen for the board's heartbeat on the UDP port it sends MAVLink to."""
    result = Signal(bool, str)

    def __init__(self, port, sysid, timeout):
        super().__init__()
        self.port = port
        self.sysid = sysid
        self.timeout = timeout
        self._running = True

    def run(self):
        try:
            conn = mavutil.mavlink_connection(f"udpin:0.0.0.0:{self.port}", source_system=255)
        except Exception as e:
            self.result.emit(False, f"Cannot listen on UDP {self.port}: {e}")
            return
        end = time.monotonic() + self.timeout
        ok, text = False, f"No MAVLink heartbeat on UDP port {self.port}"
        try:
            while self._running and time.monotonic() < end:
                m = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
                if m and m.get_srcSystem() == self.sysid and m.autopilot != mavlink.MAV_AUTOPILOT_INVALID:
                    address = getattr(conn, "last_address", None)
                    ok, text = True, "Heartbeat over Ethernet" + (f" from {address[0]}" if address else "")
                    break
        except Exception as e:
            text = f"UDP error: {e}"
        finally:
            conn.close()
        if self._running:
            self.result.emit(ok, text)

    def stop(self):
        self._running = False


class Card(QFrame):
    def __init__(self, title, badge=None):
        super().__init__()
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 14, 16, 16)
        self.body.setSpacing(10)
        head = QHBoxLayout()
        label = QLabel(title.upper())
        label.setObjectName("cardTitle")
        head.addWidget(label)
        if badge:
            b = QLabel(badge)
            b.setObjectName("badge")
            head.addWidget(b)
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
            k = muted_label(key)
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
        self.info = muted_label()
        self.head.addWidget(self.info)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(1, 1)
        self.empty = muted_label("No data")
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
            name = muted_label(f"{self.prefix}{i + 1}")
            bar = QProgressBar()
            bar.setRange(PWM_MIN, PWM_MAX)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
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


class NavButton(QPushButton):
    """Sidebar entry: glyph above a small label, like the MicoAir configurator."""

    def __init__(self, glyph, text):
        super().__init__()
        self.setObjectName("nav")
        self.setCheckable(True)
        self.setFixedSize(60, 58)
        self.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 7, 0, 7)
        lay.setSpacing(3)
        self.glyph = QLabel(glyph)
        self.text_label = QLabel(text)
        for label, size in ((self.glyph, 20), (self.text_label, 10)):
            label.setAlignment(Qt.AlignCenter)
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            label.setProperty("font_px", size)
            lay.addWidget(label)
        self.toggled.connect(self.update_colors)
        self.update_colors(False)

    def update_colors(self, checked):
        color = C["accent"] if checked else C["muted"]
        for label in (self.glyph, self.text_label):
            weight = 600 if label is self.text_label else 400
            label.setStyleSheet(f"color: {color}; font-size: {label.property('font_px')}px; font-weight: {weight};")


class ReportDialog(QDialog):
    """Asks who ran the test and which board it was before the PDF report is written."""

    def __init__(self, parent, name="", board_id=""):
        super().__init__(parent)
        self.setWindowTitle("Create Test Report")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(12)
        title = QLabel("Test Report")
        title.setObjectName("bigText")
        lay.addWidget(title)
        lay.addWidget(muted_label("The date and time are added automatically.", wrap=True))
        self.name = QLineEdit(name)
        self.name.setPlaceholderText("Name Surname")
        self.board_id = QLineEdit(board_id)
        self.board_id.setPlaceholderText("Serial number on the board label")
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.addWidget(muted_label("Name Surname"), 0, 0)
        form.addWidget(self.name, 0, 1)
        form.addWidget(muted_label("Board ID"), 1, 0)
        form.addWidget(self.board_id, 1, 1)
        lay.addLayout(form)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Create PDF")
        self.buttons.button(QDialogButtonBox.Ok).setObjectName("primary")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        for field in (self.name, self.board_id):
            field.textChanged.connect(self.update_ok)
        self.update_ok()

    def update_ok(self):
        ok = bool(self.name.text().strip() and self.board_id.text().strip())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def values(self):
        return self.name.text().strip(), self.board_id.text().strip()


class MainWindow(QMainWindow):
    PAGES = (("Board", "\u25a3"), ("Test", "\u2714"), ("Motors", "\u2742"), ("Settings", "\u2699"),
             ("Live Data", "\u223f"), ("Messages", "\u2630"))
    PAGE_BOARD, PAGE_TEST, PAGE_MOTORS, PAGE_SETTINGS, PAGE_LIVE, PAGE_MESSAGES = range(6)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {BOARD_NAME}")
        self.resize(1320, 880)

        self.conn = None
        self.reader = None
        self.closing = False
        self.port_device = None
        self.port_baud = None
        self.port_list = None
        self.job = None          # auto reboot / reconnect job, survives the reconnect
        self.eth_probe = None
        self.active_test = None
        self.tests = {t["key"]: {"state": "idle", "detail": "", "deadline": 0.0} for t in PORT_TESTS}
        self.port_backup = {}    # parameter values changed by port tests, for "Restore ports"
        self.cmd_labels = {}     # MAV_CMD -> label that shows its COMMAND_ACK
        self.status_log = []
        self.exception_counts = {}
        self.status_seq = 0
        self.reset_state()

        central = QWidget()
        central.setObjectName("central")
        central.setAttribute(Qt.WA_StyledBackground, True)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.build_topbar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.stack = QStackedWidget()
        body.addWidget(self.build_sidebar())
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)
        self.stack.addWidget(self.build_board_page())
        self.stack.addWidget(self.build_test_page())
        self.stack.addWidget(self.build_motors_page())
        self.stack.addWidget(self.build_settings_page())
        self.stack.addWidget(self.build_live_page())
        self.stack.addWidget(self.build_messages_page())
        self.nav_group.idClicked.connect(self.show_page)

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
        self.port_timer = QTimer(self)
        self.port_timer.setInterval(PORT_SCAN_MS)
        self.port_timer.timeout.connect(self.refresh_ports)
        self.job_timer = QTimer(self)
        self.job_timer.setInterval(REFRESH_MS)
        self.job_timer.timeout.connect(self.job_tick)
        self.job_timer.start()

        self.refresh_ports()
        self.set_connected_ui(False)

    # ---- layout -----------------------------------------------------------------------------

    def build_topbar(self):
        bar = QWidget()
        bar.setObjectName("topbar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        bar.setFixedHeight(58)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(10)

        logo = QLabel("BT")
        logo.setObjectName("logo")
        logo.setFixedSize(30, 30)
        logo.setAlignment(Qt.AlignCenter)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        sub = QLabel(BOARD_NAME)
        sub.setObjectName("appSub")
        titles.addWidget(title)
        titles.addWidget(sub)
        lay.addWidget(logo)
        lay.addLayout(titles)
        lay.addStretch()

        self.port_box = QComboBox()
        self.port_box.setMinimumWidth(320)
        self.baud_box = QComboBox()
        self.baud_box.addItems(BAUD_RATES)
        self.baud_box.setCurrentText(DEFAULT_BAUD)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("primary")
        self.connect_btn.setMinimumWidth(120)
        self.status_pill = QLabel()
        self.status_pill.setMinimumWidth(160)
        self.status_pill.setFixedHeight(30)
        self.status_pill.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.port_box)
        lay.addWidget(self.baud_box)
        lay.addWidget(self.connect_btn)
        lay.addSpacing(6)
        lay.addWidget(self.status_pill)
        self.connect_btn.clicked.connect(self.toggle_connection)
        return bar

    def build_sidebar(self):
        side = QWidget()
        side.setObjectName("sidebar")
        side.setAttribute(Qt.WA_StyledBackground, True)
        side.setFixedWidth(76)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(8, 10, 8, 10)
        lay.setSpacing(6)
        self.nav_group = QButtonGroup(self)
        for i, (name, glyph) in enumerate(self.PAGES):
            btn = NavButton(glyph, name)
            btn.setChecked(i == 0)
            self.nav_group.addButton(btn, i)
            lay.addWidget(btn, 0, Qt.AlignHCenter)
        lay.addStretch()
        return side

    def show_page(self, index):
        self.stack.setCurrentIndex(index)
        if self.conn:
            self.refresh()

    @staticmethod
    def page_layout(title, subtitle, scroll=True):
        page = QWidget()
        page.setObjectName("page")
        page.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)
        head = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("pageTitle")
        s = QLabel(subtitle)
        s.setObjectName("pageSubtitle")
        texts.addWidget(t)
        texts.addWidget(s)
        head.addLayout(texts)
        head.addStretch()
        lay.addLayout(head)
        if not scroll:
            return page, lay, head
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(page)
        return area, lay, head

    @staticmethod
    def scroll_body():
        body = QWidget()
        body.setObjectName("page")
        body.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(16)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(body)
        return area, lay

    def build_board_page(self):
        area, lay, head = self.page_layout("Board", "Board identity, RC input, motor outputs and onboard hardware")
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
        lay.addLayout(top)

        onboard = Card("Onboard Hardware", BOARD_NAME)
        self.onboard_table = make_table(["Item", "Interface", "Status", "Details"], 3, fit=True)
        onboard.body.addWidget(self.onboard_table)
        lay.addWidget(onboard)
        lay.addStretch()
        return area

    def build_live_page(self):
        area, lay, head = self.page_layout("Live Data", "Sensors, GPS, attitude and power from the board")
        self.param_label = muted_label()
        self.reload_btn = QPushButton("Reload IDs")
        self.reload_btn.clicked.connect(self.start_param_fetch)
        head.addWidget(self.param_label)
        head.addSpacing(10)
        head.addWidget(self.reload_btn)
        card = Card("Live Data")
        self.live_table = make_table(["Item", "Chip", "Bus", "Address / Port", "Value"], 4, fit=True)
        card.body.addWidget(self.live_table)
        lay.addWidget(card)
        lay.addStretch()
        return area

    def build_messages_page(self):
        page, lay, head = self.page_layout("Messages", "Status text, warnings and command results", scroll=False)
        self.severity_box = QComboBox()
        self.severity_box.addItem("All messages", mavlink.MAV_SEVERITY_DEBUG)
        self.severity_box.addItem("Warnings and errors", mavlink.MAV_SEVERITY_WARNING)
        self.severity_box.addItem("Errors only", mavlink.MAV_SEVERITY_ERROR)
        self.severity_box.currentIndexChanged.connect(lambda _: self.refresh_status_log())
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_status_log)
        log_label = QLabel(f"Log file: {log_path()}")
        log_label.setObjectName("pageSubtitle")
        log_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        head.addWidget(log_label)
        head.addWidget(self.severity_box)
        head.addWidget(clear_btn)
        card = Card("Messages")
        self.status_table = make_table(["Time", "Severity", "Message"], 2)
        card.body.addWidget(self.status_table)
        lay.addWidget(card, 1)
        return page

    # ---- all parameters (settings page) -------------------------------------------------------

    def build_params_section(self, lay):
        card = Card("All Parameters")
        self.param_load_label = muted_label()
        self.param_load_btn = QPushButton("Refresh All")
        self.param_load_btn.clicked.connect(self.load_all_params)
        card.head.addWidget(self.param_load_label)
        card.head.addSpacing(10)
        card.head.addWidget(self.param_load_btn)

        bar = QHBoxLayout()
        self.param_search = QLineEdit()
        self.param_search.setPlaceholderText("Search parameters, e.g. SERIAL1 or COMPASS")
        self.param_search.setClearButtonEnabled(True)
        self.param_search.textChanged.connect(self.filter_params)
        self.param_modified_only = QCheckBox("Modified only")
        self.param_modified_only.toggled.connect(self.filter_params)
        self.param_load_file_btn = QPushButton("Load File\u2026")
        self.param_load_file_btn.clicked.connect(self.load_param_file)
        self.param_save_file_btn = QPushButton("Save File\u2026")
        self.param_save_file_btn.clicked.connect(self.save_param_file)
        bar.addWidget(self.param_search, 1)
        bar.addWidget(self.param_modified_only)
        bar.addWidget(self.param_load_file_btn)
        bar.addWidget(self.param_save_file_btn)
        card.body.addLayout(bar)

        self.param_table = make_table(["Name", "Value", "Board Value", "Type"], 0)
        self.param_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
                                         | QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        self.param_table.setFocusPolicy(Qt.StrongFocus)
        self.param_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.param_table.itemChanged.connect(self.on_param_edited)
        self.param_table.setFixedHeight(PARAM_TABLE_HEIGHT)
        fixed_columns(self.param_table, {1: 180, 2: 180, 3: 90})
        card.body.addWidget(self.param_table)

        foot = QHBoxLayout()
        self.param_edit_label = muted_label(wrap=True)
        self.param_reboot = QCheckBox("Reboot && reconnect after writing")
        self.param_reboot.setChecked(True)
        self.param_revert_btn = QPushButton("Revert")
        self.param_revert_btn.clicked.connect(self.revert_params)
        self.param_write_btn = QPushButton("Write Changes")
        self.param_write_btn.setObjectName("primary")
        self.param_write_btn.clicked.connect(self.write_param_edits)
        foot.addWidget(self.param_edit_label, 1)
        foot.addWidget(self.param_reboot)
        foot.addWidget(self.param_revert_btn)
        foot.addWidget(self.param_write_btn)
        card.body.addLayout(foot)

        tools = QHBoxLayout()
        self.params_status = muted_label(wrap=True)
        self.reset_params_btn = QPushButton("Reset to Defaults")
        self.reset_params_btn.setObjectName("danger")
        self.reset_params_btn.setToolTip("Erase every parameter and reboot the board with its defaults")
        self.reset_params_btn.clicked.connect(self.reset_parameters)
        self.reboot_btn = QPushButton("Reboot && Reconnect")
        self.reboot_btn.clicked.connect(lambda: self.start_job([], title="Reboot"))
        tools.addWidget(self.params_status, 1)
        tools.addWidget(self.reset_params_btn)
        tools.addWidget(self.reboot_btn)
        card.body.addLayout(tools)
        lay.addWidget(card)

        self.param_rows = {}
        self.param_edits = {}    # name -> text typed by the user, until the board reports that value
        self.params_shown = None

    def load_all_params(self):
        if not self.target:
            return
        self.param_indices = set()
        self.full_load = {"last_rx": time.monotonic(), "tries": 0}
        self.full_loaded = False
        sysid, compid = self.target
        self.send(self.conn.mav.param_request_list_send, sysid, compid)

    def full_load_tick(self):
        load = self.full_load
        if not load:
            if (self.stack.currentIndex() == self.PAGE_SETTINGS and not self.full_loaded and self.target
                    and self.params_ready() and not self.job):
                self.load_all_params()
            return
        now = time.monotonic()
        if self.param_count and len(self.param_indices) >= self.param_count:
            self.finish_full_load()
        elif now - load["last_rx"] > PARAM_LIST_IDLE_S:
            if load["tries"] >= PARAM_LIST_TRIES:
                self.finish_full_load()
                return
            load["tries"] += 1
            load["last_rx"] = now
            sysid, compid = self.target
            if not self.param_count:
                self.send(self.conn.mav.param_request_list_send, sysid, compid)
                return
            missing = [i for i in range(self.param_count) if i not in self.param_indices][:PARAM_LIST_BATCH]
            for index in missing:
                self.send(self.conn.mav.param_request_read_send, sysid, compid, b"", index)

    def finish_full_load(self):
        self.full_load = None
        self.full_loaded = True
        missing = self.param_count - len(self.param_indices)
        if missing > 0:
            self.add_log(mavlink.MAV_SEVERITY_WARNING, f"Parameter download incomplete: {missing} missing")
        self.rebuild_param_table()

    def param_type_name(self, name):
        ptype = self.param_types.get(name)
        return enum_name("MAV_PARAM_TYPE", ptype, "MAV_PARAM_TYPE_").upper() if ptype is not None else ""

    def rebuild_param_table(self):
        table = self.param_table
        table.blockSignals(True)
        names = sorted(self.params)
        table.setRowCount(len(names))
        self.param_rows = {}
        for row, name in enumerate(names):
            self.param_rows[name] = row
            board = format_param(self.params[name])
            for col, text in ((0, name), (1, self.param_edits.get(name, board)), (2, board),
                              (3, self.param_type_name(name))):
                item = QTableWidgetItem(text)
                if col != 1:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col in (2, 3):
                    item.setForeground(QColor(C["muted"]))
                table.setItem(row, col, item)
            self.style_param_row(name)
        table.blockSignals(False)
        self.params_shown = self.params_version
        self.filter_params()
        self.update_param_edit_label()

    def style_param_row(self, name):
        item = self.param_table.item(self.param_rows[name], 1)
        edited = name in self.param_edits
        valid = True
        if edited:
            try:
                float(self.param_edits[name])
            except ValueError:
                valid = False
        color = C["err"] if not valid else C["accent"] if edited else C["text"]
        item.setForeground(QColor(color))
        font = item.font()
        font.setBold(edited)
        item.setFont(font)

    def on_param_edited(self, item):
        if item.column() != 1:
            return
        name = self.param_table.item(item.row(), 0).text()
        text = item.text().strip()
        board = self.params.get(name)
        try:
            same = board is not None and float(text) == float(board)
        except ValueError:
            same = False
        if same:
            self.param_edits.pop(name, None)
        else:
            self.param_edits[name] = text
        self.param_table.blockSignals(True)
        self.style_param_row(name)
        self.param_table.blockSignals(False)
        self.update_param_edit_label()

    def update_param_edit_label(self):
        count = len(self.param_edits)
        set_label(self.param_edit_label, f"{count} parameter(s) changed, not written yet" if count else "",
                  C["accent"] if count else None)
        self.param_write_btn.setEnabled(bool(self.conn) and count > 0)
        self.param_revert_btn.setEnabled(count > 0)

    def filter_params(self):
        text = self.param_search.text().strip().upper()
        modified = self.param_modified_only.isChecked()
        for name, row in self.param_rows.items():
            hidden = (text and text not in name) or (modified and name not in self.param_edits)
            self.param_table.setRowHidden(row, bool(hidden))

    def refresh_params(self):
        if self.full_load:
            got = len(self.param_indices)
            total = self.param_count or "?"
            text = f"Downloading parameters {got}/{total}\u2026"
        elif self.full_loaded:
            text = f"{len(self.param_rows)} parameters"
        else:
            text = "Waiting for the board" if not self.target else "Waiting for the parameter list"
        set_label(self.param_load_label, text)
        if not self.full_loaded or self.params_shown == self.params_version:
            return
        if any(name not in self.param_rows for name in self.params):
            self.rebuild_param_table()
            return
        # update board values in place; a written value that the board now reports ends the edit
        table = self.param_table
        table.blockSignals(True)
        for name, value in self.params.items():
            row = self.param_rows[name]
            board = format_param(value)
            if table.item(row, 2).text() != board:
                table.item(row, 2).setText(board)
            edit = self.param_edits.get(name)
            if edit is not None:
                try:
                    if float(edit) == float(value):
                        del self.param_edits[name]
                except ValueError:
                    pass
            if name not in self.param_edits and table.item(row, 1).text() != board:
                table.item(row, 1).setText(board)
            self.style_param_row(name)
        table.blockSignals(False)
        self.params_shown = self.params_version
        self.update_param_edit_label()

    def write_param_edits(self):
        pairs = list(self.param_edits.items())
        if self.param_reboot.isChecked():
            text, color = self.apply_and_reboot(pairs, "Parameters")
        else:
            changes = self.param_changes(pairs)
            if isinstance(changes, str):
                text, color = changes, C["err"]
            elif not changes:
                text, color = "Nothing to change", C["muted"]
            else:
                self.send_param_sets(changes)
                text, color = f"Wrote {len(changes)} parameter(s)", C["ok"]
        set_label(self.param_edit_label, text, color)

    def revert_params(self):
        self.param_edits = {}
        self.rebuild_param_table()

    def save_param_file(self):
        if not self.params:
            self.notify("No parameters to save")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "board.param",
                                              "Parameter files (*.param *.parm)")
        if not path:
            return
        with open(path, "w") as f:
            for name in sorted(self.params):
                f.write(f"{name},{format_param(self.params[name])}\n")
        self.notify(f"Saved {len(self.params)} parameters to {path}")

    def load_param_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "",
                                              "Parameter files (*.param *.parm);;All files (*)")
        if path:
            self.apply_param_file(path)

    def apply_param_file(self, path):
        """Mission Planner / MAVProxy .param files: 'NAME,VALUE' or 'NAME VALUE', '#' comments."""
        changed, unknown = 0, []
        table = self.param_table
        table.blockSignals(True)
        with open(path) as f:
            for line in f:
                parts = re.split(r"[,\s]+", line.split("#", 1)[0].strip())
                if len(parts) < 2:
                    continue
                name, text = parts[0], parts[1]
                if name not in self.param_rows:
                    unknown.append(name)
                    continue
                board = self.params.get(name)
                try:
                    if board is not None and float(text) == float(board):
                        continue
                except ValueError:
                    pass
                self.param_edits[name] = text
                table.item(self.param_rows[name], 1).setText(text)
                self.style_param_row(name)
                changed += 1
        table.blockSignals(False)
        self.update_param_edit_label()
        self.filter_params()
        note = ""
        if unknown:
            more = "\u2026" if len(unknown) > 5 else ""
            note = f", {len(unknown)} unknown ({', '.join(unknown[:5])}{more})"
        self.notify(f"{changed} parameter(s) differ from the board{note}. Review and press Write Changes.", 0)

    # ---- settings page ----------------------------------------------------------------------

    @staticmethod
    def section_title(lay, text):
        title = QLabel(text)
        title.setObjectName("sectionTitle")
        lay.addWidget(title)

    def build_settings_page(self):
        area, lay, head = self.page_layout("Settings", "Ethernet, serial ports and every board parameter")
        self.settings_param_label = muted_label()
        self.read_params_btn = QPushButton("Read Parameters")
        self.read_params_btn.clicked.connect(self.start_param_fetch)
        head.addWidget(self.settings_param_label)
        head.addSpacing(10)
        head.addWidget(self.read_params_btn)
        self.section_title(lay, "Ethernet")
        self.build_general_tab(lay)
        self.section_title(lay, "Ports")
        self.build_ports_tab(lay)
        self.section_title(lay, "Parameters")
        self.build_params_section(lay)
        lay.addStretch()
        return area

    def build_test_page(self):
        self.test_area, lay, head = self.page_layout("Test", "Port by port board test, pinouts and the test report")
        self.build_tests_tab(lay)
        self.build_report_card(lay)
        lay.addStretch()
        return self.test_area

    def build_motors_page(self):
        area, lay, head = self.page_layout("Motors", "Frame, ESC protocol, output assignment and motor test")
        self.build_motors_tab(lay)
        lay.addStretch()
        return area

    def build_tests_tab(self, lay):
        card = Card("Port Tests", BOARD_NAME)
        info = muted_label("Pick the device, plug it into the connector and press Test. The port is configured, the "
                           "board reboots and reconnects, then the device data decides PASS or FAIL. Manual rows show "
                           "what to check; Pass / Fail set the result by hand. Pins shows the connector pinout.",
                           wrap=True)
        card.body.addWidget(info)

        options = QHBoxLayout()
        options.addWidget(muted_label("Device for UART ports"))
        self.telem_device_box = QComboBox()
        for key in TELEM_TEST_DEVICES:
            self.telem_device_box.addItem(TEST_DEVICES[key]["name"], key)
        options.addWidget(self.telem_device_box)
        options.addSpacing(16)
        self.telem3_fitted = QCheckBox("TELEM3 connector fitted")
        self.telem3_fitted.setToolTip("TELEM3 (USART2) is bridged to the CM5 and has no FMU connector by default")
        self.telem3_fitted.toggled.connect(self.on_telem3_fitted)
        options.addWidget(self.telem3_fitted)
        options.addStretch()
        card.body.addLayout(options)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(4, 1)
        for col, text in enumerate(("Connector", "Function", "Device", "Result", "Details")):
            h = muted_label(text)
            h.setStyleSheet(f"color: {C['muted']}; font-weight: 700;")
            grid.addWidget(h, 0, col)
        self.test_widgets = {}
        previous = None
        for row, spec in enumerate(PORT_TESTS, 1):
            key = spec["key"]
            designator = spec["connector"]
            # the connector name only on the first row of its group
            connector = QLabel(designator if designator != previous else "")
            connector.setObjectName("bigText")
            connector.setToolTip(CONNECTORS[designator][0])
            previous = designator
            if spec["devices"] == "telem":
                device = QLabel()
                self.telem_device_box.currentIndexChanged.connect(
                    lambda _, d=device: d.setText(self.telem_device_box.currentText()))
                device.setText(self.telem_device_box.currentText())
            elif len(spec["devices"]) > 1:
                device = QComboBox()
                for dev_key in spec["devices"]:
                    device.addItem(TEST_DEVICES[dev_key]["name"], dev_key)
            else:
                device = QLabel(TEST_DEVICES[spec["devices"][0]]["name"])
            result = QLabel("Not tested")
            result.setMinimumWidth(110)
            detail = muted_label(wrap=True)
            test_btn = QPushButton("Test")
            test_btn.setObjectName("primary")
            pass_btn = QPushButton("Pass")
            pass_btn.setObjectName("pass")
            fail_btn = QPushButton("Fail")
            fail_btn.setObjectName("fail")
            pins_btn = QPushButton("Pins")
            test_btn.clicked.connect(lambda _, k=key: self.run_test(k))
            pass_btn.clicked.connect(lambda _, k=key: self.set_test_result(k, "pass", "Marked PASS by operator"))
            fail_btn.clicked.connect(lambda _, k=key: self.set_test_result(k, "fail", "Marked FAIL by operator"))
            pins_btn.clicked.connect(lambda _, d=designator: self.show_pinout(d, scroll=True))
            for col, w in enumerate((connector, muted_label(spec["function"]), device, result, detail, test_btn,
                                     pass_btn, fail_btn, pins_btn)):
                grid.addWidget(w, row, col)
            self.test_widgets[key] = {"device": device, "result": result, "detail": detail, "test": test_btn}
        card.body.addLayout(grid)

        foot = QHBoxLayout()
        self.test_summary = QLabel()
        self.test_summary.setObjectName("bigText")
        reset_btn = QPushButton("Reset Results")
        reset_btn.clicked.connect(self.reset_tests)
        self.restore_ports_btn = QPushButton("Restore Ports")
        self.restore_ports_btn.setToolTip("Put back the port parameters changed by the tests (reboots the board)")
        self.restore_ports_btn.clicked.connect(self.restore_ports)
        foot.addWidget(self.test_summary, 1)
        foot.addWidget(reset_btn)
        foot.addWidget(self.restore_ports_btn)
        card.body.addLayout(foot)
        lay.addWidget(card)

        self.pinout_card = Card("Pinout")
        self.pinout_title = QLabel()
        self.pinout_title.setObjectName("bigText")
        self.pinout_card.body.addWidget(self.pinout_title)
        self.pinout_table = make_table(["Pin", "Signal", "Voltage"], 1, fit=True)
        self.pinout_card.body.addWidget(self.pinout_table)
        lay.addWidget(self.pinout_card)
        self.show_pinout(PORT_TESTS[0]["connector"])
        self.on_telem3_fitted(False)

    def show_pinout(self, designator, scroll=False):
        tag, connector, pins = CONNECTORS[designator]
        self.pinout_title.setText(f"{designator}   \u00b7   {tag}   \u00b7   {connector}")
        table = self.pinout_table
        table.setRowCount(len(pins))
        colors = {"+5V": C["err"], "+3.3V": C["warn"], "GND": C["muted"], "---": C["off"]}
        for i, (pin, signal, voltage) in enumerate(pins):
            set_cell(table, i, 0, str(pin), bold=True)
            set_cell(table, i, 1, signal, C["off"] if voltage == "---" else None)
            set_cell(table, i, 2, voltage, colors.get(voltage, C["accent"]))
        fit_table_height(table)
        if scroll:
            self.test_area.ensureWidgetVisible(self.pinout_card)

    def build_general_tab(self, lay):
        row = QHBoxLayout()
        row.setSpacing(16)

        net = Card("Network", "Ethernet")
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
        form = self.form_grid([(None, self.net_enable), (None, self.net_dhcp), ("IP address", self.net_ip),
                               ("Netmask", self.net_mask), ("Gateway", self.net_gw), ("MAC address", self.net_mac)])
        net.body.addLayout(form)
        net.body.addStretch()

        port = Card("Network Port 1", "MAVLink")
        self.net_p1_type = make_combo(NET_PORT_TYPES)
        self.net_p1_protocol = make_combo(SERIAL_PROTOCOLS)
        self.net_p1_port = make_spin(1, 65535, 14550)
        self.net_p1_ip = QLineEdit()
        self.net_p1_ip.setPlaceholderText("Remote IP (clients) or 0.0.0.0")
        port.body.addLayout(self.form_grid([("Type", self.net_p1_type), ("Protocol", self.net_p1_protocol),
                                            ("Port", self.net_p1_port), ("IP address", self.net_p1_ip)]))
        port.body.addStretch()
        row.addWidget(net, 1)
        row.addWidget(port, 1)
        lay.addLayout(row)

        foot = QHBoxLayout()
        self.net_status = muted_label(wrap=True)
        self.net_reload_btn = QPushButton("Reload")
        self.net_reload_btn.clicked.connect(lambda: self.reload_params(NET_PARAMS, "net_loaded"))
        self.net_apply_btn = QPushButton("Apply && Reboot")
        self.net_apply_btn.setObjectName("primary")
        self.net_apply_btn.clicked.connect(self.apply_network)
        foot.addWidget(self.net_status, 1)
        foot.addWidget(self.net_reload_btn)
        foot.addWidget(self.net_apply_btn)
        lay.addLayout(foot)

    @staticmethod
    def form_grid(rows):
        form = QGridLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        form.setColumnStretch(1, 1)
        for i, (label, widget) in enumerate(rows):
            if label is None:
                form.addWidget(widget, i, 0, 1, 2)
            else:
                form.addWidget(muted_label(label), i, 0)
                form.addWidget(widget, i, 1)
        return form

    def build_motors_tab(self, lay):
        top = QHBoxLayout()
        top.setSpacing(16)

        frame = Card("Frame & ESC")
        self.frame_class = make_combo(FRAME_CLASSES)
        self.frame_type = make_combo(FRAME_TYPES)
        self.mot_pwm_type = make_combo(MOT_PWM_TYPES)
        frame.body.addLayout(self.form_grid([("Frame class", self.frame_class), ("Frame type", self.frame_type),
                                             ("ESC protocol", self.mot_pwm_type)]))
        frame.body.addWidget(muted_label("DShot works on AUX 1-6 (FMU) only. MAIN 1-8 are driven by the IO "
                                         "co-processor and support PWM only.", wrap=True))
        frame.body.addStretch()

        test = Card("Motor Test")
        warning = QLabel("Remove the propellers. The board must be disarmed and the safety switch pressed "
                         "(or BRD_SAFETY_DEFLT = 0).")
        warning.setObjectName("warning")
        warning.setWordWrap(True)
        test.body.addWidget(warning)
        self.props_off = QCheckBox("Propellers are removed")
        self.props_off.toggled.connect(self.update_motor_test_buttons)
        test.body.addWidget(self.props_off)
        throttle_row = QHBoxLayout()
        self.motor_throttle = QSlider(Qt.Horizontal)
        self.motor_throttle.setRange(1, MOTOR_TEST_MAX_THROTTLE)
        self.motor_throttle.setValue(7)
        self.motor_throttle.setPageStep(5)
        self.throttle_label = QLabel()
        self.throttle_label.setObjectName("sliderValue")
        self.throttle_label.setMinimumWidth(64)
        self.throttle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.motor_throttle.valueChanged.connect(lambda v: self.throttle_label.setText(f"{v} %"))
        self.throttle_label.setText(f"{self.motor_throttle.value()} %")
        throttle_row.addWidget(self.motor_throttle, 1)
        throttle_row.addWidget(self.throttle_label)
        self.motor_duration = make_spin(1, 10, 2, " s")
        test.body.addWidget(muted_label("Throttle"))
        test.body.addLayout(throttle_row)
        test.body.addLayout(self.form_grid([("Duration", self.motor_duration)]))
        buttons = QGridLayout()
        buttons.setSpacing(8)
        self.motor_buttons = []
        for i in range(8):
            btn = QPushButton(f"Motor {chr(65 + i)}")
            btn.clicked.connect(lambda _, n=i + 1: self.motor_test(n, 1))
            buttons.addWidget(btn, i // 4, i % 4)
            self.motor_buttons.append(btn)
        test.body.addLayout(buttons)
        actions = QHBoxLayout()
        self.motor_all_btn = QPushButton("Test All in Sequence")
        self.motor_all_btn.clicked.connect(lambda: self.motor_test(1, 8))
        self.motor_stop_btn = QPushButton("Stop All")
        self.motor_stop_btn.setObjectName("danger")
        self.motor_stop_btn.clicked.connect(self.motor_stop)
        actions.addWidget(self.motor_all_btn)
        actions.addWidget(self.motor_stop_btn)
        test.body.addLayout(actions)
        self.motor_status = muted_label(wrap=True)
        test.body.addWidget(self.motor_status)
        test.body.addStretch()
        top.addWidget(frame, 1)
        top.addWidget(test, 1)
        lay.addLayout(top)

        outputs = Card("Outputs", "SERVO1-16")
        self.output_table = make_table(["Output", "Connector", "Function", "Min", "Max", "Live PWM"], 5, fit=True)
        fixed_columns(self.output_table, {2: 240, 3: 100, 4: 100})
        self.output_table.setRowCount(16)
        self.output_widgets = []
        for i in range(16):
            set_cell(self.output_table, i, 0, f"SERVO{i + 1}", bold=True)
            set_cell(self.output_table, i, 1, OUTPUT_NAMES[i + 1], C["muted"])
            function = make_combo(SERVO_FUNCTIONS)
            spin_min, spin_max = make_spin(500, 2500, 1000), make_spin(500, 2500, 2000)
            self.output_table.setCellWidget(i, 2, function)
            self.output_table.setCellWidget(i, 3, spin_min)
            self.output_table.setCellWidget(i, 4, spin_max)
            self.output_widgets.append((function, spin_min, spin_max))
        self.output_table.verticalHeader().setDefaultSectionSize(40)
        fit_table_height(self.output_table)
        outputs.body.addWidget(self.output_table)
        lay.addWidget(outputs)

        foot = QHBoxLayout()
        self.motors_status = muted_label(wrap=True)
        self.motors_reload_btn = QPushButton("Reload")
        self.motors_reload_btn.clicked.connect(lambda: self.reload_params(SERVO_PARAMS + MOTOR_PARAMS, "motors_loaded"))
        self.motors_apply_btn = QPushButton("Apply && Reboot")
        self.motors_apply_btn.setObjectName("primary")
        self.motors_apply_btn.clicked.connect(self.apply_motors)
        foot.addWidget(self.motors_status, 1)
        foot.addWidget(self.motors_reload_btn)
        foot.addWidget(self.motors_apply_btn)
        lay.addLayout(foot)

    def build_ports_tab(self, lay):
        card = Card("Serial Ports", "SERIAL ↔ TELEM")
        self.ports_table = make_table(["Serial", "Connector", "UART", "Protocol", "Baud", ""], 5, fit=True)
        fixed_columns(self.ports_table, {3: 260, 4: 140})
        self.ports_table.setRowCount(len(SERIAL_PORTS))
        self.port_widgets = {}
        baud_options = {code: str(baud) for code, baud in SERIAL_BAUDS.items()}
        for row, (n, (connector, uart)) in enumerate(SERIAL_PORTS.items()):
            set_cell(self.ports_table, row, 0, f"SERIAL{n}", bold=True)
            set_cell(self.ports_table, row, 1, connector + ("  (CM5)" if n == CM5_SERIAL_PORT else ""))
            set_cell(self.ports_table, row, 2, uart, C["muted"])
            protocol = make_combo(SERIAL_PROTOCOLS)
            baud = make_combo(baud_options)
            self.ports_table.setCellWidget(row, 3, protocol)
            self.ports_table.setCellWidget(row, 4, baud)
            self.port_widgets[n] = (protocol, baud)
        self.ports_table.verticalHeader().setDefaultSectionSize(40)
        fit_table_height(self.ports_table)
        card.body.addWidget(self.ports_table)
        card.body.addWidget(muted_label("USART6 is the internal FMU ↔ IO co-processor link and is not a SERIAL "
                                        "port.", wrap=True))
        lay.addWidget(card)

        foot = QHBoxLayout()
        self.ports_status = muted_label(wrap=True)
        self.ports_reload_btn = QPushButton("Reload")
        self.ports_reload_btn.clicked.connect(lambda: self.reload_params(SERIAL_PARAMS, "ports_loaded"))
        self.ports_apply_btn = QPushButton("Apply && Reboot")
        self.ports_apply_btn.setObjectName("primary")
        self.ports_apply_btn.clicked.connect(self.apply_ports)
        foot.addWidget(self.ports_status, 1)
        foot.addWidget(self.ports_reload_btn)
        foot.addWidget(self.ports_apply_btn)
        lay.addLayout(foot)

    # ---- state ------------------------------------------------------------------------------

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
        self.mag_field = {}
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
        self.servo_all = []
        self.status_shown = None
        self.net_loaded = False
        self.ports_loaded = False
        self.motors_loaded = False
        self.reset_pending = False
        self.param_count = 0
        self.param_indices = set()
        self.params_version = 0
        self.full_load = None
        self.full_loaded = False

    def notify(self, text, ms=8000):
        self.statusBar().showMessage(text, ms)

    def add_log(self, severity, text, persist=False):
        name = enum_name("MAV_SEVERITY", severity, "MAV_SEVERITY_")
        self.status_log.append((time.strftime("%H:%M:%S"), severity, name, text))
        del self.status_log[:-STATUS_LOG_MAX]
        self.status_seq += 1
        if persist or severity <= mavlink.MAV_SEVERITY_WARNING:
            level = logging.ERROR if severity <= mavlink.MAV_SEVERITY_ERROR else (
                logging.WARNING if severity == mavlink.MAV_SEVERITY_WARNING else logging.INFO)
            file_log.log(level, "[%s] %s", name, text)

    def log_exception(self, exc_type, exc, tb):
        """Log an internal error once; repeats (e.g. from the refresh timer) are only counted. True if new."""
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        count = self.exception_counts.get(text, 0) + 1
        self.exception_counts[text] = count
        if count == 1:
            context = f"port={self.port_box.currentText() or '-'} connected={bool(self.target)} " \
                      f"test={self.active_test or '-'}"
            file_log.error("Internal error (%s)\n%s", context, text.rstrip())
            self.add_log(mavlink.MAV_SEVERITY_CRITICAL, f"Internal error: {exc_type.__name__}: {exc}")
        elif count in (10, 100, 1000):
            file_log.error("Internal error repeated %d times: %s: %s", count, exc_type.__name__, exc)
        return count == 1

    def clear_status_log(self):
        self.status_log = []
        self.status_seq += 1
        self.refresh_status_log()

    def link_buttons(self):
        return [self.reload_btn, self.read_params_btn, self.param_load_btn, self.net_reload_btn, self.net_apply_btn,
                self.motors_reload_btn, self.motors_apply_btn, self.ports_reload_btn, self.ports_apply_btn,
                self.reset_params_btn, self.reboot_btn, self.restore_ports_btn]

    def set_connected_ui(self, connected):
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.connect_btn.setProperty("connected", connected)
        self.connect_btn.style().unpolish(self.connect_btn)
        self.connect_btn.style().polish(self.connect_btn)
        self.port_box.setEnabled(not connected)
        self.baud_box.setEnabled(not connected)
        for btn in self.link_buttons():
            btn.setEnabled(connected)
        self.update_motor_test_buttons()
        if connected:
            self.refresh_timer.start()
            self.heartbeat_timer.start()
            self.port_timer.stop()
        else:
            self.refresh_timer.stop()
            self.heartbeat_timer.stop()
            self.param_timer.stop()
            self.port_timer.start()
        self.update_connect_enabled()
        self.update_status_pill()

    def update_connect_enabled(self):
        self.connect_btn.setEnabled(bool(self.conn) or self.port_box.currentData() is not None)

    def update_status_pill(self):
        stage = self.job["stage"] if self.job else None
        if stage == "writing":
            text, color = "Writing params", C["warn"]
        elif stage in ("rebooting", "reconnecting", "heartbeat"):
            text, color = "Rebooting…", C["warn"]
        elif not self.conn:
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
            f"border-radius: 15px; padding: 0 12px; font-weight: 700;")

    # ---- connection -------------------------------------------------------------------------

    def refresh_ports(self):
        if self.conn:
            return
        ports = available_ports()
        devices = [p.device for p in ports]
        if devices == self.port_list:
            return
        self.port_list = devices
        current = self.port_box.currentData() or self.port_device
        self.port_box.blockSignals(True)
        self.port_box.clear()
        for p in ports:
            self.port_box.addItem(f"{p.device}  —  {p.description}", p.device)
        if not ports:
            self.port_box.addItem("No serial ports found", None)
        index = self.port_box.findData(current) if current else -1
        self.port_box.setCurrentIndex(max(index, 0))
        self.port_box.blockSignals(False)
        self.update_connect_enabled()

    def toggle_connection(self):
        if self.conn:
            self.cancel_job("Cancelled by disconnect")
            self.disconnect_board()
            return
        port = self.port_box.currentData()
        if not port:
            self.notify("No port selected")
            return
        self.status_log = []
        self.status_seq += 1
        self.open_connection(port, int(self.baud_box.currentText()))

    def open_connection(self, port, baud, quiet=False):
        try:
            conn = mavutil.mavlink_connection(port, baud=baud, source_system=255,
                                              source_component=mavlink.MAV_COMP_ID_MISSIONPLANNER,
                                              autoreconnect=False)
        except Exception as e:
            if not quiet:
                self.notify(f"Could not connect: {e}")
            return False
        self.conn = conn
        self.port_device, self.port_baud = port, baud
        self.reset_state()
        self.clear_views()
        self.reader = MavlinkReader(self.conn)
        self.reader.message.connect(self.handle_message)
        self.reader.error.connect(self.connection_error)
        self.reader.start()
        self.set_connected_ui(True)
        if not quiet:
            self.notify(f"{port} opened, waiting for heartbeat...")
        return True

    def disconnect_board(self):
        for timer in (self.refresh_timer, self.heartbeat_timer, self.param_timer):
            timer.stop()
        reader, self.reader = self.reader, None
        if reader:
            reader.blockSignals(True)  # no more message or error callbacks into a half closed window
            reader.stop()
            if not reader.wait(READER_STOP_TIMEOUT_MS):
                reader.terminate()
                reader.wait()
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            if not self.job:
                self.notify("Disconnected")
        self.port_list = None
        self.set_connected_ui(False)
        self.refresh_ports()

    def connection_error(self, text):
        self.disconnect_board()
        if self.job and self.job["stage"] in ("reconnecting", "heartbeat"):
            self.job.update(stage="reconnecting", next_try=time.monotonic() + 1.0)
        else:
            self.notify(f"Connection error: {text}", 0)

    def clear_views(self):
        for table in (self.live_table, self.onboard_table):
            table.setRowCount(0)
            fit_table_height(table)
        self.status_table.setRowCount(0)
        for card in (self.board_card, self.rc_card, self.servo_card):
            card.clear()
        for w in (self.net_ip, self.net_mask, self.net_gw, self.net_p1_ip):
            w.clear()
        self.net_mac.setText("-")
        for label in (self.param_label, self.settings_param_label):
            label.setText("")
        self.param_table.setRowCount(0)
        self.param_rows = {}
        self.param_edits = {}
        self.params_shown = None
        self.update_param_edit_label()

    def send(self, func, *args):
        if not self.conn:
            return
        try:
            func(*args)
        except Exception as e:
            self.connection_error(str(e))

    def send_heartbeat(self):
        self.send(self.conn.mav.heartbeat_send, mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def send_command(self, command, *params, label=None):
        if not self.target:
            return
        sysid, compid = self.target
        values = list(params) + [0] * (7 - len(params))
        if label is not None:
            self.cmd_labels[command] = label
        self.send(self.conn.mav.command_long_send, sysid, compid, command, 0, *values)

    def on_board_found(self):
        sysid, compid = self.target
        self.send(self.conn.mav.request_data_stream_send, sysid, compid, mavlink.MAV_DATA_STREAM_ALL, 4, 1)
        self.send_command(mavlink.MAV_CMD_REQUEST_MESSAGE, mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION)
        if self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            self.send_command(mavlink.MAV_CMD_DO_SEND_BANNER)
        self.start_param_fetch()
        if self.job and self.job["stage"] == "heartbeat":
            self.finish_job()

    # ---- auto reboot / reconnect --------------------------------------------------------------

    def start_job(self, names, on_ready=None, title="Settings"):
        """Wait until the written parameters are confirmed, reboot, reconnect, then call on_ready."""
        self.job = {"stage": "writing", "names": set(names), "deadline": time.monotonic() + PARAM_CONFIRM_S,
                    "on_ready": on_ready, "title": title}
        self.notify(f"{title}: saving parameters…", 0)
        self.update_status_pill()

    def job_tick(self):
        self.check_active_test()
        if self.stack.currentIndex() == self.PAGE_TEST:
            self.refresh_tests()
        job = self.job
        if not job:
            return
        now = time.monotonic()
        stage = job["stage"]
        if stage == "writing":
            missing = sorted(n for n in job["names"] if n not in self.params)
            if not missing or now > job["deadline"]:
                if missing:
                    self.add_log(mavlink.MAV_SEVERITY_WARNING, f"Not confirmed before reboot: {', '.join(missing)}")
                self.reboot_now()
        elif stage == "reconnecting":
            if now > job["deadline"]:
                self.fail_job(f"{job['title']}: board did not come back after reboot")
            elif not self.conn and now >= job["next_try"]:
                job["next_try"] = now + 1.0
                if self.port_device in [p.device for p in available_ports()]:
                    if self.open_connection(self.port_device, self.port_baud, quiet=True):
                        job.update(stage="heartbeat", hb_deadline=now + HEARTBEAT_WAIT_S)
        elif stage == "heartbeat":
            if now > job["deadline"]:
                self.fail_job(f"{job['title']}: no heartbeat after reboot")
            elif not self.conn or now > job["hb_deadline"]:
                if self.conn:
                    self.disconnect_board()
                job.update(stage="reconnecting", next_try=now + 1.0)
        self.update_status_pill()

    def reboot_now(self):
        self.job["stage"] = "rebooting"
        self.send_command(mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN, 1)
        self.notify(f"{self.job['title']}: rebooting board…", 0)
        QTimer.singleShot(REBOOT_GRACE_MS, self.after_reboot_sent)

    def after_reboot_sent(self):
        if not self.job or self.job["stage"] != "rebooting":
            return
        self.disconnect_board()
        now = time.monotonic()
        self.job.update(stage="reconnecting", next_try=now + RECONNECT_DELAY_S, deadline=now + RECONNECT_TIMEOUT_S)
        self.notify(f"{self.job['title']}: waiting for the board to come back…", 0)

    def finish_job(self):
        job, self.job = self.job, None
        self.notify(f"{job['title']}: done, board reconnected")
        self.add_log(mavlink.MAV_SEVERITY_INFO, f"{job['title']}: rebooted and reconnected")
        if job["on_ready"]:
            job["on_ready"]()

    def fail_job(self, text):
        self.job = None
        self.notify(text, 0)
        self.add_log(mavlink.MAV_SEVERITY_ERROR, text)
        if self.active_test:
            self.set_test_result(self.active_test, "fail", text)

    def cancel_job(self, reason):
        if self.job:
            self.job = None
            if self.active_test:
                name = self.test_name(TESTS_BY_KEY[self.active_test])
                self.add_log(mavlink.MAV_SEVERITY_WARNING, f"Port test {name}: cancelled ({reason})")
                self.set_test_result(self.active_test, "idle", reason)

    # ---- parameters -------------------------------------------------------------------------

    def start_param_fetch(self):
        if not self.conn or not self.target:
            return
        if self.autopilot == mavlink.MAV_AUTOPILOT_PX4:
            names = [n for group in PX4_DEVICE_PARAMS.values() for n in group]
        elif self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            names = [n for group in AP_DEVICE_PARAMS.values() for n in group] + AP_PARAMS
        else:
            names = []
        self.param_names = list(dict.fromkeys(names))
        for name in self.param_names:
            self.params.pop(name, None)
        self.net_loaded = self.ports_loaded = self.motors_loaded = False
        self.fetch_params()

    def fetch_params(self):
        self.param_tries = 0
        self.request_missing_params()
        if self.param_names:
            self.param_timer.start()

    def request_missing_params(self):
        missing = [n for n in self.param_names if n not in self.params]
        if not missing or self.param_tries >= PARAM_MAX_TRIES or not self.target:
            self.param_timer.stop()
            return
        self.param_tries += 1
        sysid, compid = self.target
        for name in missing:
            self.send(self.conn.mav.param_request_read_send, sysid, compid, name.encode(), -1)

    def reload_params(self, names, loaded_flag):
        for name in names:
            self.params.pop(name, None)
        setattr(self, loaded_flag, False)
        self.fetch_params()

    def params_ready(self):
        return bool(self.param_names) and not self.param_timer.isActive()

    def param_int(self, name):
        value = self.params.get(name)
        return int(value) if value is not None else 0

    def param_missing(self, name):
        return name in self.param_names and name not in self.params and not self.param_timer.isActive()

    def param_value(self, name):
        return self.params.get(name)

    def param_changes(self, pairs):
        """(name, value) pairs that differ from the board; a str on invalid input."""
        changes = []
        for name, value in pairs:
            if value is None or value == "" or self.param_missing(name):
                continue
            try:
                value = float(value)
            except ValueError:
                return f"Invalid value for {name}: {value}"
            current = self.params.get(name)
            if current is not None and float(current) == value:
                continue
            changes.append((name, value))
        return changes

    def send_param_sets(self, changes):
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

    def apply_and_reboot(self, pairs, title, on_ready=None):
        """Write the changed parameters, then reboot and reconnect. Returns (status text, color)."""
        if not self.target:
            return "Not connected", C["err"]
        if self.job:
            return "Busy, wait for the running reboot", C["warn"]
        changes = self.param_changes(pairs)
        if isinstance(changes, str):
            return changes, C["err"]
        if not changes:
            if on_ready:
                on_ready()
            return "Nothing to change", C["muted"]
        self.send_param_sets(changes)
        self.start_job([n for n, _ in changes], on_ready, title)
        return f"Writing {len(changes)} parameter(s), the board will reboot and reconnect", C["ok"]

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

    def sensor_healthy(self, bit):
        state = self.sys_status_state(bit)
        return bool(state and state[0] and state[2] and self.power
                    and time.monotonic() - self.power["t"] <= STALE_S)

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
            if not self.job:
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
        self.add_log(m.severity, text)
        self.notify(f"[{enum_name('MAV_SEVERITY', m.severity, 'MAV_SEVERITY_')}] {text}")

    def on_COMMAND_ACK(self, m):
        name = enum_name("MAV_CMD", m.command, "MAV_CMD_")
        result = enum_name("MAV_RESULT", m.result, "MAV_RESULT_")
        good = m.result in (mavlink.MAV_RESULT_ACCEPTED, mavlink.MAV_RESULT_IN_PROGRESS)
        if m.command not in (mavlink.MAV_CMD_REQUEST_MESSAGE, mavlink.MAV_CMD_DO_SEND_BANNER):
            self.add_log(mavlink.MAV_SEVERITY_INFO if good else mavlink.MAV_SEVERITY_WARNING, f"{name}: {result}")
        label = self.cmd_labels.pop(m.command, None)
        if label is not None:
            set_label(label, f"{name}: {result}", C["ok"] if good else C["err"])
        if m.command == mavlink.MAV_CMD_PREFLIGHT_STORAGE and self.reset_pending:
            self.reset_pending = False
            if good:
                self.start_job([], title="Parameter reset")

    def on_PARAM_VALUE(self, m):
        name = m.param_id.rstrip("\x00") if isinstance(m.param_id, str) else m.param_id.decode().rstrip("\x00")
        value = m.param_value
        if self.autopilot == mavlink.MAV_AUTOPILOT_PX4 and m.param_type in INT_TYPES:
            value = struct.unpack("<i", struct.pack("<f", value))[0]
        elif m.param_type in INT_TYPES:
            value = int(round(value))
        self.params[name] = value
        self.param_types[name] = m.param_type
        self.params_version += 1
        if m.param_index < 0xFFFF and m.param_count:
            self.param_count = m.param_count
            self.param_indices.add(m.param_index)
        if self.full_load:
            self.full_load["last_rx"] = time.monotonic()

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

    def set_mag(self, inst, msg, x, y, z, unit, to_mgauss=None, fmt="{:.0f}"):
        self.set_live("mag", inst, msg, xyz(x, y, z, unit, fmt))
        if to_mgauss is not None:
            self.mag_field[inst] = math.sqrt(x * x + y * y + z * z) * to_mgauss

    def imu(self, inst, msg, m, unit_acc, unit_gyro, unit_mag):
        self.set_live("accel", inst, msg, xyz(m.xacc, m.yacc, m.zacc, unit_acc))
        self.set_live("gyro", inst, msg, xyz(m.xgyro, m.ygyro, m.zgyro, unit_gyro))
        if (m.xmag, m.ymag, m.zmag) != (0, 0, 0):
            self.set_mag(inst, msg, m.xmag, m.ymag, m.zmag, unit_mag, 1.0 if unit_mag == "mGauss" else None)
        temp = getattr(m, "temperature", 0)
        if temp:
            self.set_live("temp", inst, msg, f"{temp / 100:.1f} C")

    def on_RAW_IMU(self, m):
        # ArduPilot fills RAW_IMU with scaled, calibrated values (same units as SCALED_IMU)
        if self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            self.imu(getattr(m, "id", 0), "RAW_IMU", m, "mG", "mrad/s", "mGauss")
        else:
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
        self.set_mag(inst, "HIGHRES_IMU", m.xmag, m.ymag, m.zmag, "Gauss", 1000.0, "{:.3f}")
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
        self.servo_all = list(outs)
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
        self.full_load_tick()
        page = self.stack.currentIndex()
        if page == self.PAGE_BOARD:
            self.refresh_board()
            self.refresh_onboard()
        elif page == self.PAGE_TEST:
            self.refresh_tests()
        elif page == self.PAGE_MOTORS:
            self.refresh_motors()
        elif page == self.PAGE_SETTINGS:
            self.refresh_settings()
        elif page == self.PAGE_LIVE:
            self.refresh_live()
        else:
            self.refresh_status_log()

    def update_param_labels(self):
        if self.param_names:
            got = sum(1 for n in self.param_names if n in self.params)
            text = (f"Reading parameters {got}/{len(self.param_names)}..." if self.param_timer.isActive()
                    else f"{got} of {len(self.param_names)} parameters found")
        elif self.target:
            text = "Parameters not available for this autopilot"
        else:
            text = ""
        for label in (self.param_label, self.settings_param_label):
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

    def onboard_rows(self):
        """Rows of (item, interface, status, status color, details) for the expected board hardware."""
        now = time.monotonic()
        ok, warn, err, muted = C["ok"], C["warn"], C["err"], C["muted"]
        ready = self.params_ready()
        rows = []

        def recent(data):
            return data is not None and now - data["t"] <= STALE_S

        def voltage_state(v, low, high):
            return ("OK", ok) if low <= v <= high else ("Out of range", err)

        def expect(found, label, iface, details):
            if found:
                rows.append((label, iface, "Detected", ok, details))
            else:
                rows.append((label, iface, "Missing" if ready else "Unknown", err if ready else muted, details))

        imus = self.devices("accel")
        for label, iface, chip in ONBOARD_IMUS:
            dev = next((d for d in imus if d["name"] == chip), None)
            if dev:
                imus.remove(dev)
            expect(dev, label, iface, f"{dev['chip']}   {dev['bus']}" if dev else f"No {chip} in INS_ACC*_ID")

        baros = [(i, d) for i, d in enumerate(self.devices("baro")) if d["name"] in BARO_390_NAMES]
        for label, iface in ONBOARD_BAROS:
            found = baros.pop(0) if baros else None
            details = "No BMP390 in BARO*_DEVID"
            if found:
                inst, dev = found
                value = self.live.get(("baro", inst))
                details = f"{dev['bus']}   {dev['address']}" + (f"   {value[1]}" if value else "")
            expect(found, label, iface, details)

        mag = next((d for d in self.devices("mag") if d["name"] == "BMM350"), None)
        expect(mag, "Compass BMM350 (IMU-01)", "I2C4", f"{mag['bus']}   {mag['address']}" if mag else
               "No BMM350 in COMPASS_DEV_ID*")

        batt = self.param_value("BATT_MONITOR")
        pw = self.power
        reading = f"{pw['voltage']:.2f} V" if pw and pw["voltage"] is not None else "No reading"
        if batt is None:
            rows.append(("Power monitor INA238", "I2C1", "Unknown", muted, "BATT_MONITOR not read"))
        elif int(batt) == 21:
            rows.append(("Power monitor INA238", "I2C1", "Configured", ok, f"{reading}   (voltage only, no shunt)"))
        else:
            rows.append(("Power monitor INA238", "I2C1", "Not selected", warn, f"BATT_MONITOR = {int(batt)} (INA2xx is 21)"))

        ps = self.power_status if recent(self.power_status) else None
        vcc = ps["vcc"] if ps else (self.hwstatus["vcc"] if recent(self.hwstatus) else None)
        if vcc and vcc > 1:
            rows.append(("Scaled 5V", "ADC", *voltage_state(vcc, 4.5, 5.5), f"{vcc:.2f} V"))
        else:
            rows.append(("Scaled 5V", "ADC", "Not fitted", muted, "No VDD_5V_SENS circuit on this board (PB1 unused)"))
        if ps and ps["flags"] & mavlink.MAV_POWER_STATUS_SERVO_VALID:
            rows.append(("Servo rail", "ADC_6V6", "OK", ok, f"{ps['vservo']:.2f} V"))
        else:
            rows.append(("Servo rail", "ADC_6V6", "Not powered" if ps else "No data", muted,
                         "Servo rail voltage from POWER_STATUS"))

        mcu = self.mcu if recent(self.mcu) else None
        if mcu:
            rows.append(("MCU 3V3", "ADC1", *voltage_state(mcu["v"], 3.1, 3.5),
                         f"{mcu['v']:.2f} V   (min {mcu['vmin']:.2f} V, max {mcu['vmax']:.2f} V)   "
                         f"MCU {mcu['temp']:.1f} C"))
        else:
            rows.append(("MCU 3V3", "ADC1", "No data", muted, "Not reported (MCU_STATUS)"))
        rows.append(("Sensor 3V3 rails S1-S4", "ADC1", "Not reported", muted,
                     "SCALED_VDD_3V3_S1..S4 are not sent over MAVLink"))

        if ps:
            flags = ps["flags"]
            oc = flags & mavlink.MAV_POWER_STATUS_PERIPH_OVERCURRENT
            hp_oc = flags & mavlink.MAV_POWER_STATUS_PERIPH_HIPOWER_OVERCURRENT
            details = [f"PERIPH nOC {'active' if oc else 'clear'}", f"HIPOWER nOC {'active' if hp_oc else 'clear'}",
                       f"Brick {'OK' if flags & mavlink.MAV_POWER_STATUS_BRICK_VALID else 'invalid'}",
                       "USB connected" if flags & mavlink.MAV_POWER_STATUS_USB_CONNECTED else "USB not connected"]
            rows.append(("Peripheral power nEN / nOC", "GPIO", *(("Overcurrent", err) if oc or hp_oc else ("OK", ok)),
                         "   ".join(details)))
        else:
            rows.append(("Peripheral power nEN / nOC", "GPIO", "No data", muted, "Not reported (POWER_STATUS)"))

        io_enable = self.param_value("BRD_IO_ENABLE")
        if io_enable is None:
            io = ("Unknown", muted, "BRD_IO_ENABLE not available")
        elif not int(io_enable):
            io = ("Disabled", muted, "BRD_IO_ENABLE = 0")
        elif self.iomcu_msg and self.iomcu_msg[0] <= mavlink.MAV_SEVERITY_WARNING:
            io = ("Error", err, self.iomcu_msg[1])
        else:
            io = ("Enabled", ok, self.iomcu_msg[1] if self.iomcu_msg else "BRD_IO_ENABLE = 1, no IOMCU errors reported")
        rows.append(("IO co-processor (STM32F103)", "USART6", *io))

        rc_state = self.sys_status_state(mavlink.MAV_SYS_STATUS_SENSOR_RC_RECEIVER)
        rc = self.rc if recent(self.rc) else None
        if rc and rc["chans"] and (not rc_state or rc_state[2]):
            rssi = f"   RSSI {rc['rssi']}" if rc["rssi"] != 255 else ""
            rows.append(("SBUS / PPM RC", "RCIN", "Receiving", ok, f"{len(rc['chans'])} channels{rssi}"))
        elif rc_state and rc_state[0]:
            rows.append(("SBUS / PPM RC", "RCIN", "No signal", err, "RC receiver present but unhealthy"))
        else:
            rows.append(("SBUS / PPM RC", "RCIN", "No signal", muted, "No RC_CHANNELS data"))

        log_state = self.sys_status_state(mavlink.MAV_SYS_STATUS_LOGGING)
        if log_state is None:
            rows.append(("microSD card", "SDMMC2", "Unknown", muted, "Waiting for SYS_STATUS"))
        elif not log_state[0]:
            rows.append(("microSD card", "SDMMC2", "Not reported", muted, "Logging disabled"))
        elif log_state[2]:
            rows.append(("microSD card", "SDMMC2", "Present", ok, "Logging healthy"))
        else:
            rows.append(("microSD card", "SDMMC2", "Missing / error", err, "Logging unhealthy (no card or write error)"))

        proto = self.param_value(f"SERIAL{CM5_SERIAL_PORT}_PROTOCOL")
        baud = self.param_value(f"SERIAL{CM5_SERIAL_PORT}_BAUD")
        port = f"SERIAL{CM5_SERIAL_PORT}"
        if proto is None:
            rows.append(("CM5 serial bridge", "USART2 / TELEM3", "Unknown", muted, f"{port} not read"))
        else:
            name = SERIAL_PROTOCOLS.get(int(proto), f"Protocol {int(proto)}")
            rate = SERIAL_BAUDS.get(int(baud), baud) if baud is not None else "-"
            good = int(proto) in (1, 2, 45)
            rows.append(("CM5 serial bridge", "USART2 / TELEM3", "Configured" if good else name, ok if good else warn,
                         f"{port}: {name} @ {rate}"))

        params_ok = any(n in self.params for n in self.param_names)
        rows.append(("FRAM FM25V02A", "SPI5", "Params readable" if params_ok else "Unknown", ok if params_ok else muted,
                     "Parameter storage (indirect check)"))
        rows.append(("EEPROM AT24C02D", "I2C3", "Not reported", muted, "No MAVLink status for this device"))
        return rows

    def refresh_onboard(self):
        rows = self.onboard_rows()
        table = self.onboard_table
        table.setRowCount(len(rows))
        fit_table_height(table)
        for i, (item, iface, status, color, details) in enumerate(rows):
            set_cell(table, i, 0, item, bold=True)
            set_cell(table, i, 1, iface, C["muted"])
            set_cell(table, i, 2, status, color)
            set_cell(table, i, 3, details)

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

        for inst in range(3):
            dev = self.device_id("mag", inst)
            if not dev and ("mag", inst) not in self.live:
                continue
            info = decode_device_id(dev, "mag", ardupilot) if dev else empty
            rows.append((f"Compass {inst + 1}", info["chip"], info["bus"], info["address"], "", None, "group"))
            rows.append(("Field", "", "", "", *live(("mag", inst)), "child"))
            field = self.mag_field.get(inst)
            if field is not None:
                text = f"{field:.0f} mGauss   (EMI check {MAG_FIELD_MIN}-{MAG_FIELD_MAX})"
                bad = not MAG_FIELD_MIN <= field <= MAG_FIELD_MAX
                rows.append(("MagField", "", "", "", text, timed(self.live[("mag", inst)][2],
                                                               C["err"] if bad else C["ok"]), "child"))

        for inst in range(3):
            dev = self.device_id("baro", inst)
            if not dev and ("baro", inst) not in self.live:
                continue
            info = decode_device_id(dev, "baro", ardupilot) if dev else empty
            rows.append((f"Barometer {inst + 1}", info["chip"], info["bus"], info["address"],
                         *live(("baro", inst)), "single"))

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
        for i, (stamp, level, severity, text) in enumerate(entries):
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

    def refresh_settings(self):
        self.load_network()
        if self.param_missing("NET_ENABLE"):
            set_label(self.net_status, "Board has no NET_* parameters (no Ethernet support)", C["muted"])
        self.load_ports()
        self.refresh_params()

    def refresh_motors(self):
        self.load_motors()
        for i in range(16):
            pwm = self.servo_all[i] if i < len(self.servo_all) else 0
            set_cell(self.output_table, i, 5, str(pwm) if pwm else "-", None if pwm else C["muted"])

    # ---- settings: general ------------------------------------------------------------------

    def update_net_fields(self):
        static = not self.net_dhcp.isChecked()
        for w in (self.net_ip, self.net_mask, self.net_gw):
            w.setEnabled(static)

    def load_network(self):
        if self.net_loaded or any(n not in self.params for n in NET_CONFIG_PARAMS):
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
        if all(n in self.params for n in NET_PORT_PARAMS):
            set_combo_value(self.net_p1_type, self.param_int("NET_P1_TYPE"))
            set_combo_value(self.net_p1_protocol, self.param_int("NET_P1_PROTOCOL"), "Protocol {}")
            self.net_p1_port.setValue(self.param_int("NET_P1_PORT"))
            self.net_p1_ip.setText(".".join(str(self.param_int(n)) for n in NET_P1_IP_PARAMS))
        self.update_net_fields()

    def apply_network(self):
        pairs = [("NET_ENABLE", int(self.net_enable.isChecked())), ("NET_DHCP", int(self.net_dhcp.isChecked())),
                 ("NET_P1_TYPE", self.net_p1_type.currentData()), ("NET_P1_PROTOCOL", self.net_p1_protocol.currentData()),
                 ("NET_P1_PORT", self.net_p1_port.value())]
        try:
            if not self.net_dhcp.isChecked():
                ip = ipaddress.IPv4Address(self.net_ip.text().strip())
                gw = ipaddress.IPv4Address(self.net_gw.text().strip())
                mask = self.net_mask.text().strip().lstrip("/")
                bits = int(mask) if mask.isdigit() else ipaddress.IPv4Network(f"0.0.0.0/{mask}").prefixlen
                if not 0 <= bits <= 32:
                    raise ValueError(mask)
                pairs += list(zip(NET_IP_PARAMS, ip.packed)) + list(zip(NET_GW_PARAMS, gw.packed))
                pairs.append(("NET_NETMASK", bits))
            if self.net_p1_ip.text().strip():
                pairs += list(zip(NET_P1_IP_PARAMS, ipaddress.IPv4Address(self.net_p1_ip.text().strip()).packed))
        except ValueError as e:
            set_label(self.net_status, f"Invalid address: {e}", C["err"])
            return
        set_label(self.net_status, *self.apply_and_reboot(pairs, "Network"))

    # ---- settings: motors -------------------------------------------------------------------

    def load_motors(self):
        if self.motors_loaded or any(n not in self.params for n in ("SERVO1_FUNCTION", "SERVO8_FUNCTION")):
            return
        self.motors_loaded = True
        for combo, name, fmt in ((self.frame_class, "FRAME_CLASS", "Class {}"), (self.frame_type, "FRAME_TYPE", "Type {}"),
                                 (self.mot_pwm_type, "MOT_PWM_TYPE", "Type {}")):
            if name in self.params:
                set_combo_value(combo, self.param_int(name), fmt)
        for i, (function, spin_min, spin_max) in enumerate(self.output_widgets, 1):
            if f"SERVO{i}_FUNCTION" in self.params:
                set_combo_value(function, self.param_int(f"SERVO{i}_FUNCTION"), "Function {}")
            if f"SERVO{i}_MIN" in self.params:
                spin_min.setValue(self.param_int(f"SERVO{i}_MIN"))
            if f"SERVO{i}_MAX" in self.params:
                spin_max.setValue(self.param_int(f"SERVO{i}_MAX"))

    def apply_motors(self):
        if not self.motors_loaded:
            set_label(self.motors_status, "Wait until the output parameters are read", C["warn"])
            return
        pairs = [("FRAME_CLASS", self.frame_class.currentData()), ("FRAME_TYPE", self.frame_type.currentData()),
                 ("MOT_PWM_TYPE", self.mot_pwm_type.currentData())]
        for i, (function, spin_min, spin_max) in enumerate(self.output_widgets, 1):
            if spin_min.value() >= spin_max.value():
                set_label(self.motors_status, f"SERVO{i}: min must be below max", C["err"])
                return
            pairs += [(f"SERVO{i}_FUNCTION", function.currentData()), (f"SERVO{i}_MIN", spin_min.value()),
                      (f"SERVO{i}_MAX", spin_max.value())]
        set_label(self.motors_status, *self.apply_and_reboot(pairs, "Motors"))

    def update_motor_test_buttons(self):
        enabled = bool(self.conn) and self.props_off.isChecked()
        for btn in (*self.motor_buttons, self.motor_all_btn, self.motor_throttle):
            btn.setEnabled(enabled)
        self.motor_stop_btn.setEnabled(bool(self.conn))

    def motor_test(self, motor, count):
        throttle, duration = self.motor_throttle.value(), self.motor_duration.value()
        # param1 motor sequence (A = 1), param2 0 = throttle percent, param3 throttle, param4 timeout,
        # param5 number of motors to run in sequence
        self.send_command(mavlink.MAV_CMD_DO_MOTOR_TEST, motor, 0, throttle, duration, count, 0,
                          label=self.motor_status)
        what = f"motors A-{chr(64 + count)} in sequence" if count > 1 else f"motor {chr(64 + motor)}"
        set_label(self.motor_status, f"Spinning {what} at {throttle}% for {duration} s…", C["accent"])

    def motor_stop(self):
        for motor in range(1, 9):
            self.send_command(mavlink.MAV_CMD_DO_MOTOR_TEST, motor, 0, 0, 0, 1, 0)
        set_label(self.motor_status, "Stop sent to all motors", C["muted"])

    # ---- settings: ports --------------------------------------------------------------------

    def load_ports(self):
        if self.ports_loaded or any(f"SERIAL{n}_PROTOCOL" not in self.params for n in SERIAL_PORTS if n):
            return
        self.ports_loaded = True
        for n, (protocol, baud) in self.port_widgets.items():
            if f"SERIAL{n}_PROTOCOL" in self.params:
                set_combo_value(protocol, self.param_int(f"SERIAL{n}_PROTOCOL"), "Protocol {}")
            if f"SERIAL{n}_BAUD" in self.params:
                set_combo_value(baud, self.param_int(f"SERIAL{n}_BAUD"), "{}")

    def apply_ports(self):
        if not self.ports_loaded:
            set_label(self.ports_status, "Wait until the port parameters are read", C["warn"])
            return
        pairs = []
        for n, (protocol, baud) in self.port_widgets.items():
            pairs += [(f"SERIAL{n}_PROTOCOL", protocol.currentData()), (f"SERIAL{n}_BAUD", baud.currentData())]
        set_label(self.ports_status, *self.apply_and_reboot(pairs, "Ports"))

    # ---- parameters: reset ------------------------------------------------------------------

    def reset_parameters(self):
        answer = QMessageBox.question(self, "Reset Parameters",
                                      "Erase ALL parameters, including calibration, and reboot the board?")
        if answer != QMessageBox.Yes:
            return
        self.reset_pending = True
        # param1 = 2: reset all parameters to their defaults (PARAM_RESET_FACTORY_DEFAULT)
        self.send_command(mavlink.MAV_CMD_PREFLIGHT_STORAGE, 2, label=self.params_status)
        set_label(self.params_status, "Resetting parameters…", C["warn"])

    # ---- settings: port tests ---------------------------------------------------------------

    def on_telem3_fitted(self, fitted):
        state = self.tests["telem3"]
        if not fitted:
            state.update(state="na", detail="TELEM3 not fitted on this board", deadline=0.0)
            if self.active_test == "telem3":
                self.active_test = None
        elif state["state"] == "na":
            state.update(state="idle", detail="")
        self.test_widgets["telem3"]["test"].setEnabled(fitted)

    @staticmethod
    def test_name(spec):
        return f"{spec['connector']} {spec['function']}"

    def test_device(self, spec):
        if spec["devices"] == "telem":
            return self.telem_device_box.currentData()
        widget = self.test_widgets[spec["key"]]["device"]
        return widget.currentData() if isinstance(widget, QComboBox) else spec["devices"][0]

    def test_pairs(self, spec, dev_key):
        dev = TEST_DEVICES[dev_key]
        pairs = []
        n = spec.get("serial")
        if n is not None and "protocol" in dev:
            pairs.append((f"SERIAL{n}_PROTOCOL", dev["protocol"]))
            if "baud" in dev:
                pairs.append((f"SERIAL{n}_BAUD", dev["baud"]))
            # the device must only be configured on the port under test, so its data proves this port works
            for other in PORT_TESTS:
                m = other.get("serial")
                if m is None or m == n or other["devices"] != spec["devices"]:
                    continue
                if self.param_int(f"SERIAL{m}_PROTOCOL") == dev["protocol"]:
                    for field, default in TELEM_DEFAULT.items():
                        name = f"SERIAL{m}_{field}"
                        pairs.append((name, self.port_backup.get(name, default)))
        return pairs + dev["params"]

    def run_test(self, key):
        spec = TESTS_BY_KEY[key]
        state = self.tests[key]
        if self.job or (self.active_test and self.active_test != key):
            self.notify("Another test or reboot is running")
            return
        if spec.get("optional") and not self.telem3_fitted.isChecked():
            return
        if not self.target:
            self.notify("Connect a board first")
            return
        if not self.params_ready():
            self.notify("Wait until the parameters are read")
            return
        dev_key = self.test_device(spec)
        dev = TEST_DEVICES[dev_key]
        if dev["check"] is None:
            hint = spec.get("hint") or f"Plug the module into {spec['connector']} and mark PASS or FAIL"
            state.update(state="manual", detail=hint)
            return
        changes = self.param_changes(self.test_pairs(spec, dev_key))
        if isinstance(changes, str):
            self.set_test_result(key, "fail", changes)
            return
        for name, _ in changes:
            if name in self.params and name.startswith("SERIAL"):
                self.port_backup.setdefault(name, self.params[name])
        self.active_test = key
        if changes or dev.get("reboot"):
            what = f"writing {len(changes)} parameter(s) and rebooting" if changes else "rebooting to probe it"
            state.update(state="configuring", detail=f"Plug {dev['name']} into {spec['connector']}; {what}")
            if changes:
                self.send_param_sets(changes)
            self.start_job([n for n, _ in changes], lambda k=key: self.begin_test_wait(k),
                           f"{self.test_name(spec)} test")
        else:
            self.begin_test_wait(key)

    def begin_test_wait(self, key):
        spec = TESTS_BY_KEY[key]
        dev = TEST_DEVICES[self.test_device(spec)]
        self.active_test = key
        self.tests[key].update(state="waiting", deadline=time.monotonic() + TEST_TIMEOUT_S,
                               detail=f"Waiting for {dev['name']} on {self.test_name(spec)}")
        if dev["check"] == "eth":
            self.start_eth_probe(key)

    def start_eth_probe(self, key):
        """The board sends MAVLink to NET_P1_IP:NET_P1_PORT as a UDP client; this PC must own that address."""
        if not self.param_int("NET_ENABLE"):
            self.set_test_result(key, "fail", "NET_ENABLE is 0, enable networking in Settings \u2192 General")
            return
        if self.param_int("NET_P1_TYPE") != 1:
            self.set_test_result(key, "fail", "Set Network Port 1 to UDP Client (Settings \u2192 General)")
            return
        port = self.param_int("NET_P1_PORT") or 14550
        ip = ".".join(str(self.param_int(f"NET_P1_IP{i}")) for i in range(4))
        self.tests[key]["detail"] = f"Listening on UDP {port}; this PC must be {ip} on the Ethernet cable"
        self.stop_eth_probe()
        self.eth_probe = UdpProbe(port, self.target[0], TEST_TIMEOUT_S)
        self.eth_probe.result.connect(lambda ok, text, k=key: self.on_eth_result(k, ok, text))
        self.eth_probe.start()

    def on_eth_result(self, key, ok, text):
        if self.tests[key]["state"] == "waiting":
            self.set_test_result(key, "pass" if ok else "fail", text)

    def stop_eth_probe(self):
        probe, self.eth_probe = self.eth_probe, None
        if probe:
            probe.blockSignals(True)
            probe.stop()
            probe.wait(2000)

    def test_check(self, check):
        """Text describing the device data when the check passes, else None."""
        if check == "flow":
            if self.sensor_healthy(mavlink.MAV_SYS_STATUS_SENSOR_OPTICAL_FLOW):
                value = self.live.get(("flow", 0))
                return "Optical flow healthy" + (f"   {value[1]}" if value else "")
        elif check == "range":
            healthy = self.sensor_healthy(mavlink.MAV_SYS_STATUS_SENSOR_LASER_POSITION)
            if healthy or self.fresh("range"):
                value = next((v for (k, _), v in self.live.items() if k == "range"), None)
                return "Rangefinder data" + (f"   {value[1]}" if value else "")
        elif check == "i2c_mag":
            key = self.active_test
            bus = TESTS_BY_KEY[key]["i2c_bus"] if key else None
            for inst in range(3):
                dev = self.device_id("mag", inst)
                if dev and dev & 0x07 == 1 and (dev >> 3) & 0x1F == bus:
                    info = decode_device_id(dev, "mag", self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA)
                    return f"Compass {inst + 1}: {info['chip']} on {info['bus']} {info['address']}"
        elif check == "rc":
            rc = self.rc
            if rc and time.monotonic() - rc["t"] <= STALE_S and any(800 < v < 2200 for v in rc["chans"]):
                return f"{len(rc['chans'])} channels received" + (f", RSSI {rc['rssi']}" if rc["rssi"] != 255 else "")
        elif check == "usb":
            ps = self.power_status
            if ps and time.monotonic() - ps["t"] <= STALE_S and ps["flags"] & mavlink.MAV_POWER_STATUS_USB_CONNECTED:
                return "Board reports USB connected"
        elif check == "sd":
            state = self.sys_status_state(mavlink.MAV_SYS_STATUS_LOGGING)
            if state and state[0] and state[2]:
                return "microSD present, logging healthy"
        elif check in ("gps1", "gps2"):
            info = self.gps_info.get(0 if check == "gps1" else 1)
            if info and info["fix_type"] >= 1 and time.monotonic() - info["t"] <= STALE_S:
                sats = info["sats"] if info["sats"] is not None else "-"
                return f"GPS detected, fix {info['fix']}, {sats} satellites"
        return None

    def check_active_test(self):
        key = self.active_test
        if not key or self.tests[key]["state"] != "waiting":
            return
        dev = TEST_DEVICES[self.test_device(TESTS_BY_KEY[key])]
        if dev["check"] == "eth":
            # decided by the UDP probe thread; this only catches a probe that never answers
            if time.monotonic() > self.tests[key]["deadline"] + 5:
                self.stop_eth_probe()
                self.set_test_result(key, "fail", "Ethernet probe did not finish")
            return
        found = self.test_check(dev["check"])
        if found:
            self.set_test_result(key, "pass", found)
        elif time.monotonic() > self.tests[key]["deadline"]:
            self.set_test_result(key, "fail", f"No data from {dev['name']} within {TEST_TIMEOUT_S:.0f} s")

    def set_test_result(self, key, result, detail):
        self.tests[key].update(state=result, detail=detail, deadline=0.0)
        if self.active_test == key:
            self.active_test = None
        if result in ("pass", "fail"):
            name = self.test_name(TESTS_BY_KEY[key])
            self.add_log(mavlink.MAV_SEVERITY_INFO if result == "pass" else mavlink.MAV_SEVERITY_ERROR,
                         f"Port test {name}: {result.upper()} ({detail})", persist=True)

    def reset_tests(self):
        self.stop_eth_probe()
        if self.active_test:
            self.cancel_job("Cancelled")
        self.active_test = None
        for key, state in self.tests.items():
            state.update(state="idle", detail="", deadline=0.0)
        self.on_telem3_fitted(self.telem3_fitted.isChecked())

    def restore_ports(self):
        if not self.port_backup:
            self.notify("No port parameters were changed by the tests")
            return
        backup = dict(self.port_backup)
        text, _ = self.apply_and_reboot(list(backup.items()), "Restore ports", on_ready=self.port_backup.clear)
        self.notify(text)

    def refresh_tests(self):
        now = time.monotonic()
        labels = {"idle": ("Not tested", C["muted"]), "configuring": ("Configuring", C["warn"]),
                  "waiting": ("Waiting", C["accent"]), "manual": ("Check by hand", C["accent"]),
                  "pass": ("PASS", C["ok"]), "fail": ("FAIL", C["err"]), "na": ("N/A – PASS", C["ok"])}
        passed = 0
        for key, widgets in self.test_widgets.items():
            state = self.tests[key]
            text, color = labels[state["state"]]
            if state["state"] == "configuring" and self.job and self.job["stage"] != "writing":
                text = "Rebooting"
            if state["state"] == "waiting":
                text = f"Waiting {max(0, state['deadline'] - now):.0f}s"
            set_label(widgets["result"], text, color)
            widgets["result"].setStyleSheet(f"color: {color}; font-weight: 700;")
            set_label(widgets["detail"], state["detail"], C["muted"])
            passed += state["state"] in ("pass", "na")
        color = C["ok"] if passed == len(PORT_TESTS) else C["text"]
        set_label(self.test_summary, f"{passed} / {len(PORT_TESTS)} passed", color)
        self.restore_ports_btn.setText(f"Restore Ports ({len(self.port_backup)})" if self.port_backup
                                       else "Restore Ports")

    # ---- test report ------------------------------------------------------------------------

    def build_report_card(self, lay):
        card = Card("Test Report", "PDF")
        card.body.addWidget(muted_label(
            "Creates a PDF with the board information, detected sensors and serial ports, the onboard hardware "
            "table, the port test results and the live data. The date is added automatically; you enter your "
            "name and the board ID.", wrap=True))
        row = QHBoxLayout()
        self.report_status = muted_label(wrap=True)
        self.report_btn = QPushButton("Create Report")
        self.report_btn.setObjectName("primary")
        self.report_btn.clicked.connect(self.create_report)
        row.addWidget(self.report_status, 1)
        row.addWidget(self.report_btn)
        card.body.addLayout(row)
        lay.addWidget(card)
        self.report_operator = ""
        self.report_dir = os.path.expanduser("~")

    def create_report(self):
        if not self.target and QMessageBox.question(
                self, "Create Test Report", "No board is connected, the report will have no board data. "
                                            "Create it anyway?") != QMessageBox.Yes:
            return
        dialog = ReportDialog(self, self.report_operator)
        if dialog.exec() != QDialog.Accepted:
            return
        operator, board_id = dialog.values()
        self.report_operator = operator
        now = datetime.datetime.now()
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", board_id)
        default = os.path.join(self.report_dir, f"BoardTest_{safe_id}_{now:%Y%m%d_%H%M%S}.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "Save Test Report", default, "PDF files (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        self.report_dir = os.path.dirname(path)
        try:
            self.write_report_pdf(path, operator, board_id, now)
        except Exception as e:
            set_label(self.report_status, f"Could not write the report: {e}", C["err"])
            return
        set_label(self.report_status, f"Saved {path}", C["ok"])
        self.add_log(mavlink.MAV_SEVERITY_INFO, f"Test report saved: {path}")

    def test_result_counts(self):
        states = [self.tests[t["key"]]["state"] for t in PORT_TESTS]
        passed = sum(st in ("pass", "na") for st in states)
        failed = sum(st == "fail" for st in states)
        return passed, failed, len(states)

    def report_sections(self):
        """(title, headers, rows, note) for each report table; rows are lists of (text, color) cells."""
        muted = C["muted"]
        sections = []

        board = [[(key, muted), (label.text(), None)] for key, label in self.board_card.values.items()]
        board.append([("Connection", muted), (f"{self.port_device or '-'} @ {self.port_baud or '-'}", None)])
        sections.append(("Board Information", ["Item", "Value"], board, ""))

        sections.append(("Detected Sensors", ["Sensor", "Chip", "Bus", "Address / Port", "Reading"],
                         self.report_sensor_rows(), ""))

        ports = []
        for n, (connector, uart) in SERIAL_PORTS.items():
            proto = self.param_value(f"SERIAL{n}_PROTOCOL")
            baud = self.param_value(f"SERIAL{n}_BAUD")
            proto_text = SERIAL_PROTOCOLS.get(int(proto), f"Protocol {int(proto)}") if proto is not None else "-"
            baud_text = str(SERIAL_BAUDS.get(int(baud), baud)) if baud is not None else "-"
            ports.append([(f"SERIAL{n}", None), (connector, None), (uart, muted), (proto_text, None),
                          (baud_text, None)])
        sections.append(("Serial Ports", ["Serial", "Connector", "UART", "Protocol", "Baud"], ports, ""))

        onboard = [[(item, None), (iface, muted), (status, color), (details, None)]
                   for item, iface, status, color, details in self.onboard_rows()]
        sections.append(("Onboard Hardware", ["Item", "Interface", "Status", "Details"], onboard, ""))

        labels = {"idle": "NOT TESTED", "configuring": "RUNNING", "waiting": "RUNNING", "manual": "NOT CHECKED",
                  "pass": "PASS", "fail": "FAIL", "na": "N/A (PASS)"}
        colors = {"pass": C["ok"], "na": C["ok"], "fail": C["err"]}
        tests = []
        for spec in PORT_TESTS:
            state = self.tests[spec["key"]]
            widget = self.test_widgets[spec["key"]]["device"]
            device = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            tests.append([(spec["connector"], None), (spec["function"], muted), (device, None),
                          (labels[state["state"]], colors.get(state["state"], C["warn"])), (state["detail"], None)])
        passed, failed, total = self.test_result_counts()
        sections.append(("Port Test Results", ["Connector", "Function", "Device", "Result", "Details"], tests,
                         f"{passed} of {total} passed, {failed} failed."))

        return sections

    def report_sensor_rows(self):
        """One row per sensor: IMUs as a whole (accel + gyro), compasses, barometers, battery and GPS."""
        now = time.monotonic()
        muted = C["muted"]
        ardupilot = self.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        rows = []

        def reading(key):
            entry = self.live.get(key)
            if not entry:
                return "No data", C["err"]
            return entry[1], C["err"] if now - entry[2] > STALE_S else None

        def row(name, infos, value, color=None):
            chip, bus, addr = (" / ".join(dict.fromkeys(i[field] for i in infos)) or "-"
                               for field in ("chip", "bus", "address"))
            rows.append([(name, None), (chip, None), (bus, None), (addr, None), (value, color)])

        for inst in range(3):
            infos = [decode_device_id(dev, kind, ardupilot)
                     for kind, dev in ((k, self.device_id(k, inst)) for k in ("accel", "gyro")) if dev]
            if infos:
                row(f"IMU {inst + 1}", infos, *reading(("temp", inst)))
        for inst in range(3):
            dev = self.device_id("mag", inst)
            if not dev:
                continue
            field = self.mag_field.get(inst)
            if field is None:
                value, color = reading(("mag", inst))
            else:
                value = f"{field:.0f} mGauss (EMI {MAG_FIELD_MIN}-{MAG_FIELD_MAX})"
                color = C["ok"] if MAG_FIELD_MIN <= field <= MAG_FIELD_MAX else C["err"]
            row(f"Compass {inst + 1}", [decode_device_id(dev, "mag", ardupilot)], value, color)
        for inst in range(3):
            dev = self.device_id("baro", inst)
            if dev:
                row(f"Barometer {inst + 1}", [decode_device_id(dev, "baro", ardupilot)], *reading(("baro", inst)))

        batt = self.param_value("BATT_MONITOR")
        batt = int(batt) if batt is not None else None
        chip = "INA238" if batt == 21 else ("-" if batt is None else f"BATT_MONITOR {batt}")
        pw = self.power if self.power and now - self.power["t"] <= STALE_S else None
        for label, key, unit in (("Battery Voltage", "voltage", "V"), ("Battery Current", "current", "A")):
            value = pw.get(key) if pw else None
            text = f"{value:.2f} {unit}" if value is not None else ("Not measured" if pw else "No data")
            color = None if value is not None else (muted if pw else C["err"])
            rows.append([(label, None), (chip, None), ("I2C 1" if batt == 21 else "-", None), ("-", None),
                         (text, color)])

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
                bus, port = "Serial", ports[inst] if inst < len(ports) else "-"
            if info:
                sats = f", {info['sats']} sats" if info["sats"] is not None else ""
                fix_color = C["ok"] if info["fix_type"] >= 3 else C["warn"] if info["fix_type"] == 2 else C["err"]
                value, color = f"{info['fix']}{sats}", C["err"] if now - info["t"] > STALE_S else fix_color
            else:
                value, color = "No data", C["err"]
            rows.append([(f"GPS {inst + 1}", None), (chip, None), (bus, None), (port, None), (value, color)])
        return rows

    def report_html(self, operator, board_id, when):
        # screen colours are for a dark theme; use darker ones on paper
        paper = {C["ok"]: "#15803d", C["err"]: "#b91c1c", C["warn"]: "#b45309", C["accent"]: "#1d4ed8",
                 C["muted"]: "#6b7280"}
        esc = html.escape
        passed, failed, total = self.test_result_counts()
        if failed:
            verdict, verdict_color = "FAIL", "#b91c1c"
        elif passed == total:
            verdict, verdict_color = "PASS", "#15803d"
        else:
            verdict, verdict_color = "INCOMPLETE", "#b45309"
        firmware = self.version.get("firmware", self.version.get("fw_number", "-"))
        uid = self.version.get("uid", "-")
        out = [f"""<html><head><style>
            body {{ font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 9pt; color: #111827; }}
            h1 {{ font-size: 18pt; margin: 0; }}
            h2 {{ font-size: 12pt; margin-top: 16px; margin-bottom: 4px; color: #1d4ed8; }}
            th {{ background: #e5e7eb; text-align: left; font-weight: bold; }}
            td.k {{ color: #6b7280; }}
            .note {{ color: #6b7280; }}
            </style></head><body>
            <h1>Board Test Report</h1>
            <p class="note">{esc(BOARD_NAME)} &middot; generated by {esc(APP_NAME)}</p>
            <table width="100%" cellspacing="0" cellpadding="5" border="1" style="border-collapse: collapse;">
            <tr><td class="k" width="18%">Board ID</td><td width="32%"><b>{esc(board_id)}</b></td>
                <td class="k" width="18%">Result</td>
                <td width="32%"><b style="color: {verdict_color}">{verdict}</b> ({passed}/{total} passed)</td></tr>
            <tr><td class="k">Tested by</td><td>{esc(operator)}</td>
                <td class="k">Date</td><td>{when:%Y-%m-%d %H:%M:%S}</td></tr>
            <tr><td class="k">Firmware</td><td>{esc(firmware)}</td><td class="k">UID</td><td>{esc(uid)}</td></tr>
            </table>"""]
        for number, (title, headers, rows, note) in enumerate(self.report_sections(), 1):
            out.append(f"<h2>{number}. {esc(title)}</h2>")
            if rows:
                out.append('<table width="100%" cellspacing="0" cellpadding="4" border="1" '
                           'style="border-collapse: collapse;"><tr>')
                out.extend(f"<th>{esc(h)}</th>" for h in headers)
                out.append("</tr>")
                for row in rows:
                    out.append("<tr>")
                    for text, color in row:
                        style = f' style="color: {paper.get(color, color)}"' if color else ""
                        out.append(f"<td{style}>{esc(str(text))}</td>")
                    out.append("</tr>")
                out.append("</table>")
            if note:
                out.append(f'<p class="note">{esc(note)}</p>')
        out.append("</body></html>")
        return "".join(out)

    def write_report_pdf(self, path, operator, board_id, when):
        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QPageSize.A4))
        writer.setPageMargins(QMarginsF(12, 12, 12, 12), QPageLayout.Millimeter)
        writer.setTitle(f"Board Test Report {board_id}")
        writer.setCreator(APP_NAME)
        doc = QTextDocument()
        doc.setHtml(self.report_html(operator, board_id, when))
        doc.print_(writer)

    # ---- shutdown ---------------------------------------------------------------------------

    def shutdown(self):
        if self.closing:
            return
        self.closing = True
        self.job = None
        self.stop_eth_probe()
        for timer in (self.job_timer, self.port_timer):
            timer.stop()
        self.disconnect_board()
        self.port_timer.stop()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def log_path():
    return os.path.join(LOG_DIR, time.strftime("board_test_%Y%m%d.log"))


def setup_file_log():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        handler = logging.FileHandler(log_path(), encoding="utf-8")
    except OSError as exc:
        print(f"Log file disabled: {exc}", file=sys.stderr)
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    file_log.addHandler(handler)
    file_log.setLevel(logging.INFO)
    file_log.propagate = False
    file_log.info("---- %s started ----", APP_NAME)


def install_shutdown_handlers(app, window):
    def on_signal(*_):
        window.close()
        app.quit()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    # Python only runs signal handlers when the interpreter gets control, so wake it up periodically
    wake = QTimer(app)
    wake.timeout.connect(lambda: None)
    wake.start(200)
    app.aboutToQuit.connect(window.shutdown)

    def on_exception(exc_type, exc, tb):
        try:
            first = window.log_exception(exc_type, exc, tb)
        except Exception:
            traceback.print_exc()
            first = True
        if first:
            traceback.print_exception(exc_type, exc, tb)
        window.notify(f"Internal error: {exc_type.__name__}: {exc} (see {log_path()})", 0)

    sys.excepthook = on_exception


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    setup_file_log()
    window = MainWindow()
    install_shutdown_handlers(app, window)
    window.show()
    sys.exit(app.exec())
