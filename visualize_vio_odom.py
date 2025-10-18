#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from math import atan2, asin

def euler_from_quaternion(x, y, z, w):
    """Convert quaternion to roll, pitch, yaw (radians)."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = atan2(t3, t4)

    return roll, pitch, yaw

def main():
    if len(sys.argv) < 2:
        print("Usage: ./visualize_odom.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Convert to numpy arrays to avoid pandas multi-dim issues
    time = df['time'].to_numpy()
    pos_x = df['pos_x'].to_numpy()
    pos_y = df['pos_y'].to_numpy()
    pos_z = df['pos_z'].to_numpy()
    vel_x = df['vel_linear_x'].to_numpy()
    vel_y = df['vel_linear_y'].to_numpy()
    vel_z = df['vel_linear_z'].to_numpy()
    pose_var_x = df['pose_var_x'].to_numpy()
    pose_var_y = df['pose_var_y'].to_numpy()
    pose_var_z = df['pose_var_z'].to_numpy()
    vel_var_x = df['vel_var_x'].to_numpy()
    vel_var_y = df['vel_var_y'].to_numpy()
    vel_var_z = df['vel_var_z'].to_numpy()

    # Compute roll, pitch, yaw
    rolls, pitches, yaws = [], [], []
    for i in range(len(df)):
        r, p, y = euler_from_quaternion(
            df['orient_x'][i],
            df['orient_y'][i],
            df['orient_z'][i],
            df['orient_w'][i]
        )
        rolls.append(r)
        pitches.append(p)
        yaws.append(y)
    rolls = np.array(rolls)
    pitches = np.array(pitches)
    yaws = np.array(yaws)

    # Compute horizontal distance
    horiz_dist = np.sqrt(pos_x**2 + pos_y**2)

    pdf_name = csv_file.replace('.csv', '_plots.pdf')
    with PdfPages(pdf_name) as pdf:
        # 1. Position
        plt.figure()
        plt.plot(time, pos_x, label='x')
        plt.plot(time, pos_y, label='y')
        plt.plot(time, pos_z, label='z')
        plt.title('Position over Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 2. Linear Velocity
        plt.figure()
        plt.plot(time, vel_x, label='x')
        plt.plot(time, vel_y, label='y')
        plt.plot(time, vel_z, label='z')
        plt.title('Linear Velocity over Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 3. Position Variance
        plt.figure()
        plt.plot(time, pose_var_x, label='x var')
        plt.plot(time, pose_var_y, label='y var')
        plt.plot(time, pose_var_z, label='z var')
        plt.title('Position Variance')
        plt.xlabel('Time [s]')
        plt.ylabel('Variance')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 4. Velocity Variance
        plt.figure()
        plt.plot(time, vel_var_x, label='x var')
        plt.plot(time, vel_var_y, label='y var')
        plt.plot(time, vel_var_z, label='z var')
        plt.title('Velocity Variance')
        plt.xlabel('Time [s]')
        plt.ylabel('Variance')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 5. Roll, Pitch, Yaw
        plt.figure()
        plt.plot(time, rolls, label='roll')
        plt.plot(time, pitches, label='pitch')
        plt.plot(time, yaws, label='yaw')
        plt.title('Orientation (Roll, Pitch, Yaw)')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [rad]')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 6. Horizontal distance vs time
        plt.figure()
        plt.plot(time, horiz_dist, label='Horizontal distance')
        plt.title('Horizontal Distance over Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

        # 7. XY trajectory (2D odometry path)
        plt.figure()
        plt.plot(pos_x, pos_y, label='Trajectory', color='blue')
        plt.title('2D Odometry Path (X-Y Plane)')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()

    print(f"✅ Saved plots to: {pdf_name}")

if __name__ == "__main__":
    main()
