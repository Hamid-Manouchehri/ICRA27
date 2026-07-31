"""
Move a UR5e to a user-defined joint-space home configuration.

Inputs:
    UR5e IP address and six home joint angles in degrees.

Outputs:
    One joint-space moveJ command through the ur_rtde control interface.

User-adjustable parameters:
    Edit robot_ip, home_joint_q_deg, joint_speed, and joint_acceleration below.
    Joint angles must be inside the robot's joint limits. Start with a slow
    speed of 0.1 to 0.5 rad/s and acceleration of 0.1 to 1.0 rad/s^2.

Important:
    This is a joint-space move to an upright pose, not encoder homing. Verify
    the pose, speed, workspace, and emergency-stop access before running it.

Dependency:
    pip install --user ur_rtde
"""

import math

try:
    import rtde_control
except ImportError:
    rtde_control = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP/hostname [valid robot address]

# Standard upright UR5e home pose. Verify for the installed robot and tool.
home_joint_q_deg = [  # TODO home joint angles [deg], within joint limits
    0.0,
    -90.0,
    0.0,
    -90.0,
    0.0,
    0.0,
]
home_joint_q = [
    math.radians(angle_deg)
    for angle_deg in home_joint_q_deg
]

joint_speed = 0.25  # TODO moveJ speed [rad/s], recommended 0.1 to 0.5
joint_acceleration = 0.5  # TODO moveJ acceleration [rad/s^2], 0.1 to 1.0


def main():
    """Connect to the UR5e and execute the configured home move."""

    if rtde_control is None:
        print("Missing dependency: install with `pip install --user ur_rtde`")
        return

    print("UR5e home joint angles [deg]:")
    print(home_joint_q_deg)
    print("UR5e home joint angles [rad]:")
    print(home_joint_q)

    print("Connecting to UR5e at", robot_ip)
    rtde_control_interface = rtde_control.RTDEControlInterface(robot_ip)

    try:
        print("Moving UR5e to home configuration...")
        rtde_control_interface.moveJ(
            home_joint_q,
            joint_speed,
            joint_acceleration,
        )
    finally:
        rtde_control_interface.stopScript()

    print("UR5e home move completed.")


if __name__ == "__main__":
    main()
