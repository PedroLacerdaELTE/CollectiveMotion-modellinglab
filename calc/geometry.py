import numpy as np


def get_curvature_from_trajectory(vx, vy, ax, ay):

    # (vx.ay - vy.ax) / (vx^2 + vy^2)^1.5
    if np.isclose(vx, 0) and np.isclose(vy, 0): # vx**2 + vy**2 == 0:
        return 0
    return (vx * ay - vy * ax) / (vx**2 + vy**2)**1.5


def pf_get_curvature_from_trajectory(row, vx_col, vy_col, ax_col, ay_col):
    return get_curvature_from_trajectory(row[vx_col], row[vy_col], row[ax_col], row[ay_col])


def get_radius_from_curvature(curvature):
    if np.isclose(curvature, 0):
        return np.inf
    else:
        #return 1/np.abs(curvature)
        return 1 / curvature


def pf_get_radius_from_curvature(row, curvature_col):
    return get_radius_from_curvature(row[curvature_col])


def radial_velocity_from_cartesian(x, y, vx, vy):
    return (x * vx + y * vy) / np.linalg.norm([x, y])


def angular_velocity_from_cartesian(x, y, vx, vy):
    return (vy * x - vx * y) / np.linalg.norm([x, y])


def get_cartesian_velocity_on_rotating_frame_from_inertial_frame(x, y, vx, vy):
    v_radial = radial_velocity_from_cartesian(x, y, vx, vy)
    v_angular = angular_velocity_from_cartesian(x, y, vx, vy)
    return v_radial, v_angular


def radial_acceleration_from_cartesian(x, y, vx, vy, ax, ay):
    rho = np.linalg.norm([x, y])
    radial_velocity = radial_velocity_from_cartesian(x, y, vx, vy)
    return (vx ** 2 + vy ** 2 + x * ax + y * ay - radial_velocity ** 2) / rho

