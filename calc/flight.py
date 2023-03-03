import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings('error')


def get_radius_from_bank_angle(bank_angle, mass, wing_area, CL, rho=1.22, g=9.8):
    if np.isclose(bank_angle, 0):
        return np.inf
    radius = 2 * mass / (rho * wing_area * CL * np.sin(bank_angle))
    return radius


def get_bank_angle_from_radius(radius, mass, wing_area, CL, CD, rho=1.225, g=9.8):
    # rho kg/m3
    # CL -> Goksel
    # wing_load kg/m2 https://bybio.wordpress.com/tag/wing-loading/

    # Vh = get_horizontal_velocity_from_bird_parameters(mass, wing_area, CL, rho=rho)
    if radius == np.inf:
        return 0.0
    # elif np.abs(2 * wing_load/(rho * CL * row[radius_col])) > 1:
    #    return np.pi/2
    else:
        try:
            ba = np.arcsin(2 * mass / (rho * wing_area * CL * radius))
            # ba = np.arctan( Vh**2 / (radius * g))
        except RuntimeWarning as e:
            raise RuntimeWarning
        else:
            return ba


def get_CD_from_CL(CL):
    return 0.020 + 0.0371 * CL + 0.0155 * CL ** 2


def get_horizontal_velocity_from_bird_parameters(bank_angle, mass, wing_area, CL, rho=1.225, g=9.8):
    wing_load = mass * g / wing_area
    try:
        result = np.sqrt(2 * wing_load / (rho * CL * np.cos(bank_angle)))
    except RuntimeWarning:
        print(wing_load, rho, CL, bank_angle, np.cos(bank_angle))
    return result


def get_leveled_sink_rate_from_bird_parameters(mass, wing_area, CL, CD, rho=1.225, g=9.8):
    leveled_horizontal_velocity = get_horizontal_velocity_from_bird_parameters(bank_angle=0, mass=mass,
                                                                               wing_area=wing_area,
                                                                               CL=CL, rho=rho, g=g)
    return CD / CL * leveled_horizontal_velocity


def get_min_sink_rate_from_bank_angle(bank_angle, mass, wing_area, CL, CD, rho=1.225, g=9.8):
    leveled_min_sink_rate = get_leveled_sink_rate_from_bird_parameters(mass, wing_area, CL, CD, rho=rho, g=g)

    if bank_angle == np.nan:
        return np.nan
    else:
        return leveled_min_sink_rate / (np.cos(bank_angle) ** (3 / 2))  # cos is always positive
