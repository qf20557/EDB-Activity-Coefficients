"""Stage 1: refractive-index correction of the EDB radius."""
import math

from .config import EDB_ASSUMED_RI, MEDIAN_OBS_ANGLE


def ri_term(angle, ri):
    h = angle / 2.0
    return math.cos(h) + ri * math.sin(h) / math.sqrt(1 + ri ** 2 - 2 * ri * math.cos(h))


def correction_factor(new_ri, previous_ri=EDB_ASSUMED_RI, angle=MEDIAN_OBS_ANGLE):
    return ri_term(angle, previous_ri) / ri_term(angle, new_ri)
