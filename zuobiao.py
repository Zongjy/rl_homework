import math

from pynverse import inversefunc
from scipy import special


def f(x, x0, y0):
    return 2 * (x - x0) + 0.04 * math.pi * math.cos(0.02 * math.pi * x) * (math.sin(0.02 * math.pi * x) - y0)


def fd(x, x0, y0):
    return 2 - 0.0008 * (math.pi ** 2) * (math.sin(0.02 * math.pi * x) * (math.sin(0.02 * math.pi * x) - y0) + (
            math.cos(0.02 * math.pi * x) ** 2))


def cartesian_to_frenet(xe, ye):
    eps = 0.0001
    x1 = xe
    while abs(f(x1, xe, ye)) > eps:
        x2 = x1 - f(x1, xe, ye) / fd(x1, xe, ye)
        if abs(x2 - x1) < eps:
            break
        x1 = x2
    s = 50 * math.sqrt(2) * special.ellipeinc(0.02 * math.pi * x1, 1 / 2) / math.pi
    l = math.sqrt((x1 - xe) ** 2 + (math.sin(0.02 * math.pi * x1) - ye) ** 2)
    return s, l


def frenet_to_cartesian(s):
    cube = (lambda x: 50 * math.sqrt(2) * special.ellipeinc(0.02 * math.pi * x, 1 / 2) / math.pi)
    invcube = inversefunc(cube)
    return invcube(s)
