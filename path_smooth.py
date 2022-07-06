from scipy.interpolate import UnivariateSpline


def smooth(x, y):
    s = UnivariateSpline(x, y, s=1.2)
    return s

