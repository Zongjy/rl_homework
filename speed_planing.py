import numpy as np


class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        vt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return vt

    def calc_second_derivative(self, t):
        at = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return at

    def calc_third_derivative(self, t):
        jerkt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return jerkt


def quintic_polynomials_planner(dt):
    s_rough = [0, 20, 40, 60, 74, 121]
    v_rough = [10, 10.2, 9.2, 10.4, 9.2, 10.9]
    a_rough = [0, -0.1, -0.1, 0.1, -0.1, 0]
    t_rough = [0, 1.7, 4.1, 6.1, 7.6, 12.1]

    time, s_smooth, v_smooth, a_smooth, jerk_smooth = [], [], [], [], []

    for i in range(1, 6):
        sqp = QuinticPolynomial(s_rough[i - 1], v_rough[i - 1], a_rough[i - 1], s_rough[i], v_rough[i], a_rough[i],
                                t_rough[i] - t_rough[i - 1])
        for t in np.arange(t_rough[i - 1], t_rough[i] + dt, dt):
            time.append(t)
            delta_t = t - t_rough[i - 1]
            s_smooth.append(sqp.calc_point(delta_t))
            v_smooth.append(sqp.calc_first_derivative(delta_t))
            a_smooth.append(sqp.calc_second_derivative(delta_t))
            jerk_smooth.append(sqp.calc_third_derivative(delta_t))
    return np.array(time), np.array(s_smooth), np.array(v_smooth), np.array(a_smooth), np.array(jerk_smooth)
