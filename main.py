import numpy as np
from matplotlib import pyplot as plt

from path_smooth import smooth
from rtdp import RTDP, evaluate
from speed_planing import quintic_polynomials_planner
from roadenv import roadenv
from zuobiao import frenet_to_cartesian

obstac_x = np.array([2 * 15, 2 * 30])
obstac_y = np.array([6 * 0.2, -5 * 0.2])
obstac_y += np.sin(0.04 * np.pi * obstac_x)


def run():
    env = roadenv()
    rtdp = RTDP(env)
    x, y = evaluate(rtdp, env)
    curve = smooth(2 * x, 0.2 * (y - 10))
    t, s, v, a, jerk = quintic_polynomials_planner(0.1)
    # plt.plot(t, s, 'r')
    # plt.plot(t, v, 'b-.')
    # plt.plot(t, a / 4, 'k.')
    SIM_LOOP = 121
    x_area = 20.0
    y_area = 5.0
    reference_x = np.arange(0, 100, 0.5)
    reference_y = np.sin(0.02 * np.pi * reference_x)
    left_border = reference_y + 2.0
    right_border = reference_y - 2.0
    x_t = frenet_to_cartesian(s)
    y_t = curve(x_t)
    for i in range(SIM_LOOP):
        plt.cla()
        # 按esc退出
        plt.plot(reference_x, curve(reference_x))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(reference_x, reference_y, 'c-.', linewidth=0.7)
        plt.plot(reference_x, left_border, 'b', reference_x, right_border, 'b')
        plt.scatter(obstac_x, obstac_y, marker='o', s=477, c='k')
        plt.plot(x_t[i:], y_t[i:], 'or', ms=3)
        plt.xlim(x_t[i] - x_area, x_t[i] + x_area)
        plt.ylim(y_t[i] - y_area, y_t[i] + y_area)
        plt.title("v[km/h]:" + str(v[i] * 3.6)[0:4])
        plt.grid(True)
        plt.pause(0.01)


if __name__ == '__main__':
    run()
