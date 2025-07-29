import numpy as np
from mpmath import *
from matplotlib import pyplot as plt

def g_fun_gamma(x, a: float =3, lambd: float = 0.0478):
    z = float(lambd) * float(x)
    w = gammainc(a, 0, z, regularized=True)
    ww = 1 - w
    u = fadd(a, 1)
    u1 = fsub(a, 1)
    v = fdiv(u1, 2)
    y = power(ww, v)
    r = 1 - y
    gn = fmul(a, u)
    gd = float(lambd) * float(fsub(a, 1))
    g1 = fdiv(gn, gd)
    result = g1 * r
    return result

def get_inflection(time, gt):
    dg_dt = np.diff(gt)
    idx_max = np.argmax(dg_dt)
    t_start = time[idx_max]
    return t_start, gt[idx_max]


def run_gamma_parametric_example(lambd = 0.0478):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    Time = np.linspace(0, 100, 1000)
    a_vals = [1.1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    colors = plt.cm.Reds(np.linspace(0, 1, len(a_vals)))
    for a,  col in zip(a_vals, colors):
        G = np.array([float(g_fun_gamma(x, a=a, lambd=lambd)) for x in Time])
        plt.plot(Time, G,  label=f'shape: {a}', color= col)
        t_max, g_max = get_inflection(Time, G)
        plt.scatter(t_max, g_max, color=col)
    plt.legend()
    plt.grid()
    plt.xlabel(r'$t$ [h]')
    plt.ylabel(r'$g(t)$ [h]')

    plt.savefig('../plots/g for Gamma.png')
    plt.savefig('../plots/g for gamma.pdf')

    plt.show()

if __name__ == '__main__':
    run_gamma_parametric_example()