import numpy as np
import matplotlib.pyplot as plt

# from lifeline get data loader and KM fitter
from lifelines import *
from lifelines.datasets import load_waltons
from lifelines.utils import restricted_mean_survival_time

# get Time Warper
from rul_timewarping.timewarping import TimeWarping


def get_g_from_lifelines(survival_model, TimeWarp, TTF):
    """ get g function from lifelines survival model"""
    # get KM estimator of the reliability function
    R_vals = survival_model.survival_function_[survival_model._label].values
    # Mean and Variance of survival time from censored data
    mu, var = restricted_mean_survival_time(survival_model,  t=TTF.max(),  return_variance=True)
    # Compute coefficient of variation and shape parameter k
    # cv = std / mean;  k = 1 - cv**2 / (1 + cv**2)
    cv = np.sqrt(var) / mu
    k = (1 - cv ** 2) / (1 + cv ** 2)

    # Compute time transformation function T = g(t) that makes MRL linear in T
    return TimeWarp.compute_g_fun(R=R_vals, k=k, mu=mu)


def run_lifeline_demo_4():
    df = load_waltons()  # returns a Pandas DataFrame
    #print(df.head())

    T = df['T']  # TTF data
    E = df['E']  # censoring indicator

    TW = TimeWarping()  # empty wrapper clas
    TW_assuming_non_censored = TimeWarping(T.values)  # use TTF data (neglecting censoring)

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)

    #  plots survival curves and CDF
    kmf.plot_survival_function()
    kmf.plot_cumulative_density()
    plt.grid()
    plt.show()

    g_vals = get_g_from_lifelines(survival_model = kmf, TimeWarp = TW, TTF = T)
    t_vals = kmf.survival_function_.index.values

    # plot

    plt.plot(t_vals, g_vals, label='g(t) from KM estimator (censored)')
    t_non_cen, g_non_cen= TW_assuming_non_censored.t_grid, TW_assuming_non_censored.g_vals,
    plt.plot(t_non_cen, g_non_cen, label = 'g(t) from TW (assuming non censored)')
    plt.xlabel('Time')
    plt.ylabel('g(t)')
    plt.legend()
    plt.grid()
    plt.show()

    # show other examples with parametric models
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 7.5))

    kmf = KaplanMeierFitter().fit(T, E, label='KaplanMeierFitter')
    wbf = WeibullFitter().fit(T, E, label='WeibullFitter')
    exf = ExponentialFitter().fit(T, E, label='ExponentialFitter')
    lnf = LogNormalFitter().fit(T, E, label='LogNormalFitter')
    llf = LogLogisticFitter().fit(T, E, label='LogLogisticFitter')
    pwf = PiecewiseExponentialFitter([40, 60]).fit(T, E, label='PiecewiseExponentialFitter')
    ggf = GeneralizedGammaFitter().fit(T, E, label='GeneralizedGammaFitter')
    sf = SplineFitter(np.percentile(T.loc[E.astype(bool)], [0, 50, 100])).fit(T, E, label='SplineFitter')

    wbf.plot_survival_function(ax=axes[0][0])
    exf.plot_survival_function(ax=axes[0][1])
    lnf.plot_survival_function(ax=axes[0][2])
    kmf.plot_survival_function(ax=axes[1][0])
    llf.plot_survival_function(ax=axes[1][1])
    pwf.plot_survival_function(ax=axes[1][2])
    ggf.plot_survival_function(ax=axes[2][0])
    sf.plot_survival_function(ax=axes[2][1])
    plt.show()

    Models = [wbf, exf, lnf, llf, pwf, ggf, sf]
    for model in Models:
        g_vals = get_g_from_lifelines(survival_model=model, TimeWarp = TW, TTF = T)
        t_vals = model.survival_function_.index.values
        plt.plot(t_vals, g_vals, label=f'g(t) from {model._label}')
        plt.xlabel('Time')
        plt.ylabel('g(t)')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    # Press the green button in the gutter to run the script.
    run_lifeline_demo_4()


