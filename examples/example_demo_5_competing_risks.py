import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, AalenJohansenFitter

from rul_timewarping.timewarping import TimeWarping
from rul_timewarping.utils import get_g_from_lifelines
from rul_timewarping.utils import compute_mrl, compute_g_non_parametric



def generate_competing_risks_dataset(n=5_000,
                                     seed=42,
                                     censoring_rate=0.05):
    """
    Generates n subjects with two competing risks, then censors exactly
    floor(2*n*censoring_rate) of them at random times < their true failure.
    """
    np.random.seed(seed)

    # 1) true failure times and causes
    T1 = np.random.gamma(shape=2, scale=3, size=n)    # ageing
    T2 = np.random.exponential(scale=10, size=n)      # sudden
    T_true = np.minimum(T1, T2)
    cause  = np.where(T1 < T2, 1, 2)

    # 2) decide which indices will be censored
    n_censor = int(2 * censoring_rate * n)
    censor_idx = np.random.choice(n, size=n_censor, replace=False)

    # 3) build observed times & events
    T_obs = T_true.copy()
    E_obs = cause.copy()

    # for censored subjects: T_obs ~ Uniform(0, T_true), and E_obs = 0
    u = np.random.rand(n_censor)
    T_obs[censor_idx] = u * T_true[censor_idx]
    E_obs[censor_idx] = 0

    return pd.DataFrame({'times': T_obs, 'events': E_obs})


def run_competing_risks_5():
    # 1) Generate data and isolate cause‑1 (ageing)
    df = generate_competing_risks_dataset(n=500)

    # 2) Fit Kaplan and Aalen-Johansen model on cause‑1 events and casue-2 events
    aj1 = AalenJohansenFitter()
    aj1.fit(durations=df['times'], event_observed=df['events'], event_of_interest=1)

    aj2 = AalenJohansenFitter()
    aj2.fit(durations=df['times'], event_observed=df['events'], event_of_interest=2)

    df_c1 = df.copy()
    df_c1['events'] = np.where(df_c1['events'] == 1, 1, 0)
    kmf1 = KaplanMeierFitter()
    kmf1.fit(durations=df_c1['times'], event_observed=df_c1['events'])

    df_c2 = df.copy()
    df_c2['events'] = np.where(df_c2['events'] == 2, 1, 0)
    kmf2 = KaplanMeierFitter()
    kmf2.fit(durations=df_c2['times'], event_observed=df_c2['events'])

    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['times'], event_observed=df['events'])

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.step(aj1.cumulative_density_.index, aj1.cumulative_density_, where='post', label='AJF Cause 1')
    plt.step(aj2.cumulative_density_.index, aj2.cumulative_density_, where='post', label='AJF Cause 2')
    plt.step(kmf1.cumulative_density_.index, kmf1.cumulative_density_, where='post', label='KMF Cause 1')
    plt.step(kmf2.cumulative_density_.index, kmf2.cumulative_density_, where='post', label='KMF Cause 2')
    plt.step(kmf.cumulative_density_.index, kmf.cumulative_density_, where='post', label='KMF Cause 1,2')

    plt.title("Cumulative Failure / Incidence Comparison")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Build a fine time grid over [0, max T]
    t_max = df_c1['times'].max()
    t_grid = kmf1.survival_function_.index
    # Evaluate survival S(t) on that grid
    surv_vals = kmf1.survival_function_at_times(t_grid).values

    # Compute original MRL(t)
    mrl_t = compute_mrl(t_grid, surv_vals)
    plt.plot(t_grid, mrl_t)
    plt.xlabel('time')
    plt.show()

    # 3) Compute time‐warp g(t)
    TW = TimeWarping(ttf_data=df_c1.loc[df_c1['events']==1, 'times'].values)
    g_vals_kmf1 = get_g_from_lifelines(
        survival_model=kmf1,
        TimeWarp=TW,
        TTF=t_grid  # note: pass our grid for consistency
    )

    # Compute “warped” MRL: still use original mrl but plot versus g(t)
    mrl_g = mrl_t.copy()

    # === Plotting ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_grid, mrl_t, lw=2)
    plt.title("Original MRL(t)")
    plt.xlabel("t")
    plt.ylabel("MRL(t)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(kmf1.survival_function_.index, g_vals_kmf1, s=20)
    plt.show()
    # Overlay a linear fit to demonstrate linearity:
    coeffs = np.polyfit(g_vals_kmf1, mrl_g, 1)
    plt.plot(g_vals, np.polyval(coeffs, g_vals), 'r--',
             label=f"Linear fit: y={coeffs[0]:.3f}·g+{coeffs[1]:.3f}")
    plt.title("MRL vs. warped time g(t)")
    plt.xlabel("g(t)")
    plt.ylabel("MRL")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Press the green button in the gutter to run the script.
    run_competing_risks_5()