import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from propa.kraken_toolbox.run_kraken import runkraken
from localisation.verlinden.testcases.testcase_envs import testcase2_1
from scipy.optimize import minimize

# run_test = False
# if run_test:
#     testcase = testcase2_1
#     frequencies = np.arange(1, 100, 0.5)
#     max_range_m = 50 * 1e3

#     kraken_env, kraken_flp = testcase(
#         freq=frequencies,
#         max_range_m=max_range_m,
#     )

#     cpu_time = []
#     for freq in frequencies:
#         t0 = time.time()
#         pf, field_pos = runkraken(
#             kraken_env,
#             kraken_flp,
#             [freq],
#             parallel=False,
#             verbose=False,
#         )
#         t1 = time.time()
#         elapsed_time = t1 - t0
#         cpu_time.append(elapsed_time)

#     fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\parallel\frequency_bins\cpu_time.csv"
#     pd.DataFrame({"frequencies": frequencies, "cpu_time": cpu_time}).to_csv(
#         fpath, index=False
#     )

# else:
#     fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\parallel\frequency_bins\cpu_time.csv"
#     pd_data = pd.read_csv(fpath)
#     frequencies = pd_data["frequencies"]
#     cpu_time = pd_data["cpu_time"]

#     filtered_cpu = cpu_time.rolling(window=10).median()
#     nan_idx = filtered_cpu.isna()
#     z, residuals, rank, singular_values, rcond = np.polyfit(
#         frequencies[~nan_idx], filtered_cpu[~nan_idx], 2, full=True
#     )
#     poly_fit_cpu = np.poly1d(z)

#     z_lin, residuals_lin, rank_lin, singular_values_lin, rcond_lin = np.polyfit(
#         frequencies[~nan_idx], filtered_cpu[~nan_idx], 1, full=True
#     )
#     linear_fit_cpu = np.poly1d(z_lin)

#     # Save values
#     fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\parallel\frequency_bins\cpu_time_polyfit.csv"
#     pd.DataFrame(
#         {
#             "frequencies": frequencies,
#             "cpu_time": cpu_time,
#             "filtered_cpu": filtered_cpu,
#             "poly_fit_cpu": poly_fit_cpu(frequencies),
#         }
#     ).to_csv(fpath, index=False)
#     # print(poly_fit_cpu(frequencies))
#     plt.figure()
#     plt.plot(frequencies, cpu_time)
#     plt.plot(frequencies, filtered_cpu, label="filtered")
#     plt.plot(
#         frequencies,
#         poly_fit_cpu(frequencies),
#         label=f"polyfit : {z[0]:.2e}f^2 + {z[1]:.2e}f + {z[2]:.2e}",
#     )
#     plt.plot(
#         frequencies,
#         linear_fit_cpu(frequencies),
#         label=f"linear : {z_lin[0]:.2e}f + {z_lin[1]:.2e}",
#     )
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("CPU time (s)")
#     plt.grid()
#     plt.legend()

#     img_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\kraken_perf\cpu_time_vs_frequency.png"
#     plt.savefig(img_path)


def fit_cpu_time(fpath, win=10):
    pd_data = pd.read_csv(fpath)
    frequencies = pd_data["frequencies"]
    cpu_time = pd_data["cpu_time"]

    filtered_cpu = cpu_time.rolling(window=win).median()
    nan_idx = filtered_cpu.isna()
    z, residuals, rank, singular_values, rcond = np.polyfit(
        frequencies[~nan_idx], filtered_cpu[~nan_idx], 2, full=True
    )
    poly_fit_cpu = np.poly1d(z)

    plt.figure()
    plt.plot(frequencies, cpu_time)
    plt.plot(frequencies, filtered_cpu, label="filtered")
    plt.plot(
        frequencies,
        poly_fit_cpu(frequencies),
        label=f"polyfit : {z[0]:.2e}f^2 + {z[1]:.2e}f + {z[2]:.2e}",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("CPU time (s)")
    plt.grid()
    plt.legend()

    img_path = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\img\illustration\kraken_perf\cpu_time_vs_frequency.png"
    plt.savefig(img_path)

    return (frequencies, filtered_cpu, poly_fit_cpu, z)


def find_optimal_intervals(fmin, fmax, nf, z, n_workers, mean_cpu_time):

    # Initial guess
    fi = np.linspace(fmin, fmax, n_workers + 1)
    fi = fi[1:-1]
    alpha = mean_cpu_time * nf
    x0 = np.array([alpha, *fi])

    # Bounds
    bounds = [(0, alpha), *[(fmin, fmax)] * len(fi)]
    res = minimize(objective_function, x0, args=(z, fmin, fmax), bounds=bounds)
    return res


# Define objective function
def objective_function(x, z, fmin, fmax):
    Y = build_y(x, z=z, fmin=fmin, fmax=fmax)
    return np.sum(Y**2)


def build_y(x, z, fmin, fmax):
    alpha = x[0]
    fi = x[1:]
    Y = np.array([g(fi, alpha, k, z, fmin, fmax) for k in range(len(fi))])
    return Y


def g(fi, alpha, k, z, fmin, fmax):
    x = [fmin, *fi, fmax]
    a, b, c = z
    gk = (
        a / 3 * (x[k + 1] ** 3 - x[k] ** 3)
        + b / 2 * (x[k + 1] ** 2 - x[k] ** 2)
        + c * (x[k + 1] - x[k])
        - alpha
    )
    return gk


if __name__ == "__main__":
    fmin = 1
    fmax = 100
    nf = 100
    n_workers = 4

    fpath = r"C:\Users\baptiste.menetrier\Desktop\devPy\phd\propa\kraken_toolbox\tests\parallel\frequency_bins\cpu_time.csv"
    frequencies, filtered_cpu, poly_fit_cpu, z = fit_cpu_time(fpath, win=50)
    print(z)
    res = find_optimal_intervals(fmin, fmax, nf, z, n_workers, np.mean(filtered_cpu))

    print(res)
