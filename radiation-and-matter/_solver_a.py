import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

C_H = 2.28527331e11
C_H_MINUS = 1.62073152e21

def residuals(densities, N):
    N_H, N_H_plus, N_H_minus, N_e = densities

    return np.array([
        (N_H + N_H_plus + N_H_minus + N_e - N) / N,
        (N_H_plus - N_H_minus - N_e) / N,
        (N_H_plus * N_e / N_H - C_H) / C_H,
        (N_H * N_e / N_H_minus - C_H_MINUS) / C_H_MINUS
    ])

# values from 10^8 to 10^18 cm^-3
N_values = np.logspace(8, 18, 11)

# each row will contain:
# N_H, N_H_plus, N_H_minus, N_e
solutions = np.empty((len(N_values), 4))

# initial guess for the first value of N 
# (in [N_H, N_H+, N_H-, N_e] format)
guess = np.array([
    0.25 * N_values[0],
    0.25 * N_values[0],
    1e-10 * N_values[0],
    0.25 * N_values[0]
])

for i, N in enumerate(N_values):
    solution = least_squares(
        residuals,
        guess,
        args=(N,),
        bounds=(np.finfo(float).tiny, np.inf),
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=5000
    )

    solutions[i] = solution.x

    # use the current solution for the next initial guess
    if i + 1 < len(N_values):
        guess = solution.x * N_values[i + 1] / N


# print the results as a table
print(
    f"{'N':>15}"
    f"{'N_H':>15}"
    f"{'N_H_plus':>15}"
    f"{'N_H_minus':>15}"
    f"{'N_e':>15}"
)

print("-" * 75) # omg I discovered you can do this after 15 years of using Python lol!

for N, densities in zip(N_values, solutions):
    N_H, N_H_plus, N_H_minus, N_e = densities

    print(
        f"{N:15.2e}"
        f"{N_H:15.2e}"
        f"{N_H_plus:15.2e}"
        f"{N_H_minus:15.2e}"
        f"{N_e:15.2e}"
    )

# plot each species density against total particle density

plt.figure(figsize=(8, 5.5))

plt.loglog(
    N_values,
    solutions[:, 0],
    linestyle="-",
    linewidth=2,
    label=r"$N_{\mathrm{H}}$"
)

plt.loglog(
    N_values,
    solutions[:, 1],
    linestyle="-",
    linewidth=2,
    label=r"$N_{\mathrm{H}^{+}}$"
)

plt.loglog(
    N_values,
    solutions[:, 2],
    linestyle="-",
    linewidth=2,
    label=r"$N_{\mathrm{H}^{-}}$"
)

plt.loglog(
    N_values,
    solutions[:, 3],
    linestyle="--",
    linewidth=2,
    label=r"$N_{\mathrm{e}}$"
)

plt.xlabel(r"Total particle density $N\;(\mathrm{cm}^{-3})$")
plt.ylabel(r"Species number density $(\mathrm{cm}^{-3})$")

plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig(
    "hydrogen_population_densities.svg",
    format="svg",
    bbox_inches="tight"
)

plt.close()

print("ALL GOOD! :)")
