import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 5e8

C_H = 2.28527331e11
C_Hminus = 1.62073152e21

# forward reaction coefficients dont change with N
alpha_A = 3.39e-13
alpha_B = 2.82e-15
alpha_C = 8.28e-9

# reverse coefficients
zeta_A = alpha_A * C_H             # s^-1
zeta_B = alpha_B * C_Hminus        # s^-1
beta_C = alpha_C * C_H / C_Hminus  # cm^3 s^-1

initial_populations = [N / 5, 2 * N / 5, N / 5, (2 * N / 5) - N / 5]

def derivatives(t, populations):
    N_H, N_Hplus, N_Hminus, N_e = populations

    # forward reaction rates
    r_A = alpha_A * N_Hplus * N_e
    r_B = alpha_B * N_H * N_e
    r_C = alpha_C * N_Hplus * N_Hminus

    # reverse reaction rates
    r_A_rev = zeta_A * N_H
    r_B_rev = zeta_B * N_Hminus
    r_C_rev = beta_C * N_H**2

    # rates of change of the four populations
    dN_H_dt = r_A - r_A_rev - r_B + r_B_rev + 2*r_C - 2*r_C_rev
    dN_Hplus_dt = -r_A + r_A_rev - r_C + r_C_rev
    dN_Hminus_dt = r_B - r_B_rev - r_C + r_C_rev
    dN_e_dt = -r_A + r_A_rev - r_B + r_B_rev

    return [dN_H_dt, dN_Hplus_dt, dN_Hminus_dt, dN_e_dt]


# get 1000 points between 10^-12 s and 1000 s
times = np.geomspace(1e-12, 1000, 1000)

solution = solve_ivp(
    derivatives,
    (0, 1000), # time
    initial_populations, # N(0)
    method="Radau",
    t_eval=times, # when to return result; solve_ivp selects its own time steps
    rtol=1e-6, # relative error tolerance
    atol=1e-12, # absolute error tolerance (1 cm^-3)
)

# each row contains one species' density at all the stored times
N_H, N_Hplus, N_Hminus, N_e = solution.y

print("Final populations:")
print(f"N_H = {N_H[-1]:.2e} cm^-3")
print(f"N_H+ = {N_Hplus[-1]:.2e} cm^-3")
print(f"N_H- = {N_Hminus[-1]:.2e} cm^-3")
print(f"N_e = {N_e[-1]:.2e} cm^-3")

plt.figure(figsize=(8, 5))
plt.loglog(solution.t, N_H, label="H")
plt.loglog(solution.t, N_Hplus, label="H+")
plt.loglog(solution.t, N_Hminus, label="H-")
plt.loglog(solution.t, N_e, "--", label="Electrons")
plt.xlabel("Time (s)")
plt.ylabel(r"Number density (cm$^{-3}$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("part_b_populations.svg")
plt.show()


print("ALL GOOD! :)")

