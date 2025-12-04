import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physics Parameters (Setup) ---
# Experimental Efficiency
NA = 0.5
T_optics = 0.8
QE = 0.9

# Efficiency Calculations
eta_geo = (1 - np.sqrt(1 - NA**2)) / 2
eta_total = eta_geo * T_optics * QE

print(f"[System Efficiency]")
print(f" - Geometric Efficiency (NA={NA}): {eta_geo*100:.2f}%")
print(f" - Total Efficiency (w/ Loss):{eta_total*100:.2f}%")

# --- 2. Atomic Physics Constants (Yb) ---
# Yb 556nm (Intercombination Line, Green)
Gamma_556_rad = 2 * np.pi * 182.4e3 
Delta_556_Hz = -160e3 
s_556 = 1.7 # Yb 399nm (Dipole Allowed, Blue)
Gamma_399_rad = 183.0e6
Delta_399_Hz = 0
s_399 = 10 

# --- 3. Scattering Rate Calculation Function ---
def calculate_scattering_rate(gamma_rad, delta_hz, s):
    """
    Calculate photon scattering rate for a 2-level system.
    R_sc = Gamma * rho_ee
    """
    gamma_hz = gamma_rad / (2 * np.pi)
    detuning_factor = 4 * (delta_hz / gamma_hz)**2
    # rho_ee = (s / 2) / (1 + s + detuning_factor)
    rho_ee_factor = (s / 2) / (1 + s + detuning_factor)
    return gamma_rad * rho_ee_factor

# Calculate Rates
R_sc_556 = calculate_scattering_rate(Gamma_556_rad, Delta_556_Hz, s_556)
R_sc_399 = calculate_scattering_rate(Gamma_399_rad, Delta_399_Hz, s_399)

print(f"\n[Scattering Rates]")
print(f" - Yb 556nm (Green): {R_sc_556:.2e} photons/s (Delta=-160kHz)")
print(f" - Yb 399nm (Blue):  {R_sc_399:.2e} photons/s (Resonant)")

# --- 4. Simulation & Plotting (Modified) ---

# New target SNR
SNR_TARGET = 3.89

# Helper function for plotting (Updated for Noise Variance and SNR_TARGET)
def plot_performance(ax_variance, ax_snr, time_array, R_sc, eta, color, label, time_unit_str, time_scale_factor, snr_target=SNR_TARGET):
    """
    Plots Noise Variance (N = Signal) and SNR vs. Exposure Time.
    """
    # 1. Signal (Electrons) / Noise Variance (N)
    # Shot Noise Limited System에서 N = Signal
    noise_variance = R_sc * time_array * eta
    
    # 2. SNR = sqrt(Signal)
    snr = np.sqrt(noise_variance)
    
    # Plot Noise Variance (N)
    ax_variance.plot(time_array * time_scale_factor, noise_variance, color=color, linewidth=2, label=f'{label} Noise Variance ($N$)')
    ax_variance.set_ylabel(r'Noise Variance ($N = e^-$)', fontsize=12) # <- Y축 레이블 변경
    ax_variance.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_variance.legend(loc='upper left')
    
    # Plot SNR
    ax_snr.plot(time_array * time_scale_factor, snr, color=color, linewidth=2, label=f'{label} SNR')
    
    # Add Threshold Line (SNR = snr_target)
    ax_snr.axhline(y=snr_target, color='red', linestyle='--', linewidth=1.5, label=f'Limit (SNR={snr_target})')
    
    # Calculate & Mark Time Limit: t_limit = SNR_target^2 / (R * eta)
    t_limit = (snr_target**2) / (R_sc * eta) # <- 분자가 4로 변경
    ax_snr.axvline(x=t_limit * time_scale_factor, color='black', linestyle=':', 
                    label=f'Min Time (SNR={snr_target}): {t_limit*time_scale_factor:.2f} {time_unit_str}')
    
    ax_snr.set_ylabel('SNR', fontsize=12)
    ax_snr.set_xlabel(f'Exposure Time ({time_unit_str})', fontsize=12)
    ax_snr.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_snr.legend(loc='lower right')
    
    return t_limit

# Plotting Setup
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# --- Figure 1: Yb 556nm (Green) ---
t_556 = np.logspace(np.log10(10e-6), np.log10(20e-3), 500)

# ax1_n 대신 ax1_v (Noise Variance) 사용
fig1, (ax1_v, ax1_s) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
fig1.suptitle(f'Yb 556nm Imaging Performance (SNR={SNR_TARGET} Limit)\n(Dual-tone $\Delta=-160$ kHz, s={s_556}, T_optics={T_optics})', fontsize=14)

limit_556 = plot_performance(ax1_v, ax1_s, t_556, R_sc_556, eta_total, 
                             'green', '556nm', 'ms', 1e3)

plt.tight_layout()
plt.savefig('Yb_556nm_Analysis_SNR2.png', dpi=150)


# --- Figure 2: Yb 399nm (Blue) ---
t_399 = np.logspace(np.log10(0.1e-6), np.log10(20e-6), 500)

# ax2_n 대신 ax2_v (Noise Variance) 사용
fig2, (ax2_v, ax2_s) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
fig2.suptitle(f'Yb 399nm Imaging Performance (SNR={SNR_TARGET} Limit)\n(Resonant, s={s_399}, T_optics={T_optics})', fontsize=14)

limit_399 = plot_performance(ax2_v, ax2_s, t_399, R_sc_399, eta_total, 
                             'blue', '399nm', 'us', 1e6)

plt.tight_layout()
plt.savefig('Yb_399nm_Analysis_SNR2.png', dpi=150)


# Show results
print(f"\n[Calculated Detection Limits (SNR > {SNR_TARGET})]")
print(f" - 556nm requires at least: {limit_556*1e3:.3f} ms")
print(f" - 399nm requires at least: {limit_399*1e6:.3f} us")
print("\nFigures saved as 'Yb_556nm_Analysis_SNR2.png' and 'Yb_399nm_Analysis_SNR2.png'")
plt.show()