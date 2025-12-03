import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physics Parameters (Setup) ---
# Experimental Efficiency
NA = 0.5
T_optics = 0.8
QE = 0.9

# Efficiency Calculations
# Solid angle fraction: (1 - cos(theta)) / 2
eta_geo = (1 - np.sqrt(1 - NA**2)) / 2
eta_total = eta_geo * T_optics * QE

print(f"[System Efficiency]")
print(f" - Geometric Efficiency (NA={NA}): {eta_geo*100:.2f}%")
print(f" - Total Efficiency (w/ Loss):      {eta_total*100:.2f}%")

# --- 2. Atomic Physics Constants (Yb) ---
# Yb 556nm (Intercombination Line, Green)
# Natural Linewidth = 182.4 kHz -> Gamma = 2*pi * 182.4e3
Gamma_556_rad = 2 * np.pi * 182.4e3 
Delta_556_Hz = -160e3    # Detuning for Dual-tone (-160 kHz)
s_556 = 1               # Saturation Parameter (Strong drive)

# Yb 399nm (Dipole Allowed, Blue)
# Natural Linewidth = 29.1 MHz -> Gamma = 2*pi * 29.1e6
# Reference decay rate is roughly 1.83e8 /s
Gamma_399_rad = 183.0e6  
Delta_399_Hz = 0         # Resonant imaging (typically used for fast detection)
s_399 = 50               # Saturation Parameter

# --- 3. Scattering Rate Calculation Function ---
def calculate_scattering_rate(gamma_rad, delta_hz, s):
    """
    Calculate photon scattering rate for a 2-level system.
    R_sc = (Gamma / 2) * [s / (1 + s + 4(Delta/Gamma)^2)]
    """
    # Convert Gamma to Hz for detuning ratio calculation
    gamma_hz = gamma_rad / (2 * np.pi)
    
    # Detuning term: 4 * (Delta / Gamma)^2
    # Note: Delta is in Hz, Gamma is in Hz (units cancel out)
    detuning_factor = 4 * (delta_hz / gamma_hz)**2
    
    # Steady-state excited population rho_ee
    rho_ee = (s / 2) / (1 + s + detuning_factor)
    
    # Scattering rate = Gamma * rho_ee
    return gamma_rad * rho_ee

# Calculate Rates
R_sc_556 = calculate_scattering_rate(Gamma_556_rad, Delta_556_Hz, s_556)
R_sc_399 = calculate_scattering_rate(Gamma_399_rad, Delta_399_Hz, s_399)

print(f"\n[Scattering Rates]")
print(f" - Yb 556nm (Green): {R_sc_556:.2e} photons/s (Delta=-160kHz)")
print(f" - Yb 399nm (Blue):  {R_sc_399:.2e} photons/s (Resonant)")

# --- 4. Simulation & Plotting ---

# Helper function for plotting
def plot_performance(ax_noise, ax_snr, time_array, R_sc, eta, color, label, time_unit_str, time_scale_factor):
    # 1. Signal (Electrons)
    signal = R_sc * time_array * eta
    
    # 2. Shot Noise (RMS) = sqrt(Signal)
    noise = np.sqrt(signal)
    
    # 3. SNR = Signal / Noise = sqrt(Signal)
    snr = np.sqrt(signal)
    
    # Plot Noise
    ax_noise.plot(time_array * time_scale_factor, noise, color=color, linewidth=2, label=f'{label} Noise')
    ax_noise.set_ylabel(r'Shot Noise ($e^-$ rms)', fontsize=12)
    ax_noise.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_noise.legend(loc='upper left')
    
    # Plot SNR
    ax_snr.plot(time_array * time_scale_factor, snr, color=color, linewidth=2, label=f'{label} SNR')
    
    # Add Threshold Line (SNR = 5)
    ax_snr.axhline(y=5, color='red', linestyle='--', linewidth=1.5, label='Limit (SNR=5)')
    
    # Calculate & Mark Time Limit
    # t_limit = SNR^2 / (R * eta)
    t_limit = (5**2) / (R_sc * eta)
    ax_snr.axvline(x=t_limit * time_scale_factor, color='black', linestyle=':', 
                   label=f'Min Time: {t_limit*time_scale_factor:.2f} {time_unit_str}')
    
    ax_snr.set_ylabel('SNR', fontsize=12)
    ax_snr.set_xlabel(f'Exposure Time ({time_unit_str})', fontsize=12)
    ax_snr.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_snr.legend(loc='lower right')
    
    return t_limit

# Plotting Setup
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# --- Figure 1: Yb 556nm (Green) ---
# Time scale: 10 us to 5 ms (Slow)
t_556 = np.logspace(np.log10(10e-6), np.log10(5e-3), 500)

fig1, (ax1_n, ax1_s) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
fig1.suptitle(f'Yb 556nm Imaging Performance\n(Dual-tone $\Delta=-160$ kHz, s={s_556})', fontsize=14)

limit_556 = plot_performance(ax1_n, ax1_s, t_556, R_sc_556, eta_total, 
                             'green', '556nm', 'ms', 1e3)

plt.tight_layout()
plt.savefig('Yb_556nm_Analysis.png', dpi=150)


# --- Figure 2: Yb 399nm (Blue) ---
# Time scale: 0.1 us to 20 us (Ultra-Fast)
t_399 = np.logspace(np.log10(0.1e-6), np.log10(20e-6), 500)

fig2, (ax2_n, ax2_s) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
fig2.suptitle(f'Yb 399nm Imaging Performance\n(Resonant, s={s_399})', fontsize=14)

limit_399 = plot_performance(ax2_n, ax2_s, t_399, R_sc_399, eta_total, 
                             'blue', '399nm', 'us', 1e6)

plt.tight_layout()
plt.savefig('Yb_399nm_Analysis.png', dpi=150)

# Show results
print(f"\n[Calculated Detection Limits (SNR > 5)]")
print(f" - 556nm requires at least: {limit_556*1e3:.3f} ms")
print(f" - 399nm requires at least: {limit_399*1e6:.3f} us")
print("\nFigures saved as 'Yb_556nm_Analysis.png' and 'Yb_399nm_Analysis.png'")
plt.show()