import numpy as np
import matplotlib.pyplot as plt

# --- 1. Shared Physics & Plotting Functions ---

def get_efficiency(na, t_optics, qe):
    """Calculates geometric and total detection efficiency."""
    eta_geo = (1 - np.sqrt(1 - na**2)) / 2
    return eta_geo, eta_geo * t_optics * qe

def calculate_scattering_rate(gamma_rad, delta_hz, s):
    """
    Calculate photon scattering rate (R_sc = Gamma * rho_ee).
    """
    gamma_hz = gamma_rad / (2 * np.pi)
    detuning_factor = 4 * (delta_hz / gamma_hz)**2
    rho_ee_factor = (s / 2) / (1 + s + detuning_factor)
    return gamma_rad * rho_ee_factor

def plot_performance(ax_variance, ax_snr, config, R_sc, eta, snr_target=3.89):
    """
    Plots Noise Variance and SNR based on specific configuration.
    """
    time_array = config['time_range']
    scale = config['scale_factor']
    label = config['name']
    
    # Physics: Signal (N) & SNR
    noise_variance = R_sc * time_array * eta
    snr = np.sqrt(noise_variance)
    
    # 1. Noise Variance Plot
    ax_variance.plot(time_array * scale, noise_variance, color=config['color'], 
                     linewidth=2, label=f'{label} Signal ($N$)')
    ax_variance.set_ylabel(r'Signal / Variance ($N = e^-$)', fontsize=12)
    ax_variance.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_variance.legend(loc='upper left')
    
    # 2. SNR Plot
    ax_snr.plot(time_array * scale, snr, color=config['color'], 
                linewidth=2, label=f'{label} SNR')
    
    # Threshold Line
    ax_snr.axhline(y=snr_target, color='red', linestyle='--', linewidth=1.5, label=f'Limit ({snr_target})')
    
    # Calculate Time Limit
    t_limit = (snr_target**2) / (R_sc * eta)
    ax_snr.axvline(x=t_limit * scale, color='black', linestyle=':', 
                   label=f'Min Time: {t_limit*scale:.2f} {config["unit"]}')
    
    ax_snr.set_ylabel('SNR', fontsize=12)
    ax_snr.set_xlabel(f'Exposure Time ({config["unit"]})', fontsize=12)
    ax_snr.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_snr.legend(loc='lower right')
    
    return t_limit

# --- 2. Configuration & Independent Parameters ---

# Configuration for Yb 556nm (Green)
cfg_556 = {
    'name': 'Yb 556nm',
    'color': 'green',
    'unit': 'ms',
    'scale_factor': 1e3,
    'time_range': np.logspace(np.log10(10e-6), np.log10(20e-3), 500),
    # System Parameters (Independent)
    'NA': 0.6,
    'T_optics': 0.8,
    'QE': 0.9, # sCMOS usually high in green
    # Atomic Physics
    'Gamma': 2 * np.pi * 182.4e3,
    'Delta': -160e3,
    's': 1.7
}

# Configuration for Yb 399nm (Blue)
cfg_399 = {
    'name': 'Yb 399nm',
    'color': 'blue',
    'unit': 'us',
    'scale_factor': 1e6,
    'time_range': np.logspace(np.log10(0.1e-6), np.log10(20e-6), 500),
    # System Parameters (Independent)
    'NA': 0.6,      # Lens might have same NA, but verify coating
    'T_optics': 0.8,
    'QE': 0.9,      # WARNING: sCMOS QE often drops at 400nm (check datasheet)
    # Atomic Physics
    'Gamma': 183.0e6, # approx 29 MHz * 2pi
    'Delta': 0,
    's': 10
}

# --- 3. Execution & Visualization ---
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
SNR_TARGET = 3.89

configs = [cfg_556, cfg_399]

for cfg in configs:
    # 1. Calculate Efficiency
    eta_geo, eta_tot = get_efficiency(cfg['NA'], cfg['T_optics'], cfg['QE'])
    
    # 2. Calculate Scattering Rate
    R_sc = calculate_scattering_rate(cfg['Gamma'], cfg['Delta'], cfg['s'])
    
    print(f"\n[{cfg['name']} Analysis]")
    print(f" - Efficiency: {eta_tot*100:.2f}% (NA={cfg['NA']}, T={cfg['T_optics']}, QE={cfg['QE']})")
    print(f" - Scattering Rate: {R_sc:.2e} photons/s")

    # 3. Plot
    fig, (ax_v, ax_s) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(f"{cfg['name']} Imaging Performance (SNR={SNR_TARGET})", fontsize=14)
    
    t_min = plot_performance(ax_v, ax_s, cfg, R_sc, eta_tot, SNR_TARGET)
    
    print(f" -> Required Exposure: {t_min * cfg['scale_factor']:.3f} {cfg['unit']}")
    
    plt.tight_layout()
    # plt.savefig(f"{cfg['name'].replace(' ', '_')}_Analysis.png", dpi=150)

plt.show()