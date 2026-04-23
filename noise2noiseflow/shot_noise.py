import numpy as np
import matplotlib.pyplot as plt

# --- 1. Shared Physics & Plotting Functions ---

def get_efficiency(na, t_optics, qe):
    """Calculates geometric and total detection efficiency."""
    eta_geo = (1 - np.sqrt(1 - na**2)) / 2
    eta_ego = 0.08
    return eta_geo, eta_geo * t_optics * qe

def calculate_scattering_rate(gamma_hz, delta_hz, s):
    """
    Calculate photon scattering rate (photons/s).
    Gamma, Delta in Hz. Prefactor uses angular decay rate Gamma_rad = 2*pi*gamma_hz.
        R_sc = pi * gamma_hz * s / (1 + s + (2*Delta/Gamma)^2)
    """
    detuning_factor = (2 * delta_hz / gamma_hz)**2
    return np.pi * gamma_hz * s / (1 + s + detuning_factor)

def plot_performance(ax_variance, ax_snr, config, R_sc, eta, snr_target=3.29):
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
# NOTE: All Gamma and Delta values are in Hz (not rad/s)

# Configuration for Yb 556nm (Green)
cfg_556 = {
    'name': 'Yb 556nm',
    'color': 'green',
    'unit': 'ms',
    'scale_factor': 1e3,
    'time_range': np.logspace(np.log10(10e-6), np.log10(20e-3), 500),
    # System Parameters (Independent)
    'NA': 0.5,
    'T_optics': 0.09,
    'QE': 0.9, # sCMOS usually high in green
    # Atomic Physics (all Hz)
    'Gamma': 182.4e3,   # 182.4 kHz linewidth
    'Delta': -160e3,     # -160 kHz detuning
    's': 2.0
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
    'T_optics': 0.4,
    'QE': 0.9,      # WARNING: sCMOS QE often drops at 400nm (check datasheet)
    # Atomic Physics (all Hz)
    'Gamma': 29.1e6,    # 29.1 MHz linewidth
    'Delta': 0,
    's': 10
}

# Configuration 3: Saffman's Cs 852nm (The "Slow" Reference)
# Source: arXiv:2311.12217
cfg_cs = {
    'name': 'Cs 852nm (Saffman)',
    'color': 'red',
    'unit': 'ms',
    'scale_factor': 1e3,
    'time_range': np.logspace(np.log10(10e-6), np.log10(20e-3), 500),
    # System Parameters
    'NA': 0.7,      # High NA lens used in the paper
    'T_optics': 0.4,# Estimated
    'QE': 0.9,     # EMCCD at 852nm
    # Atomic Physics (all Hz)
    'Gamma': 5.2e6,              # 5.2 MHz (D2 line)
    'Delta': 9 * 5.2e6,          # 9 * Gamma (HUGE Detuning for cooling)
    's': 2.0                     # Assumed saturation for Molasses
}

thompson_yb = {
    'name': 'yb 556nm (Thompson)',
    'color': 'green',
    'unit': 'ms',
    'scale_factor': 1e3,
    'time_range': np.logspace(np.log10(10e-6), np.log10(20e-3), 500),
    # System Parameters
    'NA': 0.55,      # High NA lens used in the paper
    'T_optics': 0.9,# Estimated
    'QE': 0.9,
    # Atomic Physics (all Hz)
    'Gamma': 182.4e3,
    'Delta': -2.1 * 182.4e3,
    's': 2 * 4.5 # dual tone?
}

# --- 3. Execution & Visualization ---
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    SNR_TARGET = 2.63

    # configs = [cfg_556, cfg_399, cfg_cs]
    configs = [cfg_556]

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
        fig.suptitle(f"{cfg['name']} Imaging Performance (SNR={SNR_TARGET})\n(NA={cfg['NA']}, T={cfg['T_optics']}, QE={cfg['QE']})", fontsize=14)

        t_min = plot_performance(ax_v, ax_s, cfg, R_sc, eta_tot, SNR_TARGET)

        print(f" -> Required Exposure: {t_min * cfg['scale_factor']:.3f} {cfg['unit']}")

        plt.tight_layout()
        # plt.savefig(f"{cfg['name'].replace(' ', '_')}_Analysis.png", dpi=150)

    plt.show()
