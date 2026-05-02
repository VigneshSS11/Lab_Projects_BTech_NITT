"""
snr_analyzer.py
================
Analyses the quality of the received SSTV audio signal and produces:

  1. Wideband SNR  – ratio of in-band signal power to out-of-band noise.
  2. Sync-tone SNR – precision estimate from 1200 Hz sync pulses.
  3. Per-line SNR  – how noise evolves across the 256 scan lines.
  4. Spectrogram   – time-frequency heatmap saved to a PNG (first 20 s).
  5. Summary report – plain-text metrics table.

Author : SSTV-Pluto Bridge Project
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal as sps
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

SSTV_LOW   = 1100
SSTV_HIGH  = 2500
SYNC_FREQ  = 1200
T_SYNC     = 0.004862
T_LINE     = 0.146432
SPEC_MAX_S = 20        # seconds of audio to use for spectrogram (memory limit)


def _load_wav(path: str) -> tuple:
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)
    if np.max(np.abs(data)) > 1.0:
        data /= 32768.0
    return rate, data


def wideband_snr(samples: np.ndarray, rate: int) -> float:
    N = len(samples)
    freqs = fftfreq(N, 1 / rate)
    spectrum = np.abs(fft(samples)) ** 2 / N
    sig_mask   = (np.abs(freqs) >= SSTV_LOW)  & (np.abs(freqs) <= SSTV_HIGH)
    noise_mask = ((np.abs(freqs) >= 200) & (np.abs(freqs) < SSTV_LOW)) | \
                 ((np.abs(freqs) > SSTV_HIGH) & (np.abs(freqs) <= 5000))
    p_sig   = np.mean(spectrum[sig_mask])   if sig_mask.any()   else 1e-12
    p_noise = np.mean(spectrum[noise_mask]) if noise_mask.any() else 1e-12
    return 10 * np.log10(p_sig / (p_noise + 1e-12))


def sync_snr(samples: np.ndarray, rate: int) -> float:
    block = int(T_SYNC * rate)
    n_blocks = len(samples) // block
    if n_blocks == 0:
        return float("nan")
    omega = 2 * np.pi * SYNC_FREQ * block / rate / block
    cos_w = np.cos(2 * np.pi * SYNC_FREQ / rate * block / block *
                   (block / rate) * rate / block)
    # Simple Goertzel via correlation
    cos_w = np.cos(2 * np.pi * SYNC_FREQ / rate)
    magnitudes = []
    for i in range(min(n_blocks, 200)):   # cap at 200 blocks
        seg = samples[i * block: (i + 1) * block]
        # Narrow-band energy at sync freq
        t = np.arange(len(seg)) / rate
        energy = np.abs(np.dot(seg, np.exp(2j * np.pi * SYNC_FREQ * t))) / len(seg)
        magnitudes.append(energy)
    mags = np.array(magnitudes)
    thresh = np.percentile(mags, 70)
    sync_blocks  = mags[mags >  thresh]
    noise_blocks = mags[mags <= thresh]
    if len(noise_blocks) == 0:
        return float("nan")
    return round(20 * np.log10(np.mean(sync_blocks) /
                                (np.mean(noise_blocks) + 1e-12)), 1)


def per_line_snr(samples: np.ndarray, rate: int) -> np.ndarray:
    line_samples = int(T_LINE * rate)
    snrs = []
    for i in range(256):
        start = i * line_samples
        end   = start + line_samples
        if end > len(samples):
            snrs.append(float("nan"))
            continue
        snrs.append(wideband_snr(samples[start:end], rate))
    return np.array(snrs)


def plot_spectrogram(samples: np.ndarray, rate: int,
                     output_path: str = "spectrogram.png",
                     title: str = "SSTV Received Signal Spectrogram"):
    # Limit to SPEC_MAX_S to avoid memory issues
    seg = samples[:int(SPEC_MAX_S * rate)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1 = axes[0]
    f, t, Sxx = sps.spectrogram(seg, rate, nperseg=512, noverlap=480,
                                 scaling="density")
    band_mask = (f >= 900) & (f <= 2800)
    Sxx_db = 10 * np.log10(Sxx[band_mask] + 1e-12)
    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 99)
    im = ax1.pcolormesh(t, f[band_mask], Sxx_db, shading="gouraud",
                        cmap="inferno", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax1, label="Power (dB)")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"Spectrogram (first {SPEC_MAX_S} s)")
    for freq, label, col in [
        (1200, "Sync 1200 Hz", "cyan"),
        (1500, "Black 1500 Hz", "lime"),
        (1900, "Leader 1900 Hz", "yellow"),
        (2300, "White 2300 Hz", "red"),
    ]:
        ax1.axhline(freq, color=col, linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.text(t[-1] * 0.99, freq + 20, label,
                 color=col, fontsize=7, ha="right", va="bottom")

    ax2 = axes[1]
    ds = max(1, len(seg) // 8000)
    t_wave = np.arange(len(seg))[::ds] / rate
    ax2.plot(t_wave, seg[::ds], lw=0.4, color="steelblue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Waveform")
    ax2.set_xlim(0, SPEC_MAX_S)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SNR] Spectrogram saved → {output_path}")
    return output_path


def analyze(wav_path: str, output_dir: str = ".") -> dict:
    print(f"[SNR] Analysing: {wav_path}")
    rate, samples = _load_wav(wav_path)
    duration = len(samples) / rate
    print(f"  Duration      : {duration:.1f} s")
    print(f"  Sample rate   : {rate} Hz")
    print(f"  RMS amplitude : {np.sqrt(np.mean(samples**2)):.4f}")

    # Run metrics on a 30-second window to keep memory manageable
    analysis_seg = samples[:int(min(30, duration) * rate)]

    wb_snr  = wideband_snr(analysis_seg, rate)
    sy_snr  = sync_snr(analysis_seg, rate)
    ln_snrs = per_line_snr(samples, rate)

    mean_ln = float(np.nanmean(ln_snrs))
    min_ln  = float(np.nanmin(ln_snrs))
    max_ln  = float(np.nanmax(ln_snrs))

    spec_path = os.path.join(output_dir, "sstv_spectrogram.png")
    plot_spectrogram(samples[:int(SPEC_MAX_S * rate)], rate, spec_path)

    report = {
        "wideband_snr_dB"  : round(wb_snr, 1),
        "sync_tone_snr_dB" : round(sy_snr, 1),
        "mean_line_snr_dB" : round(mean_ln, 1),
        "min_line_snr_dB"  : round(min_ln, 1),
        "max_line_snr_dB"  : round(max_ln, 1),
        "duration_s"       : round(duration, 1),
        "sample_rate_hz"   : rate,
        "spectrogram_path" : spec_path,
    }

    print("\n" + "═" * 50)
    print("  SSTV SIGNAL QUALITY REPORT")
    print("═" * 50)
    for k, v in report.items():
        if k == "spectrogram_path":
            continue
        print(f"  {k.replace('_',' ').title():<28}: {v}")
    print("═" * 50 + "\n")
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python snr_analyzer.py <received.wav> [output_dir]")
        sys.exit(1)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    analyze(sys.argv[1], out_dir)
