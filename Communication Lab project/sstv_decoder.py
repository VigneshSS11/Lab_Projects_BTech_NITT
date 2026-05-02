"""
sstv_decoder.py
================
Reconstructs a colour image from a received SSTV audio stream produced by
the Martin M1 protocol. The decoder works in three stages:

  Stage 1 – Synchronisation
      Detects 1200 Hz horizontal sync pulses using a Goertzel filter and
      aligns the decoder to each scan line.

  Stage 2 – Demodulation
      Instantaneous frequency is extracted from the analytic (Hilbert)
      representation of the bandpass-filtered signal, then mapped back
      from [1500–2300] Hz to pixel intensities [0–255].

  Stage 3 – Reconstruction
      Each of the 256 × 320 RGB pixels is assembled from the G, B, R
      channel sub-lines and written to a PNG/JPEG output file.

Author : SSTV-Pluto Bridge Project
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile
from PIL import Image
import os

# ─── Martin M1 Constants ────────────────────────────────────────────────────
SAMPLE_RATE_EXPECTED = 44100
FREQ_SYNC   = 1200
FREQ_BLACK  = 1500
FREQ_WHITE  = 2300

T_SYNC       = 0.004862   # s – horizontal sync duration
T_SYNC_PORCH = 0.000572
T_CHAN_PORCH = 0.000572
T_PIXEL      = 0.0004576

PIXELS_PER_LINE = 320
NUM_LINES       = 256
NUM_CHANNELS    = 3       # G, B, R


def _to_mono_float(samples, rate) -> tuple:
    """Ensure audio is mono float32 in [-1, 1]."""
    if samples.ndim > 1:
        samples = samples[:, 0]
    samples = samples.astype(np.float32)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    return samples, rate


def _bandpass(samples: np.ndarray, rate: int,
              low: float = 1200, high: float = 2500) -> np.ndarray:
    """4th-order Butterworth bandpass to isolate SSTV tones."""
    sos = scipy_signal.butter(4,
                              [low / (rate / 2), high / (rate / 2)],
                              btype='band', output='sos')
    return scipy_signal.sosfiltfilt(sos, samples)


def _instantaneous_frequency(samples: np.ndarray, rate: int) -> np.ndarray:
    """
    Extract instantaneous frequency using the analytic signal (Hilbert
    transform), then finite-difference unwrapped phase.
    """
    analytic = scipy_signal.hilbert(samples)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * rate / (2 * np.pi)
    return np.append(inst_freq, inst_freq[-1])   # keep same length


def _freq_to_pixel(freq: np.ndarray) -> np.ndarray:
    """Map instantaneous frequency array to [0-255] pixel values."""
    pixel = (freq - FREQ_BLACK) / (FREQ_WHITE - FREQ_BLACK) * 255.0
    return np.clip(pixel, 0, 255).astype(np.uint8)


def _find_sync_pulses(inst_freq: np.ndarray, rate: int,
                      threshold: float = 1350) -> np.ndarray:
    """
    Locate horizontal sync pulses (regions where inst_freq ≈ 1200 Hz).
    Returns array of sample indices where each sync pulse begins.
    """
    sync_mask  = (inst_freq < threshold).astype(np.int8)
    # Rising edges of the sync mask = start of each pulse
    edges      = np.diff(sync_mask, prepend=0)
    candidates = np.where(edges == 1)[0]

    # Filter: sync pulse must be ≥ 80 % of T_SYNC duration
    min_len = int(T_SYNC * rate * 0.80)
    valid   = []
    prev    = -9999
    for idx in candidates:
        if idx - prev < min_len:
            continue
        # Measure how long inst_freq stays below threshold
        end = idx
        while end < len(sync_mask) and sync_mask[end]:
            end += 1
        if (end - idx) >= min_len:
            valid.append(idx)
            prev = idx

    return np.array(valid, dtype=int)


def _extract_channel(inst_freq: np.ndarray, start_sample: int,
                     rate: int) -> np.ndarray:
    """
    Extract one colour-channel row (320 pixels) starting at start_sample.
    """
    pixel_samples = int(round(T_PIXEL * rate))
    pixels = []
    pos = start_sample
    for _ in range(PIXELS_PER_LINE):
        end = pos + pixel_samples
        if end > len(inst_freq):
            pixels.append(0)
        else:
            mean_freq = float(np.mean(inst_freq[pos:end]))
            pixels.append(int(np.clip(
                (mean_freq - FREQ_BLACK) / (FREQ_WHITE - FREQ_BLACK) * 255,
                0, 255
            )))
        pos = end
    return np.array(pixels, dtype=np.uint8)


def decode_audio(input_wav: str, output_image: str) -> dict:
    """
    Full decode pipeline: WAV → SSTV demodulation → image file.

    Parameters
    ----------
    input_wav    : str  Path to the received WAV (mono, ~44100 Hz).
    output_image : str  Destination image path (PNG or JPEG).

    Returns
    -------
    dict  Decode metrics (lines recovered, SNR estimate, output path).
    """
    print(f"[DECODER] Reading WAV: {input_wav}")
    rate, samples = wavfile.read(input_wav)
    samples, rate = _to_mono_float(samples, rate)

    print(f"[DECODER] Audio: {len(samples)/rate:.1f} s @ {rate} Hz")
    print("[DECODER] Bandpass filtering 1200–2500 Hz …")
    filtered = _bandpass(samples, rate)

    print("[DECODER] Computing instantaneous frequency …")
    inst_freq = _instantaneous_frequency(filtered, rate)

    print("[DECODER] Searching for sync pulses …")
    sync_positions = _find_sync_pulses(inst_freq, rate)
    print(f"[DECODER] Found {len(sync_positions)} sync pulses "
          f"(need {NUM_LINES})")

    if len(sync_positions) < 10:
        raise RuntimeError(
            "Too few sync pulses detected. Check RF/audio quality."
        )

    # ── Reconstruct image ──────────────────────────────────────────────────
    image_array = np.zeros((NUM_LINES, PIXELS_PER_LINE, 3), dtype=np.uint8)

    sync_len  = int(T_SYNC * rate)
    porch_len = int(T_SYNC_PORCH * rate)
    chan_len  = int(T_CHAN_PORCH * rate)
    chan_data_len = int(T_PIXEL * rate) * PIXELS_PER_LINE

    lines_recovered = 0
    for line_idx, sync_pos in enumerate(sync_positions[:NUM_LINES]):
        if line_idx >= NUM_LINES:
            break

        # Pointer after sync + sync porch
        ptr = sync_pos + sync_len + porch_len

        # Martin M1 channel order: G → B → R
        channels = {}
        for ch_name in ('G', 'B', 'R'):
            ptr += chan_len                           # channel separator porch
            row = _extract_channel(inst_freq, ptr, rate)
            channels[ch_name] = row
            ptr += chan_data_len

        image_array[line_idx, :, 0] = channels['R']
        image_array[line_idx, :, 1] = channels['G']
        image_array[line_idx, :, 2] = channels['B']
        lines_recovered += 1

        if line_idx % 32 == 0:
            print(f"  Decoded line {line_idx}/{NUM_LINES}")

    # ── Write output image ─────────────────────────────────────────────────
    out_img = Image.fromarray(image_array, mode="RGB")
    out_img.save(output_image)

    # ── SNR estimate from sync regions ────────────────────────────────────
    sync_freqs = []
    for sp in sync_positions[:50]:
        end = min(sp + sync_len, len(inst_freq))
        sync_freqs.extend(inst_freq[sp:end].tolist())
    sync_arr = np.array(sync_freqs)
    snr_est  = 20 * np.log10(FREQ_SYNC / (np.std(sync_arr) + 1e-9))

    metadata = {
        "input_wav"       : input_wav,
        "output_image"    : output_image,
        "sync_pulses"     : len(sync_positions),
        "lines_recovered" : lines_recovered,
        "sample_rate"     : rate,
        "snr_estimate_dB" : round(snr_est, 1),
        "file_size_kb"    : round(os.path.getsize(output_image) / 1024, 1),
    }

    print(f"[DECODER] Complete. Lines: {lines_recovered}/{NUM_LINES}, "
          f"Estimated SNR: {snr_est:.1f} dB")
    return metadata


# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sstv_decoder.py <input.wav> <output_image.png>")
        sys.exit(1)
    meta = decode_audio(sys.argv[1], sys.argv[2])
    for k, v in meta.items():
        print(f"  {k:>20}: {v}")
