"""
sstv_encoder.py
================
Converts a digital image into SSTV (Slow Scan Television) audio using
the Martin M1 protocol. Each pixel's colour channel is mapped to a
frequency in the 1500–2300 Hz audio band. The resulting WAV file is
later used as the baseband signal for FM modulation.

Martin M1 Mode Specification
------------------------------
  VIS code         : 44
  Lines            : 256
  Pixels per line  : 320
  Channels / line  : Green → Blue → Red  (3 passes)
  Pixel duration   : 0.4576 ms
  Line sync        : 1200 Hz  /  4.862 ms
  Sync porch       : 1500 Hz  /  0.572 ms
  Channel porch    : 1500 Hz  /  0.572 ms
  Frequency mapping: 1500 Hz = black (0)  ↔  2300 Hz = white (255)

Author : SSTV-Pluto Bridge Project
"""

import numpy as np
import wave
import struct
from PIL import Image
import os

# ─── Audio & Timing Constants ───────────────────────────────────────────────
SAMPLE_RATE  = 44100          # Hz – standard audio sample rate
BITS         = 16             # WAV bit depth

# ─── Frequency Definitions ─────────────────────────────────────────────────
FREQ_SYNC    = 1200           # Hz – sync pulse
FREQ_BLACK   = 1500           # Hz – black level / porch / separator
FREQ_WHITE   = 2300           # Hz – peak white

# ─── Martin M1 Timing (seconds) ────────────────────────────────────────────
T_LEADER     = 0.300          # 300 ms leader tone at 1900 Hz
T_BREAK      = 0.010          # 10 ms VIS break at 1200 Hz
T_VIS_BIT    = 0.030          # 30 ms per VIS bit
T_SYNC       = 0.004862       # Horizontal sync pulse
T_SYNC_PORCH = 0.000572       # Sync porch
T_CHAN_PORCH = 0.000572       # Channel separator porch
T_PIXEL      = 0.0004576      # Time per pixel (320 px per channel)

VIS_CODE     = 44             # Martin M1 VIS identifier


def _tone(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a pure sine-wave tone as float64 samples in [-1, 1]."""
    n_samples = int(round(duration * sample_rate))
    t = np.linspace(0, duration, n_samples, endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def _pixel_freq(value: int) -> float:
    """Map a pixel intensity [0-255] to its SSTV frequency [1500-2300 Hz]."""
    return FREQ_BLACK + (FREQ_WHITE - FREQ_BLACK) * (value / 255.0)


def _vis_header() -> np.ndarray:
    """
    Build the VIS (Vertical Interval Signalling) header.
    Layout:
        300 ms  leader  @ 1900 Hz
         10 ms  break   @ 1200 Hz
        300 ms  leader  @ 1900 Hz
         30 ms  start   @ 1200 Hz
         8 bits VIS code (LSB first): 1→1100 Hz, 0→1300 Hz, 30 ms each
          1 bit even parity           same encoding
         30 ms  stop    @ 1200 Hz
    """
    segments = []

    # Leader – break – leader
    segments.append(_tone(1900, T_LEADER))
    segments.append(_tone(FREQ_SYNC, T_BREAK))
    segments.append(_tone(1900, T_LEADER))

    # Start bit
    segments.append(_tone(FREQ_SYNC, T_VIS_BIT))

    # 8 VIS bits (LSB first)
    bits = [(VIS_CODE >> i) & 1 for i in range(8)]
    for bit in bits:
        freq = 1100 if bit else 1300
        segments.append(_tone(freq, T_VIS_BIT))

    # Even parity bit
    parity = sum(bits) % 2
    freq = 1100 if parity else 1300
    segments.append(_tone(freq, T_VIS_BIT))

    # Stop bit
    segments.append(_tone(FREQ_SYNC, T_VIS_BIT))

    return np.concatenate(segments)


def _encode_line(r_row: np.ndarray, g_row: np.ndarray, b_row: np.ndarray) -> np.ndarray:
    """
    Encode a single image line in Martin M1 order: sync → G → B → R.
    Each 320-pixel row is encoded as a continuous FM sweep where sample
    frequencies change pixel-by-pixel.
    """
    segments = []

    # Horizontal sync
    segments.append(_tone(FREQ_SYNC, T_SYNC))
    segments.append(_tone(FREQ_BLACK, T_SYNC_PORCH))

    # Channels: Green → Blue → Red
    for row in (g_row, b_row, r_row):
        # Channel porch before each colour block
        segments.append(_tone(FREQ_BLACK, T_CHAN_PORCH))

        # Pixel data – build freq-modulated sweep per pixel
        pixel_samples = int(round(T_PIXEL * SAMPLE_RATE))
        channel_audio = []
        for pixel_val in row:
            freq = _pixel_freq(int(pixel_val))
            n = pixel_samples
            t = np.arange(n) / SAMPLE_RATE
            channel_audio.append(np.sin(2 * np.pi * freq * t))
        segments.append(np.concatenate(channel_audio))

    return np.concatenate(segments)


def encode_image(image_path: str, output_wav: str) -> dict:
    """
    Full pipeline: load image → resize → encode → write WAV.

    Parameters
    ----------
    image_path : str   Path to any PIL-readable image.
    output_wav : str   Destination WAV file path.

    Returns
    -------
    dict  Metadata including duration and file size.
    """
    print(f"[ENCODER] Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")

    # Martin M1 requires exactly 320 × 256
    img = img.resize((320, 256), Image.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)  # shape (256, 320, 3)

    r_data = img_array[:, :, 0]  # (256, 320)
    g_data = img_array[:, :, 1]
    b_data = img_array[:, :, 2]

    print("[ENCODER] Building VIS header …")
    audio = [_vis_header()]

    print("[ENCODER] Encoding 256 scan lines …")
    for line_idx in range(256):
        if line_idx % 32 == 0:
            print(f"  Line {line_idx}/256")
        audio.append(
            _encode_line(r_data[line_idx], g_data[line_idx], b_data[line_idx])
        )

    full_audio = np.concatenate(audio)

    # Normalise and convert to int16
    peak = np.max(np.abs(full_audio))
    if peak > 0:
        full_audio /= peak
    audio_int16 = (full_audio * 32767).astype(np.int16)

    # Write WAV
    with wave.open(output_wav, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)           # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    duration  = len(audio_int16) / SAMPLE_RATE
    file_size = os.path.getsize(output_wav)

    metadata = {
        "image_path"  : image_path,
        "output_wav"  : output_wav,
        "image_size"  : (320, 256),
        "total_lines" : 256,
        "duration_s"  : round(duration, 2),
        "sample_rate" : SAMPLE_RATE,
        "file_size_kb": round(file_size / 1024, 1),
    }

    print(f"[ENCODER] Done. Duration: {duration:.1f} s, "
          f"File size: {metadata['file_size_kb']} KB")
    return metadata


# ─── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sstv_encoder.py <image_path> <output.wav>")
        sys.exit(1)
    meta = encode_image(sys.argv[1], sys.argv[2])
    for k, v in meta.items():
        print(f"  {k:>15}: {v}")
