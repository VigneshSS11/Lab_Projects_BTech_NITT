"""
main.py
========
Master orchestrator for the SSTV-Pluto Bridge project.

Modes
-----
  1. encode     – Image  →  SSTV WAV
  2. transmit   – SSTV WAV  →  RF (PlutoSDR TX + RX)  →  received WAV
  3. decode     – Received WAV  →  Reconstructed image
  4. analyze    – Received WAV  →  SNR metrics + spectrogram
  5. full       – All of the above in sequence
  6. sim        – Full pipeline using software loopback (no hardware)

Usage Examples
--------------
  # Full hardware run
  python main.py full --image photo.jpg --freq 433.5e6 --uri ip:192.168.2.1

  # Software simulation (no PlutoSDR needed)
  python main.py sim --image photo.jpg --snr 25

  # Encode only
  python main.py encode --image photo.jpg

  # Decode a received WAV
  python main.py decode --rx-wav sstv_received.wav

Author : SSTV-Pluto Bridge Project
"""

import argparse
import os
import sys
import json
import time

from sstv_encoder      import encode_image
from sstv_decoder      import decode_audio
from snr_analyzer      import analyze
from pluto_transceiver import SimulatedTransceiver

# ── Paths ────────────────────────────────────────────────────────────────────
DEFAULT_IMAGE    = "input.jpg"
ENCODED_WAV      = "sstv_encoded.wav"
RECEIVED_WAV     = "sstv_received.wav"
DECODED_IMAGE    = "sstv_decoded.png"
REPORT_JSON      = "sstv_report.json"
OUTPUT_DIR       = "output"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _out(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


# ════════════════════════════════════════════════════════════════════════════
# Stage Runners
# ════════════════════════════════════════════════════════════════════════════

def run_encode(image_path: str) -> dict:
    _ensure_output_dir()
    print("\n" + "─" * 60)
    print("  STAGE 1 – SSTV ENCODING")
    print("─" * 60)
    meta = encode_image(image_path, _out(ENCODED_WAV))
    print(f"  ✓ Encoded WAV: {_out(ENCODED_WAV)}")
    return meta


def run_transmit(freq: float, uri: str, gain: float,
                 atten: float, duration: float = None) -> dict:
    """Start the GNU Radio flowgraph for RF transmission/reception."""
    _ensure_output_dir()
    print("\n" + "─" * 60)
    print("  STAGE 2 – RF TRANSMISSION  (PlutoSDR)")
    print("─" * 60)

    try:
        from pluto_transceiver import SSTVTransceiver, GR_AVAILABLE
        if not GR_AVAILABLE:
            raise ImportError("GNU Radio not available")

        tb = SSTVTransceiver(
            tx_wav_path = _out(ENCODED_WAV),
            rx_wav_path = _out(RECEIVED_WAV),
            center_freq = freq,
            rx_gain     = gain,
            tx_atten    = atten,
            uri         = uri,
        )

        # Determine run time from WAV
        import wave as wavemod
        with wavemod.open(_out(ENCODED_WAV)) as wf:
            wav_dur = wf.getnframes() / wf.getframerate()
        run_dur = duration or (wav_dur + 5.0)

        print(f"  Transmitting {wav_dur:.1f} s of SSTV audio …")
        tb.start()
        time.sleep(run_dur)
        tb.stop()
        tb.wait()

    except ImportError:
        print("  [!] GNU Radio not found — using simulated loopback.")
        trx = SimulatedTransceiver(
            tx_wav      = _out(ENCODED_WAV),
            rx_wav      = _out(RECEIVED_WAV),
            center_freq = freq,
            snr_db      = 30.0,
        )
        trx.run_loopback()

    print(f"  ✓ Received WAV: {_out(RECEIVED_WAV)}")
    return {"received_wav": _out(RECEIVED_WAV)}


def run_sim(freq: float, snr_db: float) -> dict:
    """Software loopback – no hardware needed."""
    _ensure_output_dir()
    print("\n" + "─" * 60)
    print("  STAGE 2 (SIM) – SOFTWARE LOOPBACK CHANNEL")
    print("─" * 60)
    trx = SimulatedTransceiver(
        tx_wav      = _out(ENCODED_WAV),
        rx_wav      = _out(RECEIVED_WAV),
        center_freq = freq,
        snr_db      = snr_db,
    )
    actual_snr = trx.run_loopback()
    print(f"  ✓ Simulated channel complete. SNR ≈ {actual_snr:.1f} dB")
    return {"received_wav": _out(RECEIVED_WAV), "sim_snr_dB": actual_snr}


def run_decode(rx_wav: str = None) -> dict:
    _ensure_output_dir()
    print("\n" + "─" * 60)
    print("  STAGE 3 – SSTV DECODING")
    print("─" * 60)
    wav = rx_wav or _out(RECEIVED_WAV)
    meta = decode_audio(wav, _out(DECODED_IMAGE))
    print(f"  ✓ Decoded image: {_out(DECODED_IMAGE)}")
    return meta


def run_analyze(rx_wav: str = None) -> dict:
    _ensure_output_dir()
    print("\n" + "─" * 60)
    print("  STAGE 4 – SNR ANALYSIS")
    print("─" * 60)
    wav = rx_wav or _out(RECEIVED_WAV)
    metrics = analyze(wav, OUTPUT_DIR)
    return metrics


def save_report(all_meta: dict):
    path = _out(REPORT_JSON)
    with open(path, "w") as f:
        json.dump(all_meta, f, indent=2, default=str)
    print(f"\n  ✓ Full report saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("mode",
                   choices=["encode", "transmit", "decode",
                            "analyze", "full", "sim"],
                   help="Pipeline stage to run")

    # Common
    p.add_argument("--image",  default=DEFAULT_IMAGE,
                   help=f"Input image path (default: {DEFAULT_IMAGE})")
    p.add_argument("--rx-wav", default=None,
                   help="Received WAV path (for decode/analyze modes)")

    # RF parameters
    p.add_argument("--freq",  type=float, default=433e6,
                   help="Carrier frequency in Hz (default: 433 MHz)")
    p.add_argument("--uri",   default="ip:192.168.2.1",
                   help="PlutoSDR libiio URI")
    p.add_argument("--gain",  type=float, default=50.0,
                   help="RX gain in dB (default: 50)")
    p.add_argument("--atten", type=float, default=0.0,
                   help="TX attenuation dBFS (default: 0)")
    p.add_argument("--duration", type=float, default=None,
                   help="Force flowgraph run time in seconds")

    # Simulation
    p.add_argument("--snr",   type=float, default=30.0,
                   help="Channel SNR for simulation mode (default: 30 dB)")
    return p


def main():
    args = _build_parser().parse_args()
    report = {}

    t0 = time.time()

    if args.mode == "encode":
        report.update(run_encode(args.image))

    elif args.mode == "transmit":
        report.update(run_transmit(args.freq, args.uri,
                                   args.gain, args.atten, args.duration))

    elif args.mode == "decode":
        report.update(run_decode(args.rx_wav))

    elif args.mode == "analyze":
        report.update(run_analyze(args.rx_wav))

    elif args.mode == "full":
        # Hardware full pipeline
        report.update(run_encode(args.image))
        report.update(run_transmit(args.freq, args.uri,
                                   args.gain, args.atten, args.duration))
        report.update(run_decode())
        report.update(run_analyze())
        save_report(report)

    elif args.mode == "sim":
        # Software-only full pipeline
        report.update(run_encode(args.image))
        report.update(run_sim(args.freq, args.snr))
        report.update(run_decode())
        report.update(run_analyze())
        save_report(report)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f} s\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
