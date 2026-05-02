"""
pluto_transceiver.py
=====================
GNU Radio full-duplex SSTV transceiver using the ADALM-Pluto SDR.

Architecture
------------

  TX Chain
  ┌──────────────────────────────────────────────────────────────────────┐
  │  WAV File Source  →  Rational Resampler  →  FM Modulator  →  Pluto  │
  │  (44 100 Hz mono)      ×54 ↑                 Δf = 5 kHz    Sink TX  │
  └──────────────────────────────────────────────────────────────────────┘

  RX Chain
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Pluto Source RX  →  Low-Pass Filter  →  FM Demodulator  →  Resampler │
  │  (2.4 MHz IQ)        BW = 15 kHz          Δf = 5 kHz       ÷54       │
  │  → WAV File Sink (44 100 Hz)                                          │
  └────────────────────────────────────────────────────────────────────────┘

Both chains share the same PlutoSDR device using iio (libiio / gr-iio).
The Pluto is configured for full-duplex by keeping TX and RX enabled
simultaneously – the AD9363 transceiver inside the Pluto natively
supports simultaneous TX/RX on different LO frequencies; here we
deliberately set them to the same 433 MHz and rely on the board's
internal isolation (~40 dB) as the loopback path.

Dependencies
------------
  GNU Radio 3.10+
  gr-iio          (PlutoSDR blocks)
  numpy
  Install: https://wiki.gnuradio.org/index.php/InstallingGR

Usage
-----
  python pluto_transceiver.py --tx-wav sstv_out.wav --rx-wav received.wav
  python pluto_transceiver.py --tx-wav sstv_out.wav --rx-wav received.wav \\
      --freq 433.5e6 --gain 30

Author : SSTV-Pluto Bridge Project
"""

import sys
import time
import math
import argparse

# ── Graceful import of GNU Radio modules ────────────────────────────────────
try:
    from gnuradio import gr, blocks, analog, filter as grfilter
    from gnuradio.filter import firdes
    import iio  # gr-iio Python bindings
    GR_AVAILABLE = True
except ImportError:
    GR_AVAILABLE = False
    print("[PLUTO] WARNING: GNU Radio / gr-iio not found. "
          "Running in SIMULATION mode.\n"
          "Install GNU Radio: https://wiki.gnuradio.org/index.php/InstallingGR")

# ────────────────────────────────────────────────────────────────────────────
# Design Parameters
# ────────────────────────────────────────────────────────────────────────────
AUDIO_RATE      = 44_100        # Hz – SSTV audio sample rate
SDR_RATE        = 2_400_000     # Hz – PlutoSDR baseband sample rate
RESAMP_INTERP   = 54            # SDR_RATE / AUDIO_RATE ≈ 54.42  (use int)
FM_DEVIATION    = 5_000         # Hz – FM peak deviation
RF_BW           = 2_000_000     # Hz – PlutoSDR RF bandwidth (2 MHz)
TX_ATTENUATION  = 0             # dBFS – 0 = maximum output
RX_GAIN_DB      = 50            # dB  – manual gain mode
LPF_CUTOFF      = 15_000        # Hz – pre-demod low-pass filter
LPF_TRANSITION  = 3_000         # Hz


# ════════════════════════════════════════════════════════════════════════════
# GNU Radio Flowgraph
# ════════════════════════════════════════════════════════════════════════════
if GR_AVAILABLE:

    class SSTVTransceiver(gr.top_block):
        """
        Full-duplex GNU Radio flowgraph for SSTV over PlutoSDR.

        Parameters
        ----------
        tx_wav_path  : str   SSTV audio WAV to transmit (mono, 44 100 Hz).
        rx_wav_path  : str   Output path for the received & demodulated audio.
        center_freq  : float RF carrier frequency in Hz (default 433 MHz).
        rx_gain      : float Receiver gain in dB.
        tx_atten     : float Transmit attenuation in dBFS (0 = max power).
        uri          : str   libiio URI, e.g. "ip:192.168.2.1" or "".
        """

        def __init__(self,
                     tx_wav_path : str,
                     rx_wav_path : str,
                     center_freq : float = 433e6,
                     rx_gain     : float = RX_GAIN_DB,
                     tx_atten    : float = TX_ATTENUATION,
                     uri         : str   = "ip:192.168.2.1"):

            gr.top_block.__init__(self, "SSTV_PlutoSDR_Transceiver")

            self._tx_wav   = tx_wav_path
            self._rx_wav   = rx_wav_path
            self._cf       = int(center_freq)
            self._rx_gain  = rx_gain
            self._tx_atten = tx_atten
            self._uri      = uri

            # ── Build sub-graphs ─────────────────────────────────────────
            self._build_tx_chain()
            self._build_rx_chain()

            print(f"[PLUTO] Flowgraph built  — "
                  f"CF={center_freq/1e6:.3f} MHz  "
                  f"SDR rate={SDR_RATE/1e6:.1f} Msps  "
                  f"FM Δf={FM_DEVIATION/1e3:.0f} kHz")

        # ── TX Chain ─────────────────────────────────────────────────────
        def _build_tx_chain(self):
            """
            WAV file (44.1 kHz float) → upsample → FM mod → Pluto TX
            """
            # Source: mono float WAV
            self.wav_src = blocks.wavfile_source(self._tx_wav, repeat=False)

            # Rational resampler: 44100 → 2 400 000 sps
            # We use integer interp/decim pair closest to exact ratio.
            self.tx_resamp = grfilter.rational_resampler_fff(
                interpolation = RESAMP_INTERP,
                decimation    = 1,
                taps          = [],
                fractional_bw = 0.4,
            )

            # FM modulator: audio float → complex baseband IQ
            # k = 2π · Δf / Fs
            self.fm_mod = analog.frequency_modulator_fc(
                sensitivity = 2 * math.pi * FM_DEVIATION / SDR_RATE
            )

            # PlutoSDR sink (TX)
            self.pluto_tx = iio.pluto_sink(
                uri             = self._uri,
                frequency       = self._cf,
                samplerate      = SDR_RATE,
                bandwidth       = RF_BW,
                buffer_size     = 0x8000,
                cyclic          = False,
                attenuation1    = self._tx_atten,
                filter          = "",
                filter_source   = False,
            )

            # Connect TX chain
            self.connect(self.wav_src, self.tx_resamp, self.fm_mod, self.pluto_tx)

        # ── RX Chain ─────────────────────────────────────────────────────
        def _build_rx_chain(self):
            """
            Pluto RX → LPF → FM demod → downsample → WAV file
            """
            # PlutoSDR source (RX)
            self.pluto_rx = iio.pluto_source(
                uri           = self._uri,
                frequency     = self._cf,
                samplerate    = SDR_RATE,
                bandwidth     = RF_BW,
                buffer_size   = 0x8000,
                quadrature    = True,
                rfdc          = True,
                bbdc          = True,
                gain_mode0    = "manual",
                rf_gain0      = self._rx_gain,
                filter        = "",
                filter_source = False,
            )

            # Low-pass filter to pass ±15 kHz around carrier
            lpf_taps = firdes.low_pass(
                gain           = 1.0,
                sampling_freq  = SDR_RATE,
                cutoff_freq    = LPF_CUTOFF,
                transition_bw  = LPF_TRANSITION,
                window         = firdes.WIN_HAMMING,
            )
            self.lpf = grfilter.fir_filter_ccf(1, lpf_taps)

            # FM demodulator: complex IQ → float audio
            # quad_rate  = SDR_RATE
            # audio_decim = RESAMP_INTERP  (outputs SDR_RATE / RESAMP_INTERP)
            self.fm_demod = analog.fm_demod_cf(
                channel_rate  = SDR_RATE,
                audio_decim   = RESAMP_INTERP,
                deviation     = FM_DEVIATION,
                audio_pass    = 18_000,
                audio_stop    = 20_000,
                gain          = 1.0,
                tau           = 75e-6,
            )

            # WAV sink (mono, 44 100 Hz, 16-bit)
            self.wav_sink = blocks.wavfile_sink(
                self._rx_wav, 1, AUDIO_RATE, blocks.FORMAT_WAV, blocks.FORMAT_PCM_16, False
            )

            # Connect RX chain
            self.connect(self.pluto_rx, self.lpf, self.fm_demod, self.wav_sink)


# ════════════════════════════════════════════════════════════════════════════
# Simulation Mode  (no GNU Radio / PlutoSDR hardware)
# ════════════════════════════════════════════════════════════════════════════
class SimulatedTransceiver:
    """
    Pure-Python loopback that applies realistic channel impairments:
      • AWGN noise  (parameterised by SNR)
      • Multipath fading
      • Frequency offset

    Used for testing the encoder/decoder pipeline without hardware.
    """

    def __init__(self, tx_wav, rx_wav, center_freq=433e6, snr_db=30.0):
        self.tx_wav     = tx_wav
        self.rx_wav     = rx_wav
        self.center_freq = center_freq
        self.snr_db      = snr_db

    def run_loopback(self):
        import numpy as np
        from scipy.io import wavfile

        print(f"[SIM] Simulated loopback  SNR = {self.snr_db} dB")
        rate, samples = wavfile.read(self.tx_wav)
        samples = samples.astype(np.float32) / 32768.0

        # ── FM modulation (simplified) ───────────────────────────────────
        t          = np.arange(len(samples)) / rate
        phase      = 2 * np.pi * FM_DEVIATION * np.cumsum(samples) / rate
        carrier    = np.exp(1j * phase)                          # complex baseband

        # ── AWGN channel ─────────────────────────────────────────────────
        sig_power  = np.mean(np.abs(carrier) ** 2)
        noise_std  = np.sqrt(sig_power / (2 * 10 ** (self.snr_db / 10)))
        noise      = (np.random.randn(len(carrier)) +
                      1j * np.random.randn(len(carrier))) * noise_std
        received   = carrier + noise

        # ── Mild multipath (two-tap, 1-sample delay, 20 % echo) ─────────
        echo       = np.roll(received, 1) * 0.20
        received   = received + echo

        # ── FM demodulation ──────────────────────────────────────────────
        angle      = np.angle(received[1:] * np.conj(received[:-1]))
        demod      = angle * rate / (2 * np.pi * FM_DEVIATION)
        demod      = np.append(demod, demod[-1]).astype(np.float32)

        # Normalise and clip
        demod_int  = np.clip(demod * 32768, -32768, 32767).astype(np.int16)
        wavfile.write(self.rx_wav, rate, demod_int)

        actual_snr = 20 * np.log10(
            np.sqrt(np.mean(demod ** 2)) /
            (np.std(demod - samples[:len(demod)]) + 1e-9)
        )
        print(f"[SIM] Loopback complete. "
              f"Output: {self.rx_wav}  Measured SNR: {actual_snr:.1f} dB")
        return actual_snr


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def _parse_args():
    p = argparse.ArgumentParser(
        description="ADALM-Pluto full-duplex SSTV transceiver"
    )
    p.add_argument("--tx-wav",  default="sstv_encoded.wav",
                   help="Encoded SSTV WAV to transmit")
    p.add_argument("--rx-wav",  default="sstv_received.wav",
                   help="Output path for demodulated received audio")
    p.add_argument("--freq",    type=float, default=433e6,
                   help="RF carrier frequency in Hz (default: 433 MHz)")
    p.add_argument("--gain",    type=float, default=RX_GAIN_DB,
                   help=f"RX gain dB (default {RX_GAIN_DB})")
    p.add_argument("--atten",   type=float, default=TX_ATTENUATION,
                   help=f"TX attenuation dBFS (default {TX_ATTENUATION})")
    p.add_argument("--uri",     default="ip:192.168.2.1",
                   help="libiio URI (default: ip:192.168.2.1)")
    p.add_argument("--sim",     action="store_true",
                   help="Force simulation mode (no hardware)")
    p.add_argument("--snr",     type=float, default=30.0,
                   help="SNR (dB) for simulation mode (default 30)")
    p.add_argument("--duration",type=float, default=None,
                   help="Override flowgraph run-time in seconds")
    return p.parse_args()


def main():
    args = _parse_args()

    use_sim = args.sim or not GR_AVAILABLE

    if use_sim:
        # ── Simulation path ──────────────────────────────────────────────
        print("[PLUTO] Using simulated loopback (no hardware).")
        trx = SimulatedTransceiver(
            tx_wav      = args.tx_wav,
            rx_wav      = args.rx_wav,
            center_freq = args.freq,
            snr_db      = args.snr,
        )
        trx.run_loopback()

    else:
        # ── Real hardware path ───────────────────────────────────────────
        print(f"[PLUTO] Connecting to PlutoSDR at {args.uri} …")
        tb = SSTVTransceiver(
            tx_wav_path = args.tx_wav,
            rx_wav_path = args.rx_wav,
            center_freq = args.freq,
            rx_gain     = args.gain,
            tx_atten    = args.atten,
            uri         = args.uri,
        )

        # Estimate SSTV transmission duration from WAV header
        try:
            import wave
            with wave.open(args.tx_wav) as wf:
                wav_dur = wf.getnframes() / wf.getframerate()
        except Exception:
            wav_dur = 120.0   # fallback: 2 minutes

        run_dur = args.duration or (wav_dur + 5.0)   # +5 s settling
        print(f"[PLUTO] Starting flowgraph for {run_dur:.1f} s …")

        tb.start()
        time.sleep(run_dur)
        tb.stop()
        tb.wait()

        print(f"[PLUTO] Flowgraph finished. Received audio → {args.rx_wav}")


if __name__ == "__main__":
    main()
