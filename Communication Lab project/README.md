# Real-Time Wireless Image Transceiver
## Full-Duplex SSTV Bridge using the ADALM-Pluto SDR

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [SSTV Protocol Deep-Dive](#3-sstv-protocol-deep-dive)
4. [Signal Chain & DSP Theory](#4-signal-chain--dsp-theory)
5. [Hardware Setup](#5-hardware-setup)
6. [Software Installation](#6-software-installation)
7. [Running the Project](#7-running-the-project)
8. [File Reference](#8-file-reference)
9. [Troubleshooting](#9-troubleshooting)
10. [Further Experiments](#10-further-experiments)

---

## 1. Project Overview

This project implements a **full-duplex wireless image transceiver** that:

- Converts a digital photo into a stream of audio-frequency tones using the **Martin M1 SSTV** (Slow Scan Television) protocol
- **FM-modulates** that audio onto a 433 MHz RF carrier using the **ADALM-Pluto SDR**
- Simultaneously **receives and demodulates** the signal on the same device
- Reconstructs the image line-by-line from the received baseband
- Computes **SNR metrics** and generates a **spectrogram**

The Pluto's AD9363 transceiver natively supports simultaneous TX/RX, making true full-duplex operation possible.

```
   ┌───────────┐   SSTV tones    ┌──────────────────┐   FM @ 433 MHz   ┌────────┐
   │ Input PNG │ ──────────────► │  sstv_encoder.py  │ ───────────────► │ Pluto  │
   └───────────┘                 └──────────────────┘                   │  TX ──►│
                                                                        │        │  (air gap / loopback)
   ┌───────────┐   image.png     ┌──────────────────┐   FM audio        │  RX ◄──│
   │  Decoded  │ ◄────────────── │  sstv_decoder.py  │ ◄─────────────── │        │
   │  Image    │                 └──────────────────┘                   └────────┘
   └───────────┘         ▲
                         │ SNR, spectrogram
                 ┌───────────────┐
                 │ snr_analyzer  │
                 └───────────────┘
```

---

## 2. System Architecture

### File Map

```
sstv_pluto_bridge/
├── main.py                  ← Master orchestrator (run this)
├── sstv_encoder.py          ← Image → Martin M1 SSTV audio
├── sstv_decoder.py          ← Received audio → reconstructed image
├── pluto_transceiver.py     ← GNU Radio full-duplex flowgraph
├── snr_analyzer.py          ← SNR metrics + spectrogram
├── requirements.txt
└── README.md
```

### Data Flow

```
Input Image (any format)
        │
        ▼  [sstv_encoder.py]
  SSTV WAV (44 100 Hz, mono)
  • VIS header (Martin M1, code 44)
  • 256 × (sync + G + B + R scan lines)
  • Pixels mapped 1500–2300 Hz
        │
        ▼  [pluto_transceiver.py  –  TX chain]
  FM Modulated IQ  @ 2.4 Msps
  • Modulation index ≈ 5 kHz deviation
  • Upsampled 44 100 → 2 400 000 sps
        │
        ▼  433 MHz carrier  ──► (antenna / cable) ──►
        │
        ▼  [pluto_transceiver.py  –  RX chain]
  FM Demodulated audio  @ 44 100 Hz
  • LPF 15 kHz → FM demod → decimate
        │
        ▼  [sstv_decoder.py]
  Reconstructed RGB image (320 × 256 PNG)
  • Sync detection (Hilbert + envelope)
  • Instantaneous frequency → pixel mapping
        │
        ▼  [snr_analyzer.py]
  SNR report + spectrogram PNG
```

---

## 3. SSTV Protocol Deep-Dive

### Why SSTV?

SSTV is a real-world analogue protocol used by ham radio operators since the 1950s to transmit images over voice radio channels. It encodes image data as **audio tones**, making it naturally compatible with any FM transceiver – including the Pluto.

### Martin M1 Mode

| Parameter              | Value             |
|------------------------|-------------------|
| VIS code               | 44                |
| Image size             | 320 × 256 pixels  |
| Colour order           | Green → Blue → Red |
| Pixel frequency range  | 1500–2300 Hz      |
| Pixel duration         | 0.4576 ms         |
| Line sync frequency    | 1200 Hz           |
| Line sync duration     | 4.862 ms          |
| Sync porch             | 1500 Hz / 0.572 ms |
| Channel porch          | 1500 Hz / 0.572 ms |
| Total line duration    | ≈ 146.43 ms       |
| Full image duration    | ≈ 114.8 s         |

### Frequency Map

```
1100 Hz  ─── VIS bit "1"
1200 Hz  ─── Horizontal sync pulse / VIS break
1300 Hz  ─── VIS bit "0"
1500 Hz  ─── Black level / porches
1900 Hz  ─── VIS leader tone
2300 Hz  ─── White level (peak brightness)
```

### VIS Header Structure

```
 300 ms @ 1900 Hz    Leader
  10 ms @ 1200 Hz    Break
 300 ms @ 1900 Hz    Leader
  30 ms @ 1200 Hz    Start bit
  8 × 30 ms          VIS bits (LSB first): 1→1100, 0→1300 Hz
  1 × 30 ms          Even parity bit
  30 ms @ 1200 Hz    Stop bit
```

VIS code 44 in binary (LSB first): **0 0 1 1 0 1 0 0**

---

## 4. Signal Chain & DSP Theory

### 4.1 FM Modulation

The SSTV audio tones are frequency-modulated onto the 433 MHz carrier:

```
s(t) = A · cos( 2π·fc·t  +  2π·Δf · ∫ m(τ)dτ )
```

Where:
- `fc` = 433 MHz carrier frequency
- `Δf` = 5 000 Hz peak FM deviation
- `m(t)` = normalised SSTV audio (tones 1500–2300 Hz)

The **modulation index** β = Δf / f_audio ≈ 5000/1900 ≈ 2.6 (wideband FM).  
Carson's rule → occupied RF bandwidth ≈ 2(Δf + f_max) ≈ 14.6 kHz.

### 4.2 Instantaneous Frequency Demodulation

The decoder extracts the audio from the received IQ using:

```
φ(t) = unwrap( arg( x(t) + j·H{x(t)} ) )     [analytic signal]

f_inst(t) = 1/(2π) · dφ/dt
```

Where `H{}` denotes the Hilbert transform. This is equivalent to what the FM demod block in GNU Radio implements.

### 4.3 Sync Detection

The 1200 Hz sync pulse is detected by finding regions where `f_inst(t) < 1350 Hz` for ≥ 80 % of the expected sync duration (4.862 ms). This threshold approach is robust to moderate noise.

### 4.4 SNR Definition

```
SNR_wideband = 10·log10( P_signal / P_noise )

P_signal = mean power in  1100–2500 Hz
P_noise  = mean power in 200–1000 Hz  ∪  2700–5000 Hz
```

---

## 5. Hardware Setup

### Required

- ADALM-Pluto SDR (any firmware ≥ 0.31)
- USB 2.0 or 3.0 port
- Two short monopole / SMA antennas  (or an SMA attenuator cable for loopback)

### Frequency Allocation

| Path         | Frequency   | Notes                           |
|-------------|-------------|----------------------------------|
| TX carrier  | 433.000 MHz | 70 cm ISM band (licence-free)   |
| RX centre   | 433.000 MHz | Same as TX for loopback          |

> **Legal note**: 433 MHz is ISM licence-exempt in most countries at low power.  
> Always check local regulations. Keep TX attenuation ≥ 60 dBFS when testing indoors.

### Connection Diagram

```
┌─────────┐            ┌──────────────────────────────┐
│ Computer│ ◄── USB ──►│       ADALM-Pluto SDR         │
└─────────┘            │                               │
                       │  TX1A ──[ attenuator ]──► RX1A│  (cable loopback)
                       │           or                  │
                       │  TX1A ──[antenna]  [antenna]──►RX1A  (over-the-air) │
                       └──────────────────────────────┘
```

### PlutoSDR Network Configuration

Default IP: `192.168.2.1`  
Test connectivity: `ping 192.168.2.1`

To use USB serial instead:
```bash
iio_info -s        # list detected devices
iio_info -u usb:   # connect via USB
```

---

## 6. Software Installation

### Step 1: Python environment

```bash
python3 -m venv sstv-env
source sstv-env/bin/activate
pip install numpy scipy Pillow matplotlib pyadi-iio
```

### Step 2: GNU Radio (for hardware TX/RX)

**Ubuntu / Debian:**
```bash
sudo add-apt-repository ppa:gnuradio/gnuradio-releases
sudo apt update
sudo apt install gnuradio gr-iio libiio-utils python3-iio
```

**conda-forge (cross-platform, recommended):**
```bash
conda create -n gnuradio -c conda-forge gnuradio gnuradio-iio
conda activate gnuradio
```

**Verify:**
```bash
gnuradio-config-info --version    # should show 3.10.x
python3 -c "from gnuradio import gr; print(gr.version())"
```

### Step 3: PlutoSDR firmware (if needed)

```bash
# Check current firmware
ssh root@192.168.2.1 cat /etc/adi_plutosdr_fw_version
# Update to ≥ 0.38
# Download from: https://github.com/analogdevicesinc/plutosdr-fw/releases
```

---

## 7. Running the Project

### Quick start – simulation (no hardware needed)

```bash
# Place any image in the project folder
cp /path/to/photo.jpg input.jpg

# Run full pipeline with software loopback
python main.py sim --image input.jpg --snr 30
```

Outputs in `output/`:
- `sstv_encoded.wav`      — SSTV audio
- `sstv_received.wav`     — After simulated channel
- `sstv_decoded.png`      — Reconstructed image
- `sstv_spectrogram.png`  — SNR spectrogram
- `sstv_report.json`      — Full metrics

### Hardware run

```bash
# Confirm Pluto is reachable
ping 192.168.2.1

# Full pipeline with real RF
python main.py full \
    --image input.jpg \
    --freq 433e6 \
    --uri ip:192.168.2.1 \
    --gain 50 \
    --atten 0
```

### Individual stages

```bash
# Encode only
python main.py encode --image input.jpg

# Transmit only (requires encoded WAV)
python main.py transmit --freq 433e6 --uri ip:192.168.2.1

# Decode only (requires received WAV)
python main.py decode --rx-wav output/sstv_received.wav

# SNR analysis only
python main.py analyze --rx-wav output/sstv_received.wav
```

### Stage scripts directly

```bash
python sstv_encoder.py  input.jpg           output/sstv_encoded.wav
python sstv_decoder.py  output/sstv_received.wav  output/decoded.png
python snr_analyzer.py  output/sstv_received.wav  output/
```

---

## 8. File Reference

| File                    | Role                                           |
|------------------------|------------------------------------------------|
| `sstv_encoder.py`       | Image → Martin M1 SSTV audio WAV              |
| `sstv_decoder.py`       | Received audio → RGB image (Hilbert demod)    |
| `pluto_transceiver.py`  | GNU Radio TX+RX flowgraph + SimulatedTransceiver |
| `snr_analyzer.py`       | Wideband & sync SNR, per-line SNR, spectrogram |
| `main.py`               | CLI orchestrator for all modes                |
| `requirements.txt`      | Python dependencies                            |

---

## 9. Troubleshooting

| Symptom                             | Likely Cause                          | Fix                                     |
|-------------------------------------|---------------------------------------|-----------------------------------------|
| `ImportError: gnuradio`             | GR not in Python path                 | Use `conda activate gnuradio` env      |
| `No such device` (PlutoSDR)         | USB / IP not connected                | `ping 192.168.2.1`; check USB cable    |
| < 10 sync pulses detected           | Low RX gain or high noise             | Increase `--gain`, move antennas       |
| Decoded image all black/white       | FM demod gain wrong                   | Adjust `RX_GAIN_DB` in transceiver     |
| Horizontal banding in image         | Multipath / timing drift              | Shorten antenna separation             |
| SNR < 15 dB                         | Signal too weak                       | Reduce attenuator, increase gain       |
| WAV file too short for 256 lines    | Encoder incomplete                    | Re-run encoder; check disk space       |

---

## 10. Further Experiments

### Experiment A – SNR vs Distance
Place TX and RX antennas at increasing distances (0.5 m, 1 m, 2 m, 5 m). Plot `sync_tone_snr_dB` vs distance. Compare with free-space path loss model.

### Experiment B – Modulation Index Sweep
Change `FM_DEVIATION` in `pluto_transceiver.py` from 2 kHz to 20 kHz. Observe the effect on bandwidth (spectrogram) and image quality (SNR). This demonstrates Carson's rule empirically.

### Experiment C – Image Compression vs Quality
Before encoding, apply JPEG compression at Q = 10, 30, 70, 100. Compare encoded WAV duration (images compress differently) and decoded quality (SSIM, PSNR).

### Experiment D – Alternative SSTV Modes
Implement **Scottie S1** (slower, 320 × 256, sequential R/G/B) or **Robot 36** (colour with luminance/chrominance). Compare spectrograms and noise resilience.

### Experiment E – GNU Radio NBFM
Replace the wide-deviation FM with **NBFM** (Δf = 2.5 kHz). Measure how the narrower bandwidth changes the SSTV tone fidelity and decoded image quality.
