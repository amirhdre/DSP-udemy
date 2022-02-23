#%% includes

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
from scipy import fftpack
import copy
import pylab as pl
import time
from IPython import display

#%%

## general simulation parameters

fs = 1024
npnts = fs * 5  # 5 seconds

# centered time vector
timevec = np.arange(0, npnts) / fs
timevec = timevec - np.mean(timevec)

# for power spectrum
hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))

#%%

## Morlet wavelet

# parameters
freq = 4  # peak frequency
csw = np.cos(2 * np.pi * freq * timevec)  # cosine wave
fwhm = 0.5  # full-width at half-maximum in seconds
gaussian = np.exp(-(4 * np.log(2) * timevec**2) / fwhm**2)  # Gaussian

# Morlet wavelet
MorletWavelet = csw * gaussian

# amplitude spectrum
MorletWaveletPow = np.abs(scipy.fftpack.fft(MorletWavelet) / npnts)


# time-domain plotting
plt.subplot(211)
plt.plot(timevec, MorletWavelet, "k")
plt.xlabel("Time (sec.)")
plt.title("Morlet wavelet in time domain")

# frequency-domain plotting
plt.subplot(212)
plt.plot(hz, MorletWaveletPow[: len(hz)], "k")
plt.xlim([0, freq * 3])
plt.xlabel("Frequency (Hz)")
plt.title("Morlet wavelet in frequency domain")
plt.tight_layout()
plt.show()

# %%

## Haar wavelet

# create Haar wavelet
HaarWavelet = np.zeros(npnts)
HaarWavelet[np.argmin(timevec**2) : np.argmin((timevec - 0.5) ** 2)] = 1
HaarWavelet[
    np.argmin((timevec - 0.5) ** 2) : np.argmin((timevec - 1 - 1 / fs) ** 2)
] = -1

# amplitude spectrum
HaarWaveletPow = np.abs(scipy.fftpack.fft(HaarWavelet) / npnts)

# time-domain plotting
plt.subplot(211)
plt.plot(timevec, HaarWavelet, "k")
plt.xlabel("Time (sec.)")
plt.title("Haar wavelet in time domain")

# frequency-domain plotting
plt.subplot(212)
plt.plot(hz, HaarWaveletPow[: len(hz)], "k")
plt.xlim([0, freq * 3])
plt.xlabel("Frequency (Hz)")
plt.title("Haar wavelet in frequency domain")
plt.tight_layout()
plt.show()

#%%

## Difference of Gaussians (DoG)
# (approximation of Laplacian of Gaussian)

# define sigmas
sPos = 0.1
sNeg = 0.5

# create the two GAussians
gaus1 = np.exp((-(timevec**2)) / (2 * sPos**2)) / (sPos * np.sqrt(2 * np.pi))
gaus2 = np.exp((-(timevec**2)) / (2 * sNeg**2)) / (sNeg * np.sqrt(2 * np.pi))

# their difference is the DoG
DoG = gaus1 - gaus2


# amplitude spectrum
DoGPow = np.abs(scipy.fftpack.fft(DoG) / npnts)

# time-domain plotting
plt.subplot(211)
plt.plot(timevec, DoG, "k")
plt.xlabel("Time (sec.)")
plt.title("DoG wavelet in time domain")

# frequency-domain plotting
plt.subplot(212)
plt.plot(hz, DoGPow[: len(hz)], "k")
plt.xlim([0, freq * 3])
plt.xlabel("Frequency (Hz)")
plt.title("DoG wavelet in frequency domain")
plt.tight_layout()
plt.show()

#%%

## general simulation parameters

fs = 1024
npnts = fs * 5  # 5 seconds

# centered time vector
timevec = np.arange(0, npnts) / fs
timevec = timevec - np.mean(timevec)

# for power spectrum
hz = np.linspace(0, fs / 2, int(np.floor(npnts / 2) + 1))

#%%

### create wavelets

# parameters
freq = 4  # peak frequency
csw = np.cos(2 * np.pi * freq * timevec)  # cosine wave
fwhm = 0.5  # full-width at half-maximum in seconds
gaussian = np.exp(-(4 * np.log(2) * timevec**2) / fwhm**2)  # Gaussian


## Morlet wavelet
MorletWavelet = csw * gaussian


## Haar wavelet
HaarWavelet = np.zeros(npnts)
HaarWavelet[np.argmin(timevec**2) : np.argmin((timevec - 0.5) ** 2)] = 1
HaarWavelet[
    np.argmin((timevec - 0.5) ** 2) : np.argmin((timevec - 1 - 1 / fs) ** 2)
] = -1


## Mexican hat wavelet
s = 0.4
MexicanWavelet = (
    (2 / (np.sqrt(3 * s) * np.pi**0.25))
    * (1 - (timevec**2) / (s**2))
    * np.exp((-(timevec**2)) / (2 * s**2))
)

#%%

## convolve with random signal

# signal
signal1 = scipy.signal.detrend(np.cumsum(np.random.randn(npnts)))

# convolve signal with different wavelets
morewav = np.convolve(signal1, MorletWavelet, "same")
haarwav = np.convolve(signal1, HaarWavelet, "same")
mexiwav = np.convolve(signal1, MexicanWavelet, "same")

# amplitude spectra
morewaveAmp = np.abs(scipy.fftpack.fft(morewav) / npnts)
haarwaveAmp = np.abs(scipy.fftpack.fft(haarwav) / npnts)
mexiwaveAmp = np.abs(scipy.fftpack.fft(mexiwav) / npnts)

### plotting
# the signal
plt.plot(timevec, signal1, "k")
plt.title("Signal")
plt.xlabel("Time (s)")
plt.show()

# the convolved signals
plt.subplot(211)
plt.plot(timevec, morewav, label="Morlet")
plt.plot(timevec, haarwav, label="Haar")
plt.plot(timevec, mexiwav, label="Mexican")
plt.title("Time domain")
plt.legend()

# spectra of convolved signals
plt.subplot(212)
plt.plot(hz, morewaveAmp[: len(hz)], label="Morlet")
plt.plot(hz, haarwaveAmp[: len(hz)], label="Haar")
plt.plot(hz, mexiwaveAmp[: len(hz)], label="Mexican")
plt.yscale("log")
plt.xlim([0, 40])
plt.legend()
plt.xlabel("Frequency (Hz.)")
plt.show()

#%% Wavelet convolution for narrowband filtering

# simulation parameters
srate = 4352  # hz
npnts = 8425
time = np.arange(0, npnts) / srate
hz = np.linspace(0, srate / 2, int(np.floor(npnts / 2) + 1))

# pure noise signal
signal1 = np.exp(0.5 * np.random.randn(npnts))

# let's see what it looks like
plt.subplot(211)
plt.plot(time, signal1, "k")
plt.xlabel("Time (s)")

# in the frequency domain
signalX = 2 * np.abs(scipy.fftpack.fft(signal1))
plt.subplot(212)
plt.plot(hz, signalX[: len(hz)], "k")
plt.xlim([1, srate / 6])
plt.ylim([0, 300])
plt.xlabel("Frequency (Hz)")
plt.show()
