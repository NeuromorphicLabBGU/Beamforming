import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


def load_pdm_file(filename):
    # Assuming the PDM file contains raw binary data
    with open(filename, 'rb') as f:
        pdm_data = np.fromfile(f, dtype=np.uint8)
    return pdm_data


def pdm_to_pcm(pdm_data):
    # Convert PDM to PCM
    pcm_data = (pdm_data.astype(np.int16) - 127) * 256
    return pcm_data


def plot_spectrogram(signal_data, sampling_rate):
    plt.figure(figsize=(10, 4))
    f, t, Sxx = signal.spectrogram(signal_data, fs=sampling_rate)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pdm_filename = "pdm1b.pdm"
    sampling_rate = 44100  # Change this to your sampling rate

    # Load PDM data
    pdm_data = load_pdm_file(pdm_filename)

    # Convert PDM to PCM
    pcm_data = pdm_to_pcm(pdm_data)

    # Apply bandpass filter (1 kHz - 13 kHz)
    nyquist = 0.5 * sampling_rate
    low = 1000 / nyquist
    high = 13000 / nyquist
    b, a = signal.butter(6, [low, high], btype='band')
    filtered_pcm_data = signal.filtfilt(b, a, pcm_data)

    # Plot spectrogram
    plot_spectrogram(filtered_pcm_data, sampling_rate)
