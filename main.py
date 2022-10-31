import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk, butter, buttord, iirnotch, filtfilt
from pathlib import Path
import os.path

def DFT(x):
    """Computes DFT of the given array.
    Parameters:
    x : ndarray
        Numpy array of the values for computing DFT.
    Returns:
    dft : ndarray
        Numpy array of the DFT computed values.
    """
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    dft = np.dot(e, x)
    return dft

def plot_dfts(dft,dft2,samples,fs):
    """Plots the comparison of two computed DFTs.
    Parameters:
    dft : nd.array
        Numpy array of the values of the first DFT.
    dft2 : nd.array
        Numpy array of the values of the second DFT.
    samples : int
        Number of samples.
    fs : int
        Fs of the signal.
    """   
    f = np.arange(dft.size) / samples * fs
    plt.figure(figsize=(9,9))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212)
    ax1.plot(f[:f.size//2+1], dft[:dft.size//2+1])
    ax1.set_title('DFT - own implementation')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('|DFT|')
    ax2.plot(f[:f.size//2+1], dft2[:dft2.size//2+1])
    ax2.set_title('DFT - Numpy')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('|DFT|')
    ax3.plot(f[:f.size//2+1], dft[:dft.size//2+1],label='own implementation')
    ax3.plot(f[:f.size//2+1],dft2[:dft2.size//2+1],label='Numpy')
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('|DFT|')
    ax3.legend(loc='upper right',title='implementation')
    # plt.show()

def plot_cosine_spectrogram(cos,fs,samples,overlap,t=None,f=None,sgr_log=None):
    """Plots the spectrogram of the cosines.
    Parameters:
    cos : nd.array
        Numpy array of the cosines signal.
    fs : int
        Fs of the signal.
    samples : int
        Number of samples of the frame.
    overlap : int
        Size of the overlap.
    t : nd.array optional
        Array of times.
        If not specified together with f and sgr_log, the n the comparison is not
        shown and is plotted only spectrogram of cosines signal.
    f : nd.array optional
        Array of frequencies.
        If not specified together with t and sgr_log, the n the comparison is not
        shown and is plotted only spectrogram of cosines signal.
    sgr_log : nd.array optional
        Computed spectrogram values converted to logaritmic values.
        If not specified together with t and fs, the n the comparison is not
        shown and is plotted only spectrogram of cosines signal.
    """
    if (t is None) | (f is None) | (sgr_log is None):   
        f2, t2, sgr2 = spectrogram(x=cos,fs=fs,nperseg=samples,noverlap=overlap)
        sgr_log2 = 10 * np.log10(sgr2)
        plt.figure(figsize=(6,6))
        plt.pcolormesh(t2,f2,sgr_log)
        plt.title('Spectrogram - Cosines')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        # plt.show()
    else:
        f2, t2, sgr2 = spectrogram(x=cos,fs=fs,nperseg=samples,noverlap=overlap)
        sgr_log2 = 10 * np.log10(sgr2)
        fig, ax = plt.subplots(1,2,figsize=(9,6))
        ax[0].pcolormesh(t,f,sgr_log)
        ax[0].set_title('Spectrogram - Signal')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Frequency [Hz]')
        ax[1].pcolormesh(t2,f2,sgr_log2)
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Frequency [Hz]')
        ax[1].set_title('Spectrogram - Cosines')
        plt.tight_layout()
        #plt.show()

def create_filters(frequencies, rp, rs, fs):
    """Returns the b and a coefficients of the filters which filter the given frequencies.
    Parameters:
    frequencies : list
        List of the frequencies.
    rp : int
        Ripple.
    rs : int
        Stop-band attenuation.
    fs : int
        Fs of the signal.
    Returns:
    b,a : tupple of ndarray
        Coefficints b and a of the filters.
    """   
    nyq = fs*0.5
    a = list()
    b = list()
    for i in range(len(frequencies)):
        Wp = [(frequencies[i]-30)/nyq, (frequencies[i]+30)/nyq]
        Ws = [(frequencies[i]-50)/nyq, (frequencies[i]+50)/nyq]
        Rp = rp
        Rs = rs
        N, Wn = buttord(Wp, Ws, Rp, Rs)
        low_high = [(frequencies[i]-50)/nyq, (frequencies[i]+50)/nyq]
        b1,a1 = butter(N=5,Wn=low_high,btype='bandstop',output='ba')
        b.append(b1)
        a.append(a1)
        #signal = lfilter(b[i],a[i],signal)
    return b,a

def plot_ir(a,b,N_imp):
    """Plots the impulse response of the 4 filters.
    Parameters:
    a : list
        List of the coefficients a of the filters.
    b : list
        List of the coefficients b of the filters.
    N_imp : int
        Number of the samples of the impulse response.
    """
    imp = [1, *np.zeros((N_imp-1))]
    fig, ax = plt.subplots(2,2,figsize=(9,9))
    idx = 0
    for i in range(2):
        for j in range(2):
            print(f'Filter {idx+1}:\n b = {b[idx]}\n a = {a[idx]}')
            h = lfilter(b[idx], a[idx], imp)
            ax[i,j].stem(np.arange(N_imp), h, basefmt=' ', markerfmt=" ")
            ax[i,j].set_xlabel('$n$', loc='right')
            ax[i,j].set_title(f'Filter {idx+1} - Impulse response $h[n]$')

            idx += 1


def plot_zp(z,p):
    """Plots the zeros and poles.
    Parameters:
    z : list
        List of the zeros for filters.
    p : list
        List of the poles of the filters.
    """
    ang = np.linspace(0, 2*np.pi,100)
    idx = 0

    _, ax = plt.subplots(2,2,figsize=(11,11))
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(np.cos(ang), np.sin(ang))
            ax[i,j].scatter(np.real(z[idx]), np.imag(z[idx]), marker='o', facecolors='none', edgecolors='r', label='zeros')
            ax[i,j].scatter(np.real(p[idx]), np.imag(p[idx]), marker='x', color='g', label='poles')
            ax[i,j].set_title(f'Filter {idx+1} - Zeros, poles')
            ax[i,j].set_xlabel('Re',loc='right')
            ax[i,j].set_ylabel('Im')
            ax[i,j].legend(loc='lower left')

            is_stable = (p[idx].size == 0) or np.all(np.abs(p[idx]) < 1) 
            print(f'Filter {idx+1} is stable: {is_stable}')
            idx += 1
    #plt.show()

def plot_freq(w,H,fs):
    """Plots the frequency response.
    Parameters:
    w : list
        List of frequencies for each filter on which the frequency response was computed.
    H : list
        List of results of the frequency response for each filter.
    """
    for i in range(4):
        fig, ax = plt.subplots(2,1,figsize=(9,8))
        ax[0].set_title(f'Filter {i+1}')
        ax[0].plot(w[i] / 2 / np.pi * fs, np.abs(H[i]))
        ax[0].set_xlabel('Frequency [Hz]', loc='right')
        ax[0].set_ylabel('$|H(e^{j\omega})|$')

        ax[1].plot(w[i] / 2 / np.pi * fs, np.angle(H[i]))
        ax[1].set_xlabel('Frequency [Hz]', loc='right')
        ax[1].set_ylabel('$\mathrm{arg}\ H(e^{j\omega})$')
    #plt.show()

def divide_to_frames(samples,overlap,signal):
    """Divides the signal to frames of given number of samples with the overlap.
    Parameters:
    samples : int
        Number of samples of each frame.
    overlap : int
        Size of the overlap
    signal : ndarray
        Values of the signal.
    Returns:
    signal_div : ndarray
        Matrix of the signal divided into frames with each frame in the columns.
    """
    cols = int(round(len(signal)/overlap))
    signal_div = np.zeros((samples,cols))
    for i in range(cols-1):
        signal_div[:,i] = signal[overlap*i:overlap*i+samples]
    last_frame = signal[overlap*(cols-1):overlap*(cols-1)+samples]
    remaining = len(last_frame)
    signal_div[0:remaining,cols-1] = last_frame
    
    return signal_div 


def main():
    #===================================================================================
    # 4.1
    # loading the signal
    path_audio=Path(os.getcwd())
    path_audio=path_audio.parents[0]
    path_audio=path_audio.cwd() / 'audio'

    input_signal_path = path_audio / 'test.wav'
    signal, fs = sf.read(input_signal_path)
    # determining the lenght in seconds and samples
    # determinig the min and max value
    len_samples = len(signal)
    len_seconds = len_samples/fs
    min_val = signal.min()
    max_val = signal.max()
    # printing the values
    print(f'Length of the signal in seconds: {len_seconds} s.')
    print(f'Length of the signal in samples: {len_samples} samples.')
    print(f'Min value: {min_val}. Max value: {max_val}.')

    # plotting the signal
    t = np.arange(signal.size)/fs # creating the vector of time values
    plt.figure()
    plt.plot(t,signal)
    plt.title('Signal')
    plt.xlabel('Time [s]')
    # plt.show()

    #===================================================================================
    # 4.2
    # substracting mean value, dividing signal to be in range -1 and 1
    signal = signal-signal.mean()
    abs_val = max(abs(signal))
    signal = signal/abs_val

    samples = 1024
    overlap = 512

    # dividing to frames
    signal_div = divide_to_frames(samples=samples,overlap=overlap,signal=signal)

    # # show each frame to choose "nice one"
    # for i in range(signal_div.shape[1]):
    #     y = signal_div[:, i]
    #     fig, ax = plt.subplots(figsize=(12,3))
    #     plt.plot(y)
    #     plt.show()

    # first frame seems "nice", show it
    t1 = np.arange(samples)/fs
    seg = signal_div[:,0]
    plt.figure()
    plt.plot(t1,seg)
    plt.title('Nice frame')
    plt.xlabel('Time [s]')
    # plt.show()

    #===================================================================================
    # 4.3
    # own implementation
    dft = abs(DFT(seg))
    # Numpy
    dft2 = abs(np.fft.fft(seg))

    plot_dfts(dft=dft,dft2=dft2,samples=samples,fs=fs)
    # print whether the results are similar
    similar = np.allclose(dft,dft2)
    print(f'Result of own implementation and Numpy fft similar: {similar}')

    #===================================================================================
    # 4.4
    # show spectrogram
    f, t, sgr = spectrogram(x=signal,fs=fs,nperseg=samples,noverlap=overlap)
    sgr_log = 10 * np.log10(sgr) 
    plt.figure(figsize=(6,6))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_title('Spectrogram')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Frequency [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Power spectral density [dB]', rotation=270, labelpad=15)
    plt.tight_layout()
    # plt.show()

    #===================================================================================
    # 4.5
    # "manually" determining the frequencies from the spectrogram
    f1 = 860
    f2 = 1720
    f3 = 2580
    f4 = 3440

    frequencies = [f1,f2,f3,f4]
    #===================================================================================
    # 4.6
    # creating the signal with cosines of determined frequencies
    t5 = np.arange(signal.size)/fs
    cosine = np.cos(2*np.pi*f1*t5) + np.cos(2*np.pi*f2*t5) + np.cos(2*np.pi*f3*t5) + np.cos(2*np.pi*f4*t5)
    max_val = max(abs(cosine))
    cosine = cosine/max_val
    plot_cosine_spectrogram(cos=cosine,fs=fs,samples=samples,overlap=overlap,t=t,f=f,sgr_log=sgr_log)
    cos4_path = path_audio / '4cos.wav'
    sf.write(file=cos4_path,data=cosine,samplerate=fs)

    #===================================================================================
    # 4.7
    Rp = 3
    Rs = 40
    Q = 60

    # creating filters
    b,a = create_filters(frequencies=frequencies,rp=Rp,rs=Rs,fs=fs)
    
    # plotting the ir response
    N_imp = 60
    plot_ir(a,b,N_imp)
    #===================================================================================
    # 4.8
    # computing zeros and poles
    z = list()
    p = list()
    for i in range(len(b)):   
        z_act, p_act, k_act = tf2zpk(b[i], a[i])
        z.append(z_act)
        p.append(p_act)

    # plotting them
    plot_zp(z,p)
    #===================================================================================
    # 4.9
    w = list()
    H = list()
    for i in range(len(b)):   
        w_act, H_act = freqz(b=b[i],a=a[i])
        w.append(w_act)
        H.append(H_act)
    
    # plotting the frequence response
    plot_freq(w,H,fs)
    #===================================================================================
    # 4.10
    # filtering signal
    for i in range(4):
        signal = lfilter(b=b[i],a=a[i],x=signal)
        #signal = filtfilt(b=b[i],a=a[i],x=signal)
    
    # if the signal is not in range -1 and 1, make it so and save the sound file
    abs_val = max(abs(signal))   
    if abs_val != 1.0:
        signal = signal/abs_val
    clean_path = path_audio / 'clean_bandstop.wav'
    sf.write(file=clean_path,data=signal,samplerate=fs)
    
    # comment not to show the graphs
    f, t, sgr = spectrogram(x=signal,fs=fs,nperseg=samples,noverlap=overlap)
    sgr_log = 10 * np.log10(sgr) 
    plt.figure(figsize=(6,6))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_title('Spectrogram')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('Frequency [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Power spectral density [dB]', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()