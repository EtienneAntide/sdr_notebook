## Useful Functions
import numpy as np
import matplotlib.pyplot as plt
import math
import sys, getopt
import random


##################### RAISED-COSINE FUNCTION ########################
def rcosfilter(N, beta, T_symb, Fs):
    """Raised Cosine filter. This filter has a 0 inter_symbols value.
    It is advised to use the function as in this example:
    t0 = K*T_symb
    rc = rcosfilter( N=int(2*K*T_symb*Fs) , beta=beta,  T_symb=T_symb, Fs=Fs )
    --> If K=3, the RRC filter then has a 6 sample length.
    It spans 3 samples to the left and 3 to the right.

    Args:
        N (int): Filter length in [samples]
        beta (float): Roll-off Factor in [0 ; 1]
        T_symb (float): Symbol Period in [sec] (1/Data_Rate)
        Fs (float): Sampling Frequency in [Hz]

    Returns:
        Array: Normalized RC filter of length N
    """
    time = (np.arange(N) - (N) / 2) / Fs
    imp_resp = np.zeros(N)

    time = (np.arange(N) - (N) / 2) / Fs
    imp_resp = np.zeros(N)

    if beta != 0:
        for index in range(0, N):
            t = time[index]
            if abs(t - T_symb / (2 * beta)) < 0.000000001:
                imp_resp[index] = (np.pi / 4 / T_symb) * (
                    np.sin(np.pi / 2 / beta) / (np.pi / 2 / beta)
                )
            elif abs(t + T_symb / (2 * beta)) < 0.000000001:
                imp_resp[index] = (np.pi / 4 / T_symb) * (
                    np.sin(np.pi / 2 / beta) / (np.pi / 2 / beta)
                )
            elif t == 0:
                imp_resp[index] = 1 / T_symb
            else:
                imp_resp[index] = (
                    (1 / T_symb)
                    * (np.sin(np.pi * t / T_symb) / (np.pi * t / T_symb))
                    * np.cos(np.pi * beta * t / T_symb)
                    / (1 - (2 * beta * t / T_symb) ** 2)
                )
    elif beta == 0:
        for index in range(0, N):
            t = time[index]
            if t == 0:
                imp_resp[index] = 1 / T_symb
            else:
                imp_resp[index] = (1 / T_symb) * (
                    np.sin(np.pi * t / T_symb) / (np.pi * t / T_symb)
                )

    return imp_resp / max(abs(imp_resp))


##################### ROOT RAISED-COSINE (RRC) FUNCTION #####################
def rrcosfilter(N, beta, T_symb, Fs):
    """Square-Root Raised Cosine filter. This filter is usually used both at the
    transmition stage to shape the data stream, and at the reception Matched Filter
    stage (MF) to result in a Raised-Cosine (RC) shape data stream at the MF output.

    Args:
        N (int): Filter length in [samples]
        beta (float): Roll-off factor in [0 ; 1]
        T_symb (float):Symbol Period in [sec] (1/Data_Rate)
        Fs (int): Sampling Frequency in [Hz]

    Returns:
        array: Normalized RRC filter of length N
    """
    time = (np.arange(N) - (N) / 2) / Fs
    imp_resp = np.zeros(N)
    for index in range(0, N):
        t = time[index]
        if beta != 0.0:
            if t == 0.0:
                imp_resp[index] = (
                    1 + beta * (4 / np.pi - 1)
                ) / T_symb  # remains true if beta==0

            elif abs(t - T_symb / (4 * beta)) < 0.000000001:
                imp_resp[index] = (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                ) * (beta / (T_symb * np.sqrt(2)))
            elif abs(t + T_symb / (4 * beta)) < 0.000000001:
                imp_resp[index] = (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                ) * (beta / (T_symb * np.sqrt(2)))

            elif abs(t) != T_symb / (4 * beta):
                imp_resp[index] = (
                    (
                        np.sin(np.pi * t * (1 - beta) / T_symb)
                        + 4
                        * beta
                        * t
                        / T_symb
                        * np.cos(np.pi * t * (1 + beta) / T_symb)
                    )
                    / (np.pi * t * (1 - (4 * beta * t / T_symb) ** 2) / T_symb)
                    / T_symb
                )
        else:
            if (t < T_symb / 2) and (t >= -T_symb / 2):
                imp_resp[index] = 1 / T_symb
    # imp_resp = imp_resp / max(abs(imp_resp))
    # normalization :
    imp_resp = imp_resp * T_symb

    return imp_resp


##################### rrcosfilter implicit FS=1, T_symb => N_symb #####################
##################### rrcosfilter implicit FS=1, T_symb => N_symb #####################
def rrcosfilterSamples(N, beta, samples_per_symbol):
    """Square-Root Raised Cosine filter. This filter is usually used both at the
    transmition stage to shape the data stream, and at the reception Matched Filter
    stage (MF) to result in a Raised-Cosine (RC) shape data stream at the MF output.
    * This version does not require the Sampling Frequency argument, but rather uses
    the oversampling factor SamplesPerSymbol.
    This function is Data rate / sampling rate agnostic, and thus requires one less
    argument.

    Args:
        N (int): Filter length in [samples]
        beta (float): Roll-off factor in [0 ; 1]
        samples_per_symbol (int): Number of samples ot represent per Symbol. This parameter
        corresponds to the "oversampling factor = Sampling_Frequency/Data_rate", or
        "oversampling factor = Symbol_period * Sampling_rate"

    Returns:
        array: Normalized RRC filter of length N
    """
    time = np.arange(N) - (N) / 2
    imp_resp = np.zeros(N)
    for index in range(0, N):
        t = time[index]
        if beta != 0.0:
            if t == 0.0:
                imp_resp[index] = (
                    1 + beta * (4 / np.pi - 1)
                ) / samples_per_symbol  # remains true if beta==0

            elif abs(t - samples_per_symbol / (4 * beta)) < 0.000000001:
                imp_resp[index] = (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                ) * (beta / (samples_per_symbol * np.sqrt(2)))
            elif abs(t + samples_per_symbol / (4 * beta)) < 0.000000001:
                imp_resp[index] = (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                ) * (beta / (samples_per_symbol * np.sqrt(2)))

            elif abs(t) != samples_per_symbol / (4 * beta):
                imp_resp[index] = (
                    (
                        np.sin(np.pi * t * (1 - beta) / samples_per_symbol)
                        + 4
                        * beta
                        * t
                        / samples_per_symbol
                        * np.cos(np.pi * t * (1 + beta) / samples_per_symbol)
                    )
                    / (
                        np.pi
                        * t
                        * (1 - (4 * beta * t / samples_per_symbol) ** 2)
                        / samples_per_symbol
                    )
                    / samples_per_symbol
                )
        elif abs(t) < samples_per_symbol / 2:
            imp_resp[index] = 1
    # imp_resp = imp_resp / max(abs(imp_resp))
    imp_resp = imp_resp * samples_per_symbol

    return imp_resp


################### GRAY CODE  #####################
def gray_code(n):
    """generates the gray code that corresponds to a given number of Bit per Symbols.
    In this data representation, there is always only one bit that differs from one
    symbol to the next/previous in a M-PSK constellation.

    Args:
        n (int): Number of bits per symbol. The function consequently generates (2**n) symbols

    Returns:
        symbols_list (array): strings that corresponds to the sorted gray symbols
        symbols_values (array): floats that corresponds to the values of the sorted gray symbols
    """

    symbols_values = []
    symbols_list = []
    for i in range(0, 1 << n):
        gray: int = i ^ (i >> 1)
        symbols_values = np.append(symbols_values, gray)
        symbols_list = np.append(symbols_list, bin(int(gray))[2:].zfill(n))

    return symbols_list, symbols_values


################### BARKER SEQUENCE ################
def barker(barker_len):
    """returns the barker sequence of the desired length. Barker sequences have good
    autocorrelation properties (maximum at delay = 0, low elsewhere). It is typically
    used as communication preamble.

    Args:
        barker_len (int): Number of symbols in the barker sequence to generate.
        Allowed values are { 2; 3; 4; 5; 7; 11; 13}

    Returns:
        array: Sequence of +1 and -1, of the desired length 'barker_len'.
    """
    if barker_len == 2:
        barker_code = np.array([-1, +1])
    elif barker_len == 3:
        barker_code = np.array([-1, -1, +1])
    elif barker_len == 4:
        barker_code = np.array([-1, -1, +1, -1])
    elif barker_len == 5:
        barker_code = np.array([-1, -1, -1, +1, -1])
    elif barker_len == 7:
        barker_code = np.array([-1, -1, -1, +1, +1, -1, +1])
    elif barker_len == 11:
        barker_code = np.array([-1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1])
    elif barker_len == 13:
        barker_code = np.array([-1, -1, -1, -1, -1, +1, +1, -1, -1, +1, -1, +1, -1])
    else:
        pass  # should return an error

    return barker_code


def correlation(signal, h):
    # Correlation function
    # s := array of N elements (signal)
    # h := array of K elements
    K = len(h)
    N = len(signal)
    h_conj = np.conjugate(h)
    signal_extended = np.zeros(len(signal) + K, dtype=complex)
    C_n = np.zeros(len(signal), dtype=complex)
    corr_k = np.zeros(K, dtype=complex)
    signal_extended[int(K / 2) - 1 : N + int(K / 2) - 1] = signal  # centered
    for n in range(N):
        # for n in range (len(signal)):
        for k in range(K):
            corr_k[k] = signal_extended[n + k] * h_conj[k]
        C_n[n] = np.sum(corr_k)
        # reset:
        corr_k = corr_k * 0
    return C_n


def correlation2(signal, h):
    K = len(h)
    N = len(signal)
    C_n = np.zeros(N + K - 1, dtype=complex)
    signal_extended = np.zeros(N + K - 1, dtype=complex)
    signal_extended[int(K / 2) - 1 : N + int(K / 2) - 1] = signal  # centered
    corr_k = np.zeros(K, dtype=complex)
    for n in range(len(C_n)):
        for k in range(K):
            if (n - k) < 0:
                a = 0
            elif (n - k) > N - 1:
                a = 0
            else:
                a = np.conjugate(signal[n - k])
            corr_k[k] = a * h[k]
        C_n[n] = np.sum(corr_k)
        # reset:
        corr_k = corr_k * 0
    return C_n


def normalized_correlation(signal, h):
    # Correlation function, normalized in energy
    # s := array of N elements (signal)
    # h := array of K elements
    K = len(h)
    N = len(signal)
    h_conj = np.conjugate(h)
    signal_extended = np.zeros(len(signal) + K)
    C_Norm = np.zeros(len(signal))
    C_n = np.zeros(len(signal))
    corr_k = np.zeros(K)
    energy_signal_n = 0.0
    energy_signal_k = np.zeros(K)
    energy_h = sum(abs(h_conj) ** 2)
    signal_extended[int(K / 2) - 1 : N + int(K / 2) - 1] = signal  # centered
    for n in range(N):
        # for n in range (len(signal)):
        for k in range(K):
            corr_k[k] = signal_extended[n + k] * h_conj[k]
            energy_signal_k[k] = (abs(signal_extended[n + k])) ** 2
        energy_signal_n = sum(energy_signal_k)
        C_n[n] = np.sum(corr_k)
        C_Norm[n] = np.abs(C_n[n]) / np.sqrt(energy_signal_n * energy_h)
        # reset:
        corr_k = corr_k * 0
        energy_signal_k = energy_signal_k * 0
    return C_Norm


def convolution(input_signal, filter):
    """Convolution between two input arrays x[n] and h[n].
    The function returns the same result y[n] if the two inputs are switched:
    y[n] = x[n]*h[n] = h[n]*x[n]

    Args:
        input_signal (array): input signal x[n], of size N
        filter (array): filter impulse response h[n], of size M

    Returns:
        y_n (array): convolution result, of size N+M-1.
    """
    len_signal = len(input_signal)
    len_filter = len(filter)

    N = max(len_signal, len_filter)
    M = min(len_signal, len_filter)

    if len_signal >= len_filter:
        x = input_signal
        h = filter
    else:
        h = input_signal
        x = filter

    y_n = []
    for n in range(N + M - 1):
        sum = 0
        for k in range(N):
            if (n - k) >= 0 and (n - k) < M:
                sum += x[k] * h[n - k]
        y_n = np.append(y_n, sum)

    return y_n


def convolution_optimized(input_signal, filter):
    """Convolution between two input arrays x[n] and h[n].
    The function returns the same result y[n] if the two inputs are switched:
    y[n] = x[n]*h[n] = h[n]*x[n]

    Args:
        input_signal (array): input signal x[n], of size N
        filter (array): filter impulse response h[n], of size M

    Returns:
        y_n (array): convolution result, of size N+M-1.
    """
    len_signal = len(input_signal)
    len_filter = len(filter)

    N = max(len_signal, len_filter)
    M = min(len_signal, len_filter)

    if len_signal >= len_filter:
        x = input_signal
        h = filter
    else:
        h = input_signal
        x = filter

    # Pre-allocate the result array with zeros
    y_n = np.zeros(N + M - 1)
    for n in range(N + M - 1):
        sum = 0
        for k in range(N):
            if (n - k) >= 0 and (n - k) < M:
                sum += x[k] * h[n - k]
        y_n[n] = sum

    return y_n


# def correlation_sample_per_sample(input_signal, filter):
#     """Convolution between two input arrays x[n] and h[n].
#     The function returns the same result y[n] if the two inputs are switched:
#     y[n] = x[n]*h[n] = h[n]*x[n]

#     Args:
#         input_signal (array): input signal x[n], of size N
#         filter (array): filter impulse response h[n], of size M

#     Returns:
#         y_n (array): convolution result, of size N+M-1.
#     """
#     len_signal = len(input_signal)
#     len_filter = len(filter)

#     K = len_signal
#     M = len_filter

#     x = input_signal
#     h = filter

#     y_n = []
#     for n in range(K + M - 1):
#         sum = 0
#         for k in range(K):
#             if (n - k) >= 0 and (n - k) < M:
#                 sum += np.conj(x[k]) * h[n + k]
#         y_n = np.append(y_n, sum)
#     return y_n


####
def _create_zadoffchu_code(nseqs, nsamples):
    """
    Generate complex zadoffchu sequences.
    """

    # if nsamples % 2 == 0:
    #     tools.print_err(
    #         "Zadoff-Chu sequence : length %d should be odd to generate orthogonal sequences"
    #         % (nsamples)
    #     )

    codebook = []

    U = []
    idx = np.array(range(nsamples))
    for n in range(nseqs):
        # get a new u prime to nsamples
        got_new_u = False
        ntries = 0
        while got_new_u == False:
            u = random.sample(range(1, nsamples - 1), 1)[0]
            if math.gcd(u, nsamples) == 1 and u not in U:
                got_new_u = True
                for v in U:
                    if math.gcd(u - v, nsamples) != 1:
                        got_new_u = False

            ntries += 1
            # if ntries > 1000:
            #     tools.print_err(
            #         "ZadoffChu sequence : could not find a new unique sequence (%dth)"
            #         % (n + 1)
            #     )

        U.append(u)

        # generate the sequence
        arr = np.exp(-1j * np.pi * u * idx * (idx + 1) / nsamples)
        scaling = np.sqrt(arr.real.var() + arr.imag.var())
        arr.real = arr.real / scaling
        arr.imag = arr.imag / scaling
        codebook.append(arr)

    return codebook


####### Function TESTS
def main(argv):
    # CONVOLUTION TEST #
    N = 48
    M = 32
    x_n = np.zeros(N)
    h_n = np.zeros(M)
    for m in range(16):
        h_n[m + 8] = m / M
    for n in range(32):
        x_n[n + 8] = 1

    y_n0 = np.convolve(x_n, h_n)
    y_n1 = convolution(x_n, h_n)
    y_n1_sym = convolution(h_n, x_n)

    plt.figure()
    plt.subplot(221)
    plt.title("x[n]")
    plt.plot(x_n, label="x[n]")
    plt.grid()
    plt.subplot(222)
    plt.title("h[n]")
    plt.plot(h_n, label="h[n]")
    plt.grid()
    plt.subplot(212)
    plt.title("convolutions result")
    plt.plot(y_n0, label="y[n] numpy")
    plt.plot(y_n1, lw=3, label="x[n]*h[n]")
    plt.plot(y_n1_sym, label="h[n]*x[n]")
    plt.xlim(0, len(y_n0))
    plt.legend()
    plt.grid()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    autocorr = correlation(x_n, x_n)
    plt.figure()
    plt.subplot(221)
    plt.title("x[n]")
    plt.plot(x_n, label="x[n]")
    plt.grid()
    plt.subplot(222)
    plt.title("h[n]")
    plt.plot(x_n, label="h[n]")
    plt.grid()
    plt.subplot(212)
    plt.title("autocorrelation")
    plt.plot(autocorr, label="autocorr")
    # plt.plot(y_n1, label="x[n]*h[n]")
    # plt.plot(y_n1_sym, label="h[n]*x[n]")
    plt.xlim(0, len(y_n0))
    plt.legend()
    plt.grid()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    Nzc = 139
    codebook = _create_zadoffchu_code(3, Nzc)
    zc_0 = codebook[0]
    zc_1 = codebook[1]
    zc_2 = codebook[2]

    plt.figure()
    plt.suptitle("Zadoff-Chu sequences")
    plt.subplot(311)
    plt.title("sequence zc0")
    plt.plot(zc_0.real, label="Re")
    plt.plot(zc_0.imag, label="Im")
    plt.legend()
    plt.grid()
    plt.subplot(312)
    plt.title("sequence zc1")
    plt.plot(zc_1.real, label="Re")
    plt.plot(zc_1.imag, label="Im")
    plt.grid()
    plt.subplot(313)
    plt.title("sequence zc2")
    plt.plot(zc_2.real, label="Re")
    plt.plot(zc_2.imag, label="Im")
    plt.grid()
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    plt.figure()
    plt.suptitle(r"3 Zadoff-Chu sequences compared MF")
    plt.subplot(311)
    plt.title(r"abs(MF output): $h[n]= zc_0[-n]*$")
    plt.plot(np.abs(correlation(zc_0, (zc_0))), label="|zc0[n]*h[n]|")
    plt.plot(np.abs(correlation(zc_1, (zc_0))), label="|zc1[n]*h[n]|")
    plt.plot(np.abs(correlation(zc_2, (zc_0))), label="|zc2[n]*h[n]|")
    plt.legend()
    plt.grid()
    plt.subplot(312)
    plt.title(r"abs(MF output): $h[n]= zc_1[-n]*$")
    plt.plot(np.abs(correlation(zc_0, (zc_1))).real, label="|zc0[n]*h[n]|")
    plt.plot(np.abs(correlation(zc_1, (zc_1))).real, label="|zc1[n]*h[n]|")
    plt.plot(np.abs(correlation(zc_2, (zc_1))).real, label="|zc2[n]*h[n]|")
    plt.grid()
    plt.legend()
    plt.subplot(313)
    plt.title(r"abs(MF output): $h[n]= zc_2[-n]*$")
    plt.plot(np.abs(correlation(zc_0, (zc_2))).real, label="|zc0[n]*h[n]|")
    plt.plot(np.abs(correlation(zc_1, (zc_2))).real, label="|zc1[n]*h[n]|")
    plt.plot(np.abs(correlation(zc_2, (zc_2))).real, label="|zc2[n]*h[n]|")
    plt.grid()
    plt.legend()
    # plt.subplot(212)
    # plt.plot(MF_00.imag, label="Im")
    # plt.plot(MF_01.imag, label="Im")
    # plt.plot(MF_02.imag, label="Im")
    # plt.grid()
    # plt.legend()
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()


####

if __name__ == "__main__":
    main(sys.argv[1:])
