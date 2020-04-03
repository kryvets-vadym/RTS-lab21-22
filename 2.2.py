import matplotlib.pyplot as plt
import numpy as np
import random
import cmath
import math
import time




def getValues(n, N, W):
    generated_signal = np.zeros(N)
    start = time.time()
    for i in range(n):
        fi = 2 * math.pi * random.random()
        A = 5 * random.random()
        w = W - i * W / (n)
        x = A * np.sin(np.arange(0, N, 1) * w + fi)
        generated_signal += x
    # print(f"Execution time: {time.time() - start}")
    return generated_signal

def draw(arr, x_label, y_label, title, legend, file_name=None):
    result, = plt.plot(range(len(arr)), arr, '-', label=legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    return result

def dft(signal):
    start = time.time()
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N):
        spectre[p] = np.dot(signal, np.cos(2 * math.pi * p / N * np.linspace(0, N-1, N))) \
                     -1j * np.dot(signal, np.sin(2 * math.pi * p / N * np.linspace(0, N-1, N)))
    print(f"Execution DFT time: {time.time() - start}")
    return spectre

signal = getValues(6, 256, 2048)

spectr = dft(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "Polar Spectr", "Amplitude")
plt.legend(handles=[ampl], loc='upper right')
plt.grid()
plt.show()
phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "Polar Spectr", "Phase")
plt.legend(handles=[phase], loc='upper right')
plt.grid()
plt.show()


def fft(signal):
    start = time.time()
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N // 2):
        E_m = np.dot(signal[0:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[0:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        W_p = (np.cos(2 * math.pi * p / N) - 1j * np.sin(2 * math.pi * p / N))
        O_m = np.dot(signal[1:N:2], np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))) - 1j * np.dot(signal[1:N:2],
              np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1)))
        spectre[p] = E_m + W_p * O_m
        spectre[p + N // 2] = E_m - W_p * O_m
    print(f"Execution FFT time: {time.time() - start}")
    return spectre

spectr = fft(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "Polar Spectr", "Amplitude")
plt.legend(handles=[ampl], loc='upper right')
plt.grid()
plt.show()
phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "Polar Spectr", "Phase")
plt.legend(handles=[phase], loc='upper right')
plt.grid()
plt.show()


# Additional task (Table method)
def make_table(signal):
    start = time.time()
    N = len(signal)
    table = np.cos(2 * math.pi / N * np.linspace(0, N-1, N)) \
            -1j * np.sin(2 * math.pi / N * np.linspace(0, N-1, N))
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N):
        indicies = np.linspace(0, N-1, N, dtype=np.int32) * p % N
        spectre[p] = np.dot(signal, table[indicies])
    print(f"Execution MakeTable time: {time.time() - start}")
    return spectre

spectr = make_table(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "Polar Spectr", "Amplitude")
plt.legend(handles=[ampl], loc='upper right')
plt.grid()
plt.show()
phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "Polar Spectr", "Phase")
plt.legend(handles=[phase], loc='upper right')
plt.grid()
plt.show()

