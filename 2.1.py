import matplotlib.pyplot as plt
import numpy as np
import random
import cmath
import math
import time




def getSignals(n, N, W):
    generated_signal = np.zeros(N)
    start = time.time()
    for i in range(n):
        fi = 2 * math.pi * random.random()
        A = 5 * random.random()
        w = W - i * W / (n)
        x = A * np.sin(np.arange(0, N, 1) * w + fi)
        generated_signal += x
    print(f"Execution time: {time.time() - start}")
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
    print(f"Execution time: {time.time() - start}")
    return spectre

signal = getSignals(6, 256, 2100)

plot = draw(signal, "t", "x(t)", "Signal", "X(t)", "blue")
plt.grid()
plt.show()
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