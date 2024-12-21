import numpy as np
import matplotlib.pyplot as plt

def g1(x):
    return np.sin(2 * np.pi * x)

def g2(x):
    return np.sin(8 * np.pi * x)

def ghat(x, bin_1, bin_2):
    return np.where(x < 0.5, bin_1, bin_2)

def mean_squared_bias(g_original, expectation_g_hat, n):
    _x = np.linspace(0, 1, n)
    bias_2 = (expectation_g_hat(_x) - g_original(_x))**2
    return np.mean(bias_2)

n = 256
m = 128

x = np.linspace(0, 1, n)

#0.5 as the number of bins can only be 2
c1_bin1 = np.mean(g1(np.linspace(0, 0.5, n//2)))
c2_bin1 = np.mean(g1(np.linspace(0.5, 1, n//2)))

c1_bin2 = np.mean(g2(np.linspace(0, 0.5, n//2)))
c2_bin2 = np.mean(g2(np.linspace(0.5, 1, n//2)))

mean_squared_bias_bin1 = mean_squared_bias(g1, lambda x: ghat(x, c1_bin1, c2_bin1), n//2)
mean_squared_bias_bin2 = mean_squared_bias(g2, lambda x: ghat(x, c1_bin2, c2_bin2), n//2)

_, (plot_1, plot_2) = plt.subplots(2, 1, figsize=(10, 12))

plot_1.plot(x, g1(x), label='g1(x) = sin(2πx)', color='blue')
plot_1.plot(x, ghat(x, c1_bin1, c2_bin1), label='g-hat 128(x) for g1', color='red', linestyle='--')
plot_1.set_title(f'g1(x) and its estimator g-hat 128(x)\nAverage Bias-Squared: {mean_squared_bias_bin1}')
plot_1.set_xlabel('x')
plot_1.set_ylabel('y')
plot_1.legend()

plot_2.plot(x, g2(x), label='g2(x) = sin(8πx)', color='green')
plot_2.plot(x, ghat(x, c1_bin2, c2_bin2), label='g-hat 128(x) for g2', color='red', linestyle='--')
plot_2.set_title(f'g2(x) and its estimator g-hat 128(x)\nAverage Bias-Squared: {mean_squared_bias_bin2}')
plot_2.set_xlabel('x')
plot_2.set_ylabel('y')
plot_2.legend()

plt.show()

print(f"Mean squared bias for g1: {mean_squared_bias_bin1}")
print(f"Mean squared bias for g2: {mean_squared_bias_bin2}")
