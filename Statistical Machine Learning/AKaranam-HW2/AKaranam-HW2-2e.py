import numpy as np
import matplotlib.pyplot as plt

n = 256
variance = 0.5
given_m_list = [1, 2, 4, 8, 16, 32]

def g(x):
    return 2 * np.sin(10 * np.pi * x + 3) * np.exp((x - 1.5)**3)

np.random.seed(42)
points = np.linspace(1/n, 1, n)
y_of_points = g(points) + np.random.normal(0, np.sqrt(variance), n)

empirical_arr = []
mean_square_bias_arr = []
variance_arr = []
net_errors_arr = []

for m in given_m_list:
    emp_error = 0
    bias_2 = 0

    for j in range(1, n//m + 1):
        boundary_begin = (j-1)*m
        boundary_end = j*m

        expectation_g_hat = np.mean(y_of_points[boundary_begin:boundary_end])

        g_true_hat = np.mean(g(points[boundary_begin:boundary_end]))

        emp_error += np.sum((expectation_g_hat - g(points[boundary_begin:boundary_end]))**2)
        bias_2 += np.sum((g_true_hat - g(points[boundary_begin:boundary_end]))**2)

    average_emp_error = emp_error / n
    average_variance = variance / m
    average_bias_2 = bias_2 / n

    empirical_arr.append(average_emp_error)
    variance_arr.append(average_variance)
    mean_square_bias_arr.append(average_bias_2)
    net_errors_arr.append(average_variance + average_bias_2)

plt.figure(figsize=(12, 6))
plt.plot(given_m_list, mean_square_bias_arr, 'ro--', label='Average Bias-Squared')
plt.plot(given_m_list, empirical_arr, 'bo--', label='Average Empirical Error')
plt.plot(given_m_list, variance_arr, 'ko--', label='Average Variance')
plt.plot(given_m_list, net_errors_arr, 'go--', label='Average Total Error')
#plt.xscale('log')
plt.xlabel('m (n/m is the number of bins)')
plt.ylabel('Error Values')
plt.title('Comparing different values for m')
plt.legend()
plt.show()

# Find the best m
best_m = given_m_list[np.argmin(net_errors_arr)]
print(f"The best choice of m is: {best_m}")
