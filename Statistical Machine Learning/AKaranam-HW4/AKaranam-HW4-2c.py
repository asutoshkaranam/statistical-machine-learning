import numpy as np
import matplotlib.pyplot as plt

x_axis = np.linspace(-10,10,500)
y_axis = np.linspace(-10,10,500)
x,y = np.meshgrid(x_axis, y_axis)
plt.contourf(x, y, ((np.abs(x)**(1/3))+(np.abs(y)**(1/3)))**3, levels=[0, 8], colors=['lightgrey'])
plt.contour(x, y, ((np.abs(x)**(1/3))+(np.abs(y)**(1/3)))**3, levels=[8])
plt.axis('scaled')
plt.title('Set {(x₁, x₂) : g(x₁, x₂) ≤ 8}\nwhere g(x₁, x₂) = (|x₁|¹/³ + |x₂|¹/³)³')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
