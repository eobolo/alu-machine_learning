#!/usr/bin/env python3
"""
Complete the following source code to plot y as a line graph:

y should be plotted as a solid red line
The x-axis should range from 0 to 10
"""
import numpy as np
import matplotlib.pyplot as plt
y = np.arange(0, 11) ** 3
plt.plot(y, color="red")
plt.ylim(-50, 1050)
plt.yticks(np.arange(0, 1020, 200))
plt.xlim(0, 10)
plt.xticks(np.arange(0, 11, 2))
plt.show()