#!/usr/bin/env python3
"""
Complete the following source code to plot all 5 previous graphs in one figure:

All axis labels and plot titles should have a font size of x-small (to fit nicely in one figure)
The plots should make a 3 x 2 grid
The last plot should take up two column widths (see below)
The title of the figure should be All in One
"""
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here
plt.figure(figsize=(15, 11))
plt.subplots_adjust(wspace=0.8, hspace=0.8)
plt.suptitle("All in One")

# Question 1
ax = plt.subplot2grid((3, 2), (0, 0), 1, 1)
ax.plot(y0, color="red")
ax.set_ylim(-50, 1100)
ax.set_yticks(np.arange(0, 1020, 500))
ax.set_xlim(0, 10)
ax.set_xticks(np.arange(0, 11, 2))


# Question 2
ax = plt.subplot2grid((3, 2), (0,1), 1, 1)
ax.scatter(x1, y1, c="magenta", s=6)
ax.set_title("Men's Height vs Weight")
ax.set_xlabel("Height (in)")
ax.set_ylabel("Weight (lbs)")
ax.set_ylim(160, 195)
ax.set_yticks(np.arange(170, 191, 10))
ax.set_xlim(55, 85)
ax.set_xticks(np.arange(60, 90, 10))

# Question 3
ax = plt.subplot2grid((3, 2), (1, 0), 1, 1)
ax.set_title("Exponential Decay of C-14")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Exponential Decay of C-14")
ax.semilogy(x2, y2)
ax.set_xlim(0, 28650)
ax.set_xticks(np.arange(0, 28651, 10000))

# Question 4
ax = plt.subplot2grid((3, 2), (1, 1), 1, 1)
ax.set_title("Exponential Decay of Radioactive Elements")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Fraction Remaining")
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.5, 0.5))
ax.set_xlim(0, 20000)
ax.set_xticks(np.arange(0, 20500, 2500))
ax.plot(x3, y31, "--r", x3, y32, "-g")
ax.legend(["C-14", "Ra-226"], loc=1)

# Question 5
max_max = max(student_grades)

# your code here
ax = plt.subplot2grid((3, 2), (2, 0), 1, 2)
ax.set_title("Project A")
ax.set_xlabel("Grades")
ax.set_ylabel("Number of Students")
ax.hist(student_grades, bins=np.arange(0, max_max, 10), edgecolor="black")
ax.set_ylim(0, 30)
ax.set_yticks(np.arange(0, 31, 10))
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 10))