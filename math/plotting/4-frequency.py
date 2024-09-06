#!/usr/bin/env python3
"""
Complete the following source code to plot a histogram of student scores for a project:

The x-axis should be labeled Grades
The y-axis should be labeled Number of Students
The x-axis should have bins every 10 units
The title should be Project A
The bars should be outlined in black
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
max_max = max(student_grades)

# your code here
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.hist(student_grades, bins=np.arange(0, max_max, 10), edgecolor="black")
plt.ylim(0, 30)
plt.yticks(np.arange(0, 31, 5))
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, 10))
plt.show()