
import numpy as np

# Dataset
data = [10, 12, 14, 15, 16, 18, 20, 22, 100]

# Calculate Q1 and Q3
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate boundaries
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Detect outliers
outliers = []

for value in data:
    if value < lower or value > upper:
        outliers.append(value)

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Boundary:", lower)
print("Upper Boundary:", upper)
print("Outliers:", outliers)
