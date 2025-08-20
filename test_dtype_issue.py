import numpy as np

# Integer patterns
patterns_int = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],
])

# Float patterns  
patterns_float = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
])

print("Integer patterns:")
print(f"dtype: {patterns_int.dtype}")
print(f"Mean: {patterns_int[0].mean()}")
print(f"Std: {patterns_int[0].std(ddof=1)}")

# Test variance calculation with integers
var_int = 0.0
mean_int = sum(patterns_int[0]) / len(patterns_int[0])
print(f"Manual mean: {mean_int}")
for x in patterns_int[0]:
    var_int += (x - mean_int) ** 2
print(f"Manual var: {var_int}")
print(f"Manual std: {np.sqrt(var_int / 3)}")

print("\nFloat patterns:")
print(f"dtype: {patterns_float.dtype}")
print(f"Mean: {patterns_float[0].mean()}")
print(f"Std: {patterns_float[0].std(ddof=1)}")