"""
Visualize how different time series types are classified by the detection system.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Add the project root to Python path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driada.information.time_series_types import analyze_time_series_type

# Set up the figure
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
fig.suptitle("Time Series Type Detection Examples", fontsize=16)

# Generate different types of time series
np.random.seed(42)

# Row 1: Discrete types
# Binary
data_binary = np.random.choice([0, 1], size=100, p=[0.4, 0.6])
ax = axes[0, 0]
ax.plot(data_binary, "o-", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_binary)
ax.set_title(
    f"Binary Data\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.set_ylim(-0.5, 1.5)
ax.grid(True, alpha=0.3)

# Categorical
data_categorical = np.random.choice(["A", "B", "C", "D"], size=100)
# Convert to numeric for plotting
cat_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
data_cat_numeric = np.array([cat_to_num[x] for x in data_categorical])
ax = axes[0, 1]
ax.plot(data_cat_numeric, "o-", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_cat_numeric)
ax.set_title(
    f"Categorical Data (4 categories)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.set_ylim(-0.5, 3.5)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["A", "B", "C", "D"])
ax.grid(True, alpha=0.3)

# Count (monotonic)
data_count = np.cumsum(np.random.poisson(2, 50))
ax = axes[0, 2]
ax.plot(data_count, "o-", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_count)
ax.set_title(
    f"Count Data (cumulative)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Row 2: Timeline and ambiguous
# Integer timeline (non-monotonic, like hour of day observations)
# Create timeline data that represents observations at regular time intervals
# but not in monotonic order (like if we shuffle daily observations)
hours = np.arange(0, 24, 2)  # Every 2 hours: 0, 2, 4, ..., 22
# Create multiple days of observations in mixed order
data_timeline_int = np.concatenate([hours for _ in range(5)])  # 5 days
# Add a bit of shuffling to simulate non-sequential data collection
for i in range(0, len(data_timeline_int), 12):  # Shuffle within each day
    chunk = data_timeline_int[i : i + 12]
    np.random.shuffle(chunk)
    data_timeline_int[i : i + 12] = chunk
ax = axes[1, 0]
ax.plot(data_timeline_int, "o", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_timeline_int)
ax.set_title(
    f"Hour Timeline (0, 2, 4, ..., 22)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Non-integer timeline (with some repeated timestamps)
# Create a timeline where some timestamps appear multiple times (like multiple events at same time)
timestamps = []
for t in np.arange(0, 25, 0.5):
    # Some timestamps appear 1-3 times
    count = np.random.choice([1, 1, 2, 3], p=[0.7, 0.15, 0.1, 0.05])
    timestamps.extend([t] * count)
data_timeline_float = np.array(timestamps)
ax = axes[1, 1]
ax.plot(data_timeline_float, "o", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_timeline_float)
ax.set_title(
    f"Non-integer Timeline (0, 0.5, 1, ...)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Ambiguous discrete (many unique integers)
data_ambiguous = np.random.randint(0, 50, 100)
ax = axes[1, 2]
ax.plot(data_ambiguous, "o-", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_ambiguous)
ax.set_title(
    f"Ambiguous Discrete (many categories)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Row 3: Continuous types
# Linear continuous
data_linear = np.random.normal(50, 15, 100)
ax = axes[2, 0]
ax.plot(data_linear, "-", alpha=0.7)
result = analyze_time_series_type(data_linear)
ax.set_title(
    f"Linear Continuous (Gaussian)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Circular (angles in radians)
data_circular = np.random.vonmises(0, 2, 100)
# Wrap to [0, 2π]
data_circular = np.mod(data_circular, 2 * np.pi)
ax = axes[2, 1]
ax.plot(data_circular, "o", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_circular, name="angle_radians")
ax.set_title(
    f"Circular Data (angles)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.set_ylim(-0.5, 2 * np.pi + 0.5)
ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
ax.grid(True, alpha=0.3)

# Circular (degrees)
data_circular_deg = np.random.uniform(-180, 180, 100)
ax = axes[2, 2]
ax.plot(data_circular_deg, "o", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_circular_deg, name="heading_degrees")
ax.set_title(
    f"Circular Data (degrees)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.set_ylim(-200, 200)
ax.set_yticks([-180, -90, 0, 90, 180])
ax.grid(True, alpha=0.3)

# Row 4: Edge cases
# Mixed data (discrete values with noise)
data_mixed = np.random.choice([1, 2, 3, 4, 5], 100) + np.random.normal(0, 0.1, 100)
ax = axes[3, 0]
ax.plot(data_mixed, "o", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_mixed)
ax.set_title(
    f"Mixed (discrete + noise)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Perfect linspace (detected as timeline)
data_linspace = np.linspace(0, 10, 50)
ax = axes[3, 1]
ax.plot(data_linspace, "o-", markersize=4, alpha=0.7)
result = analyze_time_series_type(data_linspace)
ax.set_title(
    f"Perfect Linspace\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Periodic but not circular
t = np.linspace(0, 4 * np.pi, 100)
data_periodic = 5 * np.sin(t) + 10
ax = axes[3, 2]
ax.plot(data_periodic, "-", alpha=0.7)
result = analyze_time_series_type(data_periodic)
ax.set_title(
    f"Periodic (sine wave)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}\nPeriodicity: {result.periodicity:.1f} samples"
    if result.periodicity
    else f"Periodic (sine wave)\nDetected: {result.primary_type}/{result.subtype}\nConfidence: {result.confidence:.2f}"
)
ax.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
output_path = (
    "/Users/nikita/PycharmProjects/driada2/temp/ts_type_detection_examples.png"
)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Visualization saved to: {output_path}")

# Also create a summary table
print("\n" + "=" * 80)
print("TIME SERIES TYPE DETECTION SUMMARY")
print("=" * 80)

test_cases = [
    ("Binary [0,1]", np.array([0, 1, 0, 1, 1, 0] * 20)),
    ("Categorical (5 values)", np.random.choice([1, 2, 3, 4, 5], 100)),
    ("Count (monotonic)", np.cumsum(np.random.poisson(1, 50))),
    ("Timeline (integers)", np.arange(0, 100, 10)),
    ("Timeline (floats)", np.arange(0, 50, 0.5)),
    ("Poisson counts", np.random.poisson(5, 100)),
    ("Gaussian continuous", np.random.normal(0, 1, 100)),
    ("Uniform continuous", np.random.uniform(0, 10, 100)),
    ("Circular [0, 2π]", np.random.uniform(0, 2 * np.pi, 100)),
    ("Circular [-π, π]", np.random.uniform(-np.pi, np.pi, 100)),
    ("Degrees [0, 360]", np.random.uniform(0, 360, 100)),
    ("Perfect linspace", np.linspace(0, 10, 100)),
]

print(
    f"{'Data Type':<25} {'Primary':<12} {'Subtype':<12} {'Confidence':<10} {'Circular':<10} {'Period':<10}"
)
print("-" * 80)

for name, data in test_cases:
    # Add context for circular data
    context_name = None
    if "Circular" in name or "Degrees" in name:
        context_name = "angle"

    result = analyze_time_series_type(data, name=context_name)
    period_str = f"{result.circular_period:.2f}" if result.circular_period else "-"
    print(
        f"{name:<25} {result.primary_type:<12} {result.subtype:<12} {result.confidence:<10.2f} {str(result.is_circular):<10} {period_str:<10}"
    )

# plt.show()  # Don't show interactively, just save
