"""
Test the updated TimeSeries class with type information storage.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driada.information.info_base import TimeSeries
from driada.information.time_series_types import TimeSeriesType

# Example 1: Auto-detection
print("1. Auto-detection examples:")
print("-" * 50)

# Binary data
binary_data = np.array([0, 1, 0, 1, 1, 0] * 20)
ts_binary = TimeSeries(binary_data)
print(f"Binary data: discrete={ts_binary.discrete}, is_binary={ts_binary.is_binary}")
print(
    f"  Type info: {ts_binary.type_info.primary_type}/{ts_binary.type_info.subtype}, confidence={ts_binary.type_info.confidence:.2f}"
)

# Circular data with name context
circular_data = np.random.uniform(0, 2 * np.pi, 100)
ts_circular = TimeSeries(circular_data, name="phase_angle")
print(f"\nCircular data: discrete={ts_circular.discrete}")
print(
    f"  Type info: {ts_circular.type_info.primary_type}/{ts_circular.type_info.subtype}"
)
print(
    f"  Is circular: {ts_circular.type_info.is_circular}, period={ts_circular.type_info.circular_period}"
)

# Count data
count_data = np.cumsum(np.random.poisson(2, 50))
ts_count = TimeSeries(count_data)
print(f"\nCount data: discrete={ts_count.discrete}")
print(
    f"  Type info: {ts_count.type_info.primary_type}/{ts_count.type_info.subtype}, confidence={ts_count.type_info.confidence:.2f}"
)

# Example 2: Legacy discrete parameter
print("\n\n2. Legacy discrete parameter:")
print("-" * 50)

# Force continuous despite integer values
integer_data = np.array([1, 2, 3, 4, 5] * 20)
ts_forced_cont = TimeSeries(integer_data, discrete=False)
print(f"Integer data forced continuous: discrete={ts_forced_cont.discrete}")
print(
    f"  Type info: {ts_forced_cont.type_info.primary_type}, confidence={ts_forced_cont.type_info.confidence}"
)

# Example 3: Manual type specification
print("\n\n3. Manual type specification:")
print("-" * 50)

# Create a custom type specification
custom_type = TimeSeriesType(
    primary_type="continuous",
    subtype="circular",
    confidence=0.95,
    is_circular=True,
    circular_period=360,
    periodicity=None,
    metadata={"custom": True, "units": "degrees"},
)

# Use it for angle data
angle_data = np.random.uniform(-180, 180, 100)
ts_manual = TimeSeries(angle_data, ts_type=custom_type)
print(f"Manual type specification: discrete={ts_manual.discrete}")
print(f"  Type info: {ts_manual.type_info.primary_type}/{ts_manual.type_info.subtype}")
print(f"  Circular period: {ts_manual.type_info.circular_period}")
print(f"  Custom metadata: {ts_manual.type_info.metadata}")

# Example 4: Access to full type information
print("\n\n4. Accessing full type information:")
print("-" * 50)

# Timeline data
timeline_data = np.arange(0, 50, 0.5)
# Shuffle to make it non-monotonic but still timeline
np.random.shuffle(timeline_data)
ts_timeline = TimeSeries(timeline_data[:100])

print("Timeline data:")
print(f"  Primary type: {ts_timeline.type_info.primary_type}")
print(f"  Subtype: {ts_timeline.type_info.subtype}")
print(f"  Confidence: {ts_timeline.type_info.confidence:.2f}")
print(f"  Is circular: {ts_timeline.type_info.is_circular}")
print(f"  Periodicity: {ts_timeline.type_info.periodicity}")
print(
    f"  Metadata stats: n_unique={ts_timeline.type_info.metadata.get('n_unique', 'N/A')}, "
    f"uniqueness_ratio={ts_timeline.type_info.metadata.get('uniqueness_ratio', 'N/A'):.2f}"
)
