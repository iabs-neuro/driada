import numpy as np
import matplotlib.pyplot as plt
from driada.rsa.visual import plot_rdm

# Create test RDM
rdm = np.random.rand(5, 5)
rdm = (rdm + rdm.T) / 2  # Make symmetric
np.fill_diagonal(rdm, 0)

labels = ["A", "B", "C", "D", "E"]

fig = plot_rdm(rdm, labels=labels, title="Test RDM")

print(f"Number of axes: {len(fig.axes)}")
for i, ax in enumerate(fig.axes):
    print(f"Axis {i}: {ax}")
    print(f"  Title: {ax.get_title()}")
    print(f"  Xticklabels: {[label.get_text() for label in ax.get_xticklabels()]}")
    print(f"  Is colorbar: {'<colorbar>' in str(ax.get_label())}")
    print()

# Find the main heatmap axis
main_ax = None
for ax in fig.axes:
    if 'colorbar' not in str(ax.get_label()) and ax.get_title():
        main_ax = ax
        break

if main_ax:
    print(f"Main axis found: {main_ax}")
    print(f"Main axis xticklabels: {[label.get_text() for label in main_ax.get_xticklabels()]}")

plt.close(fig)