"""Update imports in info_base.py, entropy.py, and intense_base.py"""

# Step 1: Update info_base.py - Remove FFT functions and add re-exports
print("Step 1: Updating info_base.py...")
with open('src/driada/information/info_base.py', 'r', encoding='utf-8') as f:
    info_base_lines = f.readlines()

# Remove lines 1725-2669 (0-indexed: 1724-2669)
new_info_base = info_base_lines[:1724] + info_base_lines[2669:]

# Add re-export imports after the existing imports (around line 21)
# Find where to insert (after "from .entropy import...")
insert_idx = None
for i, line in enumerate(new_info_base):
    if 'from .entropy import' in line:
        insert_idx = i + 1
        break

if insert_idx:
    reexport = """
# Re-export FFT functions from info_fft module (for backward compatibility)
from .info_fft import (
    compute_mi_batch_fft,
    compute_mi_gd_fft,
    compute_mi_mts_fft,
    compute_mi_mts_mts_fft,
    compute_mi_mts_discrete_fft,
)

"""
    new_info_base.insert(insert_idx, reexport)

with open('src/driada/information/info_base.py', 'w', encoding='utf-8') as f:
    f.writelines(new_info_base)

old_lines = len(info_base_lines)
new_lines = len(new_info_base)
print(f"  Removed {old_lines - new_lines + 9} lines, added 9 re-export lines")
print(f"  info_base.py: {old_lines} → {new_lines} lines")

# Step 2: Update entropy.py - Remove mi_cd_fft and add re-export
print("\nStep 2: Updating entropy.py...")
with open('src/driada/information/entropy.py', 'r', encoding='utf-8') as f:
    entropy_lines = f.readlines()

# Remove lines 378-501 (0-indexed: 377-501)
new_entropy = entropy_lines[:377] + entropy_lines[501:]

# Add re-export at the end of imports
insert_idx = None
for i, line in enumerate(new_entropy):
    if line.strip().startswith('def ') or line.strip().startswith('class '):
        insert_idx = i
        break

if insert_idx:
    reexport = """
# Re-export mi_cd_fft from info_fft module (for backward compatibility)
from .info_fft import mi_cd_fft

"""
    new_entropy.insert(insert_idx, reexport)

with open('src/driada/information/entropy.py', 'w', encoding='utf-8') as f:
    f.writelines(new_entropy)

old_lines = len(entropy_lines)
new_lines = len(new_entropy)
print(f"  Removed {old_lines - new_lines + 3} lines, added 3 re-export lines")
print(f"  entropy.py: {old_lines} → {new_lines} lines")

# Step 3: Update intense_base.py - Change import source
print("\nStep 3: Updating intense_base.py...")
with open('src/driada/intense/intense_base.py', 'r', encoding='utf-8') as f:
    intense_lines = f.readlines()

new_intense = []
for line in intense_lines:
    if 'from ..information.info_base import' in line and 'compute_mi_' in line:
        # Replace with info_fft import
        line = line.replace('from ..information.info_base import', 'from ..information.info_fft import')
    new_intense.append(line)

with open('src/driada/intense/intense_base.py', 'w', encoding='utf-8') as f:
    f.writelines(new_intense)

print(f"  Updated FFT imports to use info_fft module")
print(f"  intense_base.py: {len(intense_lines)} lines (unchanged)")

print("\n✓ All files updated successfully!")
print("\nSummary:")
print(f"  - Created: info_fft.py (1129 lines)")
print(f"  - Updated: info_base.py ({old_lines} → {new_lines} lines, -{old_lines-new_lines+9} lines)")
print(f"  - Updated: entropy.py ({len(entropy_lines)} → {len(new_entropy)} lines, -{len(entropy_lines)-len(new_entropy)+3} lines)")
print(f"  - Updated: intense_base.py (import source changed)")
