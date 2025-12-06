"""
Combine two existing phase portrait images into a single figure.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# UPDATE THESE PATHS to your actual image locations
img1_path = "belief_inertia/phase_portrait_damped.png"
img2_path = "belief_inertia/phase_portrait_orbit.png"

# Load images
img1 = mpimg.imread(img1_path)
img2 = mpimg.imread(img2_path)

# Create combined figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(img1)
ax1.set_title('(a) Overdamped', fontsize=12)
ax1.axis('off')

ax2.imshow(img2)
ax2.set_title('(b) Underdamped', fontsize=12)
ax2.axis('off')

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "phase_portraits_combined.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')

print(f"Saved: {output_path}")
