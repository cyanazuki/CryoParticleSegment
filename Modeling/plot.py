"""Define plotting function."""
import matplotlib.pyplot as plt


def plot_image(image, mask=None):
  fig, ax = plt.subplots(figsize=(4, 4))

  ax.imshow(image, cmap='gray')
  if mask is not None:
    ax.imshow(mask, cmap='inferno', alpha=0.2)
  ax.set_axis_off()

  plt.tight_layout()
  plt.show()

def plot_result(image, output, mask=None):
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  ax = axes.flatten()

  ax[0].imshow(image, cmap='gray')
  if mask is not None:
    ax[0].imshow(mask, cmap='inferno', alpha=0.2)
  ax[0].set_axis_off()

  ax[1].imshow(output, cmap='inferno')
  ax[1].set_axis_off()

  plt.tight_layout()
  plt.show()