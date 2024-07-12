import numpy as np
import matplotlib.pyplot as plt

# generate marginal probability vector in \R^n supported on [lend, rend]
# resulting vector is a linear combination of normal distributions with means=locs and standard deviations=scales
def plot_vectors(lend, rend, n, list_vectors, list_labels):
    x = np.linspace(lend, rend, n)
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    plt.figure(figsize=(8, 6))
    i = 0
    for (vec, label) in zip(list_vectors, list_labels):
        plt.plot(x, vec, label=rf'{label}', color=colors[i])
        i += 1
    plt.legend()
    plt.show()

def _plot_matrix(matrix, filename, save=False):
    fig = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.inferno, extent=(0.5,np.shape(matrix)[0]+0.5,0.5,np.shape(matrix)[1]+0.5))
    plt.colorbar()
    plt.show()
    if save == True:
        plt.savefig(fig, f"{filename}.png")

# display and save plots of a list of pytorch tensors
def plot_matrices(matrices, titles, filename="matrix_plot", save=False):
  if len(matrices) == 1:
      _plot_matrix(matrices[0],titles[0], filename, save)
  elif len(matrices) == 2:
      # Create a figure and subplots
      fig, axes = plt.subplots(1, 2, figsize=(14,14)) # Adjust figsize
  elif len(matrices) == 3:
      # Create a figure and subplots
      fig, axes = plt.subplots(1, 3, figsize=(14,14))  # Adjust figsize for better visualization
  elif len(matrices) == 4:
      # Create a figure and subplots
      fig, axes = plt.subplots(2, 2, figsize=(14,14)) # Adjust figsize
  else:
      print("Plots support 4 tensors at maximum.")

  # Loop through subplots and plot each matrix
  axes = axes.flatten()
  for i, matrix in enumerate(matrices):
    ax = axes[i]
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.inferno, extent=(0.5, np.shape(matrix)[0] + 0.5, 0.5, np.shape(matrix)[1] + 0.5))
    ax.set_title(titles[i])  # Add title to each subplot (optional)
  # Adjust layout (optional)
  fig.colorbar(im, ax=axes, shrink=0.5, location="bottom")
  plt.show()

  if save == True:
      fig.savefig(f"{filename}.png")