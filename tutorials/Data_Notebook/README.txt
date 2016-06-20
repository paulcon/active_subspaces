The Data_demonstration notebook reads in data (inputs, outputs, gradients) and computes the active subspace,
using OLS if gradients are absent. It makes plots of eigenvalues and subspace errors (both with bootstrap
ranges), eigenvector components, and sufficient summary plots for each output.

There are 3 files that can be present: inputs.txt, outputs.txt, and gradients.txt. These should have NO
header and NO index column, and be comma-separated. Each row of the file should correspnd to a single run 
of the simulation (each column represents a variable or output). The top part of the notebook has variables
that must be changed case-by-case: variable limits (xl, xu), labels (in_labels, out_labels), sstype, and
wrt_orig.

The notebook will also use the animate.py file to make animations demonstrating the active subspace if the 
input dimension is low enough and there are few enough outputs. These will be saved as .mp4's and this requires
installation of ffmpeg or mencoder.

The folders in this directory contain examples; copy/paste the data files into the directory with the notebook,
alter the top part of the notebook, and run the notebook to see it in action.
