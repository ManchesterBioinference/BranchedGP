
Matlab Package for DPT (Diffusion Pseudotime) analysis of single-cell data

Code written by:
Laleh Haghverdi, Fabian J. Theis

This package uses Matlab’s Statistics and Machine learning toolbox.

===========pseudotime and branching 
The complete procedure of DPT analysing a branching or non-branching data consists of 4 main steps:

1) Provide (load) the n*G data matrix and (if applicable) cell labels and (if applicable) do preprocessing. n=number of cells, G=number of Genes (or features).

2) Build the transition matrix and its first (non-informative) left
eigenvector using one of the three methods 'classic', 'loc' or 'nn’. (transition_matrix.m). 

3) Calculate accumulated transition matrix M and input and organise the parameters to find the tips of branches (dpt_input.m).

4) Do pseudotime ordering and separate the branches (dpt_analyse.m).

This procedure is implemented in examples/toy1.m, examples/ESC_qpcr.m and dpt_DropSea_data/ESC_dropseq.m to respectively generate figures of DPT performance on toy data, early blood qPCR data and the ESC DropSeq data as referenced in the paper.

The DropSeq data and example is available separately in dpt_DropSea_data/ from out website, due to large size.

===========automated branch finding
For automated identification of all branches and subbranches, it is possible to run the auto_tree function.

gstatmin=1.01; k=20; nsig=10; [I,DPT,phi]=test_autotree(data,'nn',[k,nsig],root,gstatmin);
figure; scatter(phi(:,2),phi(:,3),50,I,'fill') colormap lines;
figure; scatter(phi(:,2),phi(:,3),50,DPT,'fill') colormap jet;

By this you find the branchings in the data but might include false positive (too finegrain) branches as well depending on “gstatmin”.

===========diffusion map and data projection
The package also includes diffusion map plotting and projection of new data onto the diffusion map. see  examples/test_plot_diffusionmap.m and examples/test_data_projection.m

===========pre-post processing functions
several other useful pre-post processing functions are provided in the pre_post_process folder, including cosine-distance_normalisation.m and lam.m respectively for performing DPT with cosine distance and local approximation of missing values.
 
===========universal time from time-lapse data
examples/traj_G1G2.m is a demonstration of obtaining universal time from a (time-lapse) single cell trajectory. It generates supplementary figure 1.
===========

If you have any problem or question using the package please contact laleh.haghverdi@helmholtz-muenchen.de
