# File 1) all cells - non–branching data
# When loaded in R each file contains 2 objects:
# molecule.counts – a sparse matrix (requires Matrix package) where 
# rows are genes and columns are cells and the entries are raw 
# molecule counts
# norm.expression – a matrix of normalized expression with the e
# ffects of sequencing depth and cell cycle removed; only the more 
# variable genes are included in this matrix

setwd("/home/mqbssaby/pythonprojects/BranchedGP/singlecell/datawishbone")
#load("~/Downloads/neuronal_data_CA77CGE.Rda")
#write.csv(norm.expression, 'neuronal_data_CA77CGE.csv')
# 5920 cell, 3523 genes

# File 2) only post-mitotic cells – we see a branch when we focus 
# just on these cells
a <- read.csv('/home/mqbssaby/pythonprojects/BranchedGP/singlecell/datawishbone/wishbone_rna_seq.csv')
b=a[,2:2313]
norm.expression <- as.matrix(t(b))
#write.csv(norm.expression, 'neuronal_data_CA77CGE_postmitotic.csv')
# 2405 cells, 3299 genes


library(monocle)

HSMM <- newCellDataSet(norm.expression)
HSMM <- reduceDimension(HSMM, max_components=2)
HSMM <- orderCells(HSMM, reverse=FALSE)
plot_cell_trajectory(HSMM)

# save
write.csv(HSMM@reducedDimS, file='monocle/monocleDimRed.csv')
write.csv(HSMM$Pseudotime, file='monocle/monoclePT.csv')
write.csv(HSMM$State, file='monocle/monocleState.csv')
HSMMDefault = HSMM

HSMMRoot <- orderCells(HSMM, reverse=FALSE, root_state=41)
plot_cell_trajectory(HSMMRoot)
write.csv(HSMMRoot@reducedDimS, file='monocle/monocleDimRed.csv')
write.csv(HSMMRoot$Pseudotime, file='monocle/monoclePT.csv')
write.csv(HSMMRoot$State, file='monocle/monocleState.csv')
HSMM = HSMMRoot

# could also do 
BEAM_res <- BEAM(HSMM, branch_point=9, cores = 8)
BEAM_res <- BEAM_res[order(BEAM_res$qval),]
BEAM_res <- BEAM_res[,c("gene_short_name", "pval", "qval")]

gene_short_name = row.names(BEAM_res)
# plot heatmap
plot_genes_branched_heatmap(HSMM[row.names(subset(BEAM_res, qval < 1e-4)),],
                            branch_point = 9,
                            num_clusters = 4,
                            cores = 1,
                            use_gene_short_name = T,
                            show_rownames = T)

hsmm_genes <- c("Ulk3", "Lrp12", "Gap43")
plot_genes_branched_pseudotime(HSMM[hsmm_genes,],
                               branch_point=9,
                               ncol=1)

                    
