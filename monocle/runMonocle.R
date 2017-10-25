# Script to reproduce monocle analysis
# 
# To run this script 
# Download hematopoiesis data from https://github.com/ManuSetty/wishbone 
#  Store in file data/wishbone_rna_seq.csv
#
#
#
#

library(monocle)
rf = read.csv(file = 'data/wishbone_rna_seq.csv', header=TRUE, row.names=1)
datafile = t(rf)
dim(datafile)
# (2312, 4423) # G X N

# First create a CellDataSet from the relative expression levels
HSMM <- newCellDataSet(as.matrix(datafile),
                        lowerDetectionLimit = 0.001,
                        expressionFamily = tobit(Lower = 0.001))
# Apply Monocle-DDRTree algorithm
HSMM <- estimateSizeFactors(HSMM)
HSMM <- reduceDimension(HSMM) #no normalization is required for gaussian data
HSMM <- orderCells(HSMM, reverse=FALSE)
HSMM <- orderCells(HSMM, reverse=TRUE)


# Apply BEAM and branching time point analysis
b = BEAM(HSMM)
HSMMILRS <- calILRs(HSMM, branch_point = 1, return_all = T)
branchtimepoint <- detectBifurcationPoint(HSMMILRS$str_logfc_df, return_cross_point = T)

# Save output data
write.csv(HSMM$Pseudotime, file='MonoclePseudotime.csv')
write.csv(HSMM$State, file='MonocleBranch.csv')
write.csv(HSMM@reducedDimS, file='MonocleLatentSpace.csv')
write.csv(b, file='MonocleBEAM.csv')
write.csv(branchtimepoint, file='Monocle_branchtimepoint.csv')
write.csv(HSMMILRS$str_branchA_expression_curve_matrix, file='Monocle_str_branchA_expression.csv')
write.csv(HSMMILRS$str_branchB_expression_curve_matrix, file='Monocle_str_branchB_expression.csv')

