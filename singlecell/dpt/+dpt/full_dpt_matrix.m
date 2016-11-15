function fulldptmatrix=full_dpt_matrix(M)
%calculates the full pairwise dpt distances as an n*n matrix (fulldptmatrix)
%In DPT only the dpt distance to the root or tip cells are used and there
%is no need to calculate the fulldptmatrix as here. However, the
%full dpt matrix can be useful for other tree constructions
% M : the accumulated transition matrix

n=size(M,1);
fulldptmatrix=zeros(n,n);
for i=1:n
        fulldptmatrix(i,:)=dpt.dpt_to_root(M,i);
end

%%%%example 
% fulldptmatrix=full_dpt_matrix(M);
% tree=linkage(fulldptmatrix,'ward');
% c = cluster(tree,'maxclust',4);
% scatter(phi(:,2),phi(:,3),10,c,'fill')