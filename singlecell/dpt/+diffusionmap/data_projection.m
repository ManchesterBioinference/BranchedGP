function [phi1,phi2]=data_projection(T1,T2)
%transition matrix of data1
%transition matrix of data2 (transitions relative to data1)
%phi1 eig.vects of T1 which provides the DM coordinates of data1
%lambda1 eig.vals of T1
%phi2 projection DM coordinates of data2

[phi1, lambda1] = diffusionmap.eig_decompose_normalized(T1,4);
phi2=(T2*phi1)./repmat(lambda1',size(T2,1),1);
