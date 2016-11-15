function M=low_dim_M(phi,lambda,l)
%low dimensional approximation of the accumulated transition matrix M
%useful to implement (uncoment) in dpt_input.m for more efficient performance of DPT with large cell numbers  
%phi : eig.vects of diffusion transition matrix
%lambda : eig.vals of diffusion transition matrix
%l : how many of first components to use for approximation 

 phi(:,1)=[];
 lambda(1)=[];
lambda=diag(lambda);
M= phi*( (lambda*(eye(l-1)-lambda)^-1) ) *phi';

