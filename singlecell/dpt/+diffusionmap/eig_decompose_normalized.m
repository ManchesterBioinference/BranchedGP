function [ phi, lambda ] = eig_decompose_normalized(T,l)
% calculates l sorted eigenvalues and eigenvectors of T (lambda and phi) 

[phi,lambda]=eigs(T,l);
[lambda,ind]=sort(diag(lambda),'descend');
phi=phi(:,ind);

end

