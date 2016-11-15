function [T1,T2, phi0] = T_classic(data1,data2, sigma)
%calculates classic diffusion transition matrix with fixed kernel width (sigma) for all
%cells
%data1=high dimensional matrix data to be analysied 
%data2=high dimensional matrix data to be projected on data1 
%sigma=diffusion scale parameter of the Gauusian kernel
%T1 : transition matrix of data1
%phi0: zeroth eig.vector of T1, corresponding eig.val=1
%T2 : transition matrix of data2
tic

n1=size(data1,1);
d2=pdist(data1).^2;
d2=squareform(d2);
S=sigma*sigma';
S2 = bsxfun(@plus,sigma.^2,sigma.^2');  
W1=sqrt(2*S./S2).*exp(-d2./S2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D1=sum(W1,2);
q=bsxfun(@times,reshape(D1,1,n1),reshape(D1,n1,1)).^1; %zx * zy
W1=W1./q;
%W1(1:n1+1:end)=0;
W1(d2==0)=0;

D1_=diag(sum(W1,2));

 T1=D1_^(-0.5)*W1*D1_^(-0.5);
 phi0 = diag(D1_)/sqrt(sum(diag(D1_).^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ( isequal(data1,data2) )
    T2=T1;
else
n2=size(data2,1);
W2=zeros(n2,n1);
d2=pdist2(data2,data1).^2;

sigma2=sigma; %global sigma in 'classic'
S=sigma2*sigma';
S2 = bsxfun(@plus,sigma2.^2,sigma.^2');  
W2=sqrt(2*S./S2).*exp(-d2./S2);
 
W2(d2==0)=0;
D2=sum(W2,2);

q=bsxfun(@times,reshape(D2,1,n2),reshape(D1,n1,1)).^(1);
W2=W2./q'; 
D2_=diag(sum(W2,2));
T2=D2_^(-0.5)*W2*D1_^(-0.5);

end

fprintf('%.04f', toc/60);
    
