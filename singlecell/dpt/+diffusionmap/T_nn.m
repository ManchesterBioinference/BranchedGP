function [T1,T2, phi0] = T_nn(data1,data2,k,nsig)
%performs nearest neighbours version of diffusion map with locally ajdusted kernel width, recommended for large data size as in Dropseq and RNA-Seq. 

% k= no. of nearest neighbours for building the nn-graph
% nsig= no. of nearest neighbours for adjusting the Gauusian kernel width.
% nsig<k must hold
%data1=high dimensional matrix data to be analysied 
%data2=high dimensional matrix data to be projected on data1 

%T1 : transition matrix of data1
%phi0: zeroth eig.vector of T1, corresponding eig.val=1
%T2 : transition matrix of data2



import third_party.parfor_spdists_knngraph;
import third_party.spdists_undirected;
tic
%l=50;
ch_s=min((size(data1,1)-1),1000);
%lnn = parfor_spdists_knngraph( data, k, 'distance', 'Euclidean', 'chunk_size', ch_s, 'SNN', true, 'verbose', true );
lnn = parfor_spdists_knngraph( data1, k, 'distance', 'Euclidean', 'chunk_size', ch_s, 'verbose', true);
lnn = spdists_undirected( lnn ); %make lnn undirected	

n1=size(data1,1);
[row, col, valuesF] = find(lnn);

m=sum((sum(lnn~=0)));
d2s=reshape(valuesF,m,1); 

sd=@(x) sort(x,2,'ascend');
sdists=spfun(sd,lnn); 
[rows, cols, vals] = find(sdists);
sdistsS=[rows, cols, vals]; 

%%%%%%%%%%%%%%%%%%%
sigma=zeros(n1,1);
for i=1:n1
    temp=sdistsS(sdistsS(:,2)==i);
    sigma(i)=sum(temp(nsig:nsig+2))/3/2;
end

sigma_ij=zeros(m,1);
pref_ij=zeros(m,1);

for rowi=1:m
sigma_ij(rowi,1)=(sigma(row(rowi))^2+sigma(col(rowi))^2);
pref_ij(rowi,1)=sqrt( 2*sigma(row(rowi))*sigma(col(rowi)) / sigma_ij(rowi,1) );
end

W1=sparse(row, col, pref_ij.*exp(-d2s.^2 ./(2*sigma_ij)), n1, n1);
D1=(sum(W1));
W1=sparse(row, col, pref_ij.*exp(-d2s.^2 ./(2*sigma_ij)) ./ (D1(1,row) .* D1(1,col))', n1, n1);
D1_=diag(sum(W1, 2));
T1=full(D1_)^(-0.5)*W1*full(D1_)^(-0.5); 

phi0 = diag(D1_)/sqrt(sum(diag(D1_).^2));

if ( isequal(data1,data2) )
    T2=T1;
else
    n2=size(data2,1);
    [ idx, d ] = knnsearch( data1, data2, 'k', k + 1 );

	idx( :, 1 ) = []; d( :, 1 ) = []; % remove self neighbor
    sigma2=sum(d(:,nsig:nsig+2),2)/3/2;
        
    js = repmat( (1:n2)', 1, k ); 
    indices = sub2ind( [n2,n1],js(:), idx(:));
    d2=zeros(n2,n1);
    d2(indices)=d(:).^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    S=sigma2*sigma'; 
    S2 = bsxfun(@plus,sigma2.^2,sigma.^2');  
    W2=sqrt(2*S./S2).*exp(-d2./S2);

    W2(d2==0)=0;
    D2=sum(W2,2);

    q=bsxfun(@times,reshape(D2,1,n2),reshape(D1,n1,1)).^(1);
    W2=W2./q'; 
    D2_=diag(sum(W2,2));
    T2=full(D2_)^(-0.5)*W2*full(D1_)^(-0.5);

end

toc/60
    
