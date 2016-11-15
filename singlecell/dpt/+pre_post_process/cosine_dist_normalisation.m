function Y=cosine_dist_normalisation(X)
%performs per cell normalisation Y=X/norm(X)
%using Euclidian distance after this normalisation will yield the same
%result as using cosine distance
%This is because norm(Y)=1 for all cells.
%i.e.: (Y1-Y2)^2=Y1^2+Y2^2-2Y1*Y2=1+1-2Y1*Y2=2*( 1-( Y1*Y2 ) )
%=2*( 1- ( X1*X2 )/( |X1|*|X2| ))= 2*cosin_dist(X1,X2)

G=size(X,2);
cellnorm=sqrt(sum(X.^2,2));
cellnorm_nonzero=cellnorm;
cellnorm_nonzero(cellnorm==0)=1;
Y=X./repmat(cellnorm_nonzero,1,G);
Y(cellnorm==0,:)=0;

