function plot_average_path(DPT,B1,X)
%plots the average path X over pseudotime on for cell indices in B
%DPT is the pseudotime for all cells
%B1 is the indices for the cells of interest (e.g belonging to a branch)
%X is the 2D coordinates (e.g. from PCA or DM)

%[a,b,c]=pca(data);
%X=b;
%B1=[find(Branch==1);find(Branch==3);find(Branch==0)];

%X=phi(:,2:3);
%[val,ind]=sort(vb{1}(B1));
%B=B1(ind);

[~,ind]=sort(DPT(B1));
B=B1(ind);

Y=X(B,:);
windowSize = 50;
for i=1:size(Y,2), 
    Y(:,i)=smooth(Y(:,i),windowSize); 
end

hold on, plot(Y(:,1),Y(:,2),'k-', 'LineWidth',2), %hold off