
function [I,DPT,phi]=test_autotree(data,method,params,root,gstatmin)
%example application of auto_tree
%data: n*G input data matrix
%method: 'nn' , 'loc' or 'classic' correspondingly for nearest neighbours, local
% kernel width or the classic version of diffusion matrix
%params: diffusion map parameter e.g. [k,nsig]
%root: root cell for pseudotime ordering
%gstatmin: statistical value for stopping the iterative branch searching

%% test automatic assigment and splitting up of cells
% Fabian 12-Nov-15
addpath(fileparts(fileparts(mfilename('fullpath'))));
n=size(data,1);

%% now the tree version
if nargin<5, gstatmin=1.01; end

tr=dpt.auto_tree(data,method,params,root,gstatmin);
%% plot it
close all
I=zeros(n,1); for i=tr.breadthfirstiterator, I(tr.get(i))=i; end
[T, phi0]=transition_matrix(data,method,params);
[phi, ~] = diffusionmap.eig_decompose_normalized(T,10);
X=phi(:,2:3);
scatter(X(:,1),X(:,2),50,I,'fill')
xlabel('DC1')
ylabel('DC2')
colormap jet

[M, tips] = dpt_input(T, phi0, 1, 'maxdptdist', root);
[~,DPT]=dpt_analyse(M,1,tips);

%% extract time courses
for leave=tr.findleaves
    ts=[];
    for i=tr.findpath(1,leave)
        ts=[ts, tr.get(i)];
    end
    Y=X(ts,:);
    windowSize = 100;
    %for i=1:size(Y,2), Y(:,i)=filter((1/windowSize)*ones(1,windowSize),1,Y(:,i)); end
    for i=1:size(Y,2), Y(:,i)=smooth(Y(:,i),windowSize); end
    hold on, plot(Y(:,1),Y(:,2),'LineWidth',3), 
   
end

%% number of cells in the tree
tl=tr.treefun(@length); 
l=0; for i=tl.depthfirstiterator, l=l+tl.get(i); end,
