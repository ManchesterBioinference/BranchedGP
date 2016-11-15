% auto_tree - diffusion pseudotime
%   determine diffusion pseudotime and splits
%   the data into multiple branches
%
%   usage:
%       tr=auto_tree(data,method,params,root,gstatmin)
%
%   input:
%       data   data set in the form cells x genes
%       root   sample index of the root cell; if not specified determined
%               automatically (but ordering/direction in tree will be off); 
%               changed: needs to be set or picked, 0 not ok
%               if set to -1 user can pick it in a GUI
%       method  method for calculation of transition matrrix 'loc','nn' or
%       'classic'
%       params  parameters for calculation of transition matrrix e.g. 
%       gstatmin min of gap statistics for allowing a cut
%       
%
%   output:
%       tree object
%
%   Fabian Theis 2015
%
function tr=auto_tree(data,method,params,root,gstatmin)

%data: n*G input data matrix
%method: 'nn' , 'loc' or 'classic' correspondingly for nearest neighbours, local
% kernel width or the classic version of diffusion matrix
%params: diffusion map parameters e.g. [k,nsig]
%root: root cell for pseudotime ordering
%gstatmin: statistical value for stopping the iterative branch searching

if nargin<4, root=0; end
if nargin<5, gstatmin=1.1; end

k=params(1);
if strcmp(method,'classic') 
    k=30;
end
%nsig=difmap_params(2);
[n,~]=size(data);

[T,phi0] = transition_matrix(data,method,params);

%% automatic assignment of cells
x=root;
branching=1;

[M, tips] = dpt_input(T, phi0, branching, 'maxdptdist', root);

%rand('seed',2)
temp=ceil(rand*n);
dtemp= dpt.dpt_to_root(M,temp);  
[~,x]=max(dtemp);
dx= dpt.dpt_to_root(M,x);  
[~,y]=max(dx);
dy= dpt.dpt_to_root(M,y);  
[~,z]=max(dx+dy);

%[M, tips] = dpt_input(T, phi0, branching, 'maxdptdist');%, root);
%tips{1}=x; tips{2}=y; tips{3}=z; 

gstat=max(dx+dy)/min(dx+dy);
fprintf('gap statistics %f\n',gstat); 

%fprintf('found extreme cells %i, %i, %i\n',x,y,z);
% split off branch is too close to original tree, dont cut
if gstat<gstatmin
    [~,DPT]=dpt_analyse(M,0,tips);
    [~,indb]=sort(DPT); 
    tr=tree(indb);
    return;
end

[Branch,DPT]=dpt_analyse(M,1,tips);
%%% now do the same for each sub branch
minbranchlengthforfurthersplitup=k;%10;

br{1}=[find(Branch==1)];
br{2}=[find(Branch==2)];%;find(Branch==0)];
br{3}=[find(Branch==3)];%;find(Branch==0)];
br{4}=find(Branch==0);  %undecided cell, it is possible to include them...
%in either br{2} or br{3} instead

if length(br{1})<k || length(br{2})<k || length(br{3})<k
    [~,DPT]=dpt_analyse(M,0,tips);
    [~,indb]=sort(DPT); 
    tr=tree(indb);
    return;
end

%%%
%tree=[0 1 1];
tr=[];
lastnode=0;
for i=1:length(br)
    b=br{i}';
    [~,ind1]=sort(DPT(b));
    if length(b)>minbranchlengthforfurthersplitup % need minbranchlengthforfurthersplitup samples in each branch to divide it further
        root2=ind1(1);
        sub=dpt.auto_tree(data(b,:),method,params,root2,gstatmin); % do tree splitup in subbranch,
        sub=sub.treefun(@(x) b(x)); % remap indices back to branch index from above  
    else        
        sub=tree(b);
    end
    if i==1 && length(b)>minbranchlengthforfurthersplitup  % either replace or attach to bottom
        tr=sub;
        % also save node index where the split happens
        splitid=b(end);
        lastnode=find(~tr.treefun(@(x) find(x==splitid)).isemptynode);
    else
        tr=tr.graft(lastnode,sub);
    end
end

