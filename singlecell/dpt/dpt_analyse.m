function [Branch,DPT] = dpt_analyse(M,branching, tips)
% Given the accumulated transition matrix M, dpt_analyse performs pseudotime ordering 
% and in case of branching, assignes cell to three branches
%
% M: the accumulated transition matrix
% tips{i} tip cells of branch i
% branching detect a branching? (logical 0/1)
%
% Branch is branch labels for each cell, 1,2,3 or 0 for undeceided
% DPT is the diffusion pseudotime in respect to the root cell

import dpt.dpt_to_root;
import dpt.cutbranches_finitek2ways;
import dpt.organise_branch_labels;



n=size(M,1);
dpt_branch=cell(3,1);
indb=cell(3,1);

if (branching)
    for b=1:3
        sb=tips{b};
        dptbi=zeros(length(sb),n);
        for i=1:length(sb)
            dptbi(i,:)=dpt_to_root(M,sb(i));
        end
        dpt_branch{b}=mean(dptbi,1);
        [~,indb{b}]=sort(dpt_branch{b});
    end
    % cut it into three branches
    nBs=3;
    corrK=cell(nBs,3);
    cut=cell(nBs,3);
    branch=cell(nBs,1);
    %%%%%%%%%%%%%%%%%%%%%%
% branch{i} indices of cells in branch i
% unassigned cells not assigend to any of the branches
% dpt_branch{i} pseudotime with root at the tip of branch i
    for b=1:3
        [corrK{b},cut{b}, branch{b}]=cutbranches_finitek2ways(dpt_branch,indb ,n, b);
    end
    unassigned=setdiff(1:n,union(union(branch{1},branch{2}),branch{3}),'stable');
else  %if num_branche==1
     sb=tips{1};

     dptbi=zeros(length(sb),n);
     for i=1:length(sb)
         dptbi(i,:)=dpt_to_root(M,sb(i));
     end

     dpt_branch{1}=mean(dptbi,1);
     branch{1}=1:n;
     unassigned=[];
end
DPT=dpt_branch{1}';
Branch=organise_branch_labels(branch,unassigned);
end

