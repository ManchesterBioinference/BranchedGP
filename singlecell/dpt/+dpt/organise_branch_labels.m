function Branch=organise_branch_labels(branch,unassigned)
% This function organizes the cell arrays branch{i} and unassigned (created in
% dpt_analyse.m)
% to build Branch labels (length=number of cells) indicating the branch each cell belongs to.
% Cells which are assigned to more than one branch in dpt_analyse.m as well
% as cells which are not assigned to any branch are defined as undeceided
% (label 0)

if length(branch)>1
    
    inters23=intersect(branch{2},branch{3});
    branch{2}=setdiff(branch{2},inters23);
    branch{3}=setdiff(branch{3},inters23);
    branch{1}=setdiff(branch{1},inters23);

    

    for i=2:3
        inters1i=[];
        inters1i=intersect(branch{i},branch{1});
        branch{1}=setdiff(branch{1},inters1i);
    end
    

    branch{4}=union(unassigned,inters23);
    n=max([branch{4}, branch{1},branch{2},branch{3}]);

    Branch=zeros(n,1);
    for i=1:n
        b=0;
        lab=0;
        while ~lab
            b=b+1;
            lab=ismember(i,branch{b});
        end
        Branch(i)=b;
    end
    Branch(Branch==4)=0;
else
    n=max(branch{1});
    Branch=ones(n,1);
end


