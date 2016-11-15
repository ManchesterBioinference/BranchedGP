
function tips=suggest_tips(method,params,num_branches)
% finds tip of branches by maxdptdist or manual method
% use dpt_input.m to set the params
%num_branches 1 (if nonbranching) or 3 (if branching) 

import dpt.dpt_to_root;
import dpt.manual_select_cell;


if strcmp(method,'maxdptdist')

    if length(params)==3   
        M=params{1};
        phi0=params{2};
        root=params{3}(1);
        tips{1}=root;
        if num_branches>1
            x=root;
            dx=dpt_to_root(M,x);
            [~,y]=max(dx);
            dy=dpt_to_root(M,y);
            [~,z]=max(dx+dy);
            tips{2}=y; tips{3}=z;
        end
    else
        M=params{1};
        phi0=params{2};
        rn=ceil(rand*size(M,1));
        drn=dpt_to_root(M,rn);
        [~,x]=max(drn);
        tips{1}=x;
        if num_branches>1
            dx=dpt_to_root(M,x);
            [~,y]=max(dx);
            dy=dpt_to_root(M,y);
            [~,z]=max(dx+dy);
            tips{2}=y; tips{3}=z;
        end
    end

elseif strcmp(method,'manual')


    embedV1=params{1}(:,1);  
    embedV2=params{1}(:,2); 
    k=params{2}; 
    labels=params{3};
    tips{1}=manual_select_cell(embedV1,embedV2,k,labels);
    if num_branches>1
        tips{2}=manual_select_cell(embedV1,embedV2,k,labels);
        tips{3}=manual_select_cell(embedV1,embedV2,k,labels);
    end
else
    error('unknown method: %s', method)
end

X = sprintf('tip cell indices: %d %d %d',cell2mat(tips));
disp(X)

end
