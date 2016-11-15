%given two ordering dpt_branch1 and dpt_branch2, this code computes the delta (e.i. finite) Keandall
%correlation of adding a new cell with [newdpt_branch1, newdpt_branch2] to the orderings.

function finitK=finite_kendall_corr(dpt_branch1,dpt_branch2,newdpt_branch)

%dpt_branch1=dpt_branch{b1};
%dpt_branch2=dpt_branch{b2};
dpt_branch11=zeros(size(dpt_branch1,1),1);
dpt_branch11(dpt_branch1>=newdpt_branch(1))=1;
dpt_branch11(dpt_branch1<newdpt_branch(1))=-1;

dpt_branch22=zeros(size(dpt_branch1,1),1);
dpt_branch22(dpt_branch2>=newdpt_branch(2))=1;
dpt_branch22(dpt_branch2<newdpt_branch(2))=-1;

finitK=dot(dpt_branch11,dpt_branch22);
end
