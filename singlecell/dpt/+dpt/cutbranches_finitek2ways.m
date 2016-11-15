%this fuction assignes cells to a branch b3 by maximizing finite Kendall correlation of dpts of the
%other two branches on b3+ finite Kendall anticorrelation of dpts on b2 and b1 

%dpt_branch{i} is the pseudo time distance in branch i (i=1:3)
%indb{i} cell index in ascending pseudotime distance of branch i
%n number of cells 
%b which branch to cut 


function [corrK, cut, branch]=cutbranches_finitek2ways(dpt_branch,indb ,n, b)

import dpt.finite_kendall_corr;

B=[1,2,3];
B(B==b)=[];
b3=b;
b2=B(1);
b1=B(2);


dpt_branch1=[];
dpt_branch2=[]; 
dpt_branchr1=dpt_branch{b1}(indb{b3}(1:n));
dpt_branchr2=dpt_branch{b2}(indb{b3}(1:n));
corrK=zeros(n-1,1);

for ss=1:n-1
    
    dpt_branch1=[dpt_branch1,dpt_branch{b1}(indb{b3}(ss))];
    dpt_branch2=[dpt_branch2,dpt_branch{b2}(indb{b3}(ss))];
    
    dpt_branchr1(1)=[];
    dpt_branchr2(1)=[];
    
    finiteKp=finite_kendall_corr( dpt_branch1,dpt_branch2,[dpt_branch{b1}(indb{b3}((ss+1))),dpt_branch{b2}(indb{b3}((ss+1)))] );
    finiteKn=finite_kendall_corr( dpt_branchr1,dpt_branchr2,[dpt_branch{b1}(indb{b3}((ss))),dpt_branch{b2}(indb{b3}((ss)))] );
        
     corrK(ss)=finiteKp/ss-finiteKn/(n-ss);
%    corrK(ss)=corr(dpt_branch1',dpt_branch2','type','kendall')- corr(dpt_branchr1',dpt_branchr2','type','kendall');
%    corrK(ss)=corr(dpt_branch1',dpt_branch2','type','spearman')-corr(dpt_branchr1',dpt_branchr2','type','spearman');
end 

corrK=pre_post_process.ksmooth(corrK,5);
[~,cut]=(max(corrK));

branch=indb{b3}(1:cut);

end

% figure; subplot(2,1,1); plot(corrK,'m','LineWidth',1.1)
% title([b1;b2;b3] ,'FontSize',15)
%         
% 
% subplot(2,1,2)
% scatter(dpt_branch{b1}(indb{b3}(1:n)),dpt_branch{b2}(indb{b3}(1:n)),20,dpt_branch{b3}(indb{b3}(1:n)))
% hold on
% scatter(dpt_branch{b1}(indb{b3}(1:cut)),dpt_branch{b2}(indb{b3}(1:cut)),10,'m')
