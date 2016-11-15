% this piece of code produces the dpt pseudotime and branching for
% nonlinearly transformed toy data as in Suppl. Fig 14

%step 1: load data and (if applicable) cell labels and (if applicable) do preprocessing  
load toydata3B.mat; data0=toydata3B; %labels=sum(data0,2);
%%%%%%%%%%%%%%%%
n=size(data0,1);
dat=data0;
dat=log(dat+1);
d=size(dat,2);
%%%%%%%%%%%%%%%%%%%nonlinear transformation of toy data
dat=dat-repmat(min(dat),n,1);
range_dat=repmat(( max(dat)-min(dat) ),n,1);
range_dat(range_dat==0)=1;
dat=dat./range_dat;
data=dat';
hG=3;
rng(1);
randM=randn(hG,d);
data=( randM * (data) )';%+randn(hG,n) )';
data=(1+exp(-data)).^(-1);
data=log(data-min(data(:))+1);
%%%%%%%%%%%%%%%%%%%%%%
[U,S,V] = svd(data,'econ');

root=937;
nsig=200;
[I,DPT,phi]=test_autotree(data,'loc',[nsig],root,1.01);

%[T, phi0]=transition_matrix(data,'nn', [200,50]);
% [T, phi0]=transition_matrix(data,'loc', 200);
% [phi, lambda] = diffusionmap.eig_decompose_normalized(T,4);
% root=937;
% [M, tips] = dpt_input(T, phi0, 1, 'maxdptdist',root);
% [Branch,DPT]=dpt_analyse(M,1,tips);

%save artdata_toy.mat data
%%%%%%%%%save artdata.mat data0 dat randM data tips
figure; scatter(U(:,1),U(:,2),50,DPT,'fill');
colormap jet

figure; scatter(U(:,1),U(:,2),50,I,'fill');
colormap jet

set(gca,'xtick',[])
set(gca,'ytick',[])
xlabel('IC1')
ylabel('IC2')
