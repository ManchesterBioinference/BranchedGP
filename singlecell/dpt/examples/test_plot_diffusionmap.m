addpath(fileparts(fileparts(mfilename('fullpath'))));
load toydata3B.mat; data=toydata3B; %labels=data(:,3);
%data=pre_post_process.cosine_dist_normalisation(data);

method='nn'; % or 'loc' or 'classic' 
k=50; nsig=20; sigma=200;
methparam=[k,nsig]; %or nsig or sigma
l=4; %no. DM components to compute
[phi,lambda]=diffusionmap.diffusionmap(data,method,methparam,l);
scatter(phi(:,2),phi(:,3),20,'fill');