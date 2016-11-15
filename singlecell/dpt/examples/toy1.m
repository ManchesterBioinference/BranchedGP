%This piece of code generates suplementary figure 4.
%On a normal PC takes about 3 seconds
addpath(fileparts(fileparts(mfilename('fullpath'))));

%step 1: load data and (if applicable) cell labels and (if applicable) do preprocessing  
load toydata3B.mat; data=toydata3B; %labels=data(:,3);
%LOD=1000; k=20; data=pre_post_process.lam(data,LOD,k); 
%data=pre_post_process.cosine_dist_normalisation(data); %preprocess, not
%needed in this example

%step 2: bulid the transition matrix and its first (noninformative) left
%eigenvector using one of the three methods 'classic', 'loc', 'nn'
sigma=200; [T,phi0] = transition_matrix(data,'classic',sigma);
%[T, phi0]=transition_matrix(data,'nn', [30,10]);

%two other possibilities:
%nsig=50; [T,phi0] = transition_matrix(data,'loc',nsig);
% sigma=1000; k=20; [T,phi0] = transition_matrix(data,'nn',[k,sigma]);  

%step 3: calculate accumulated transition matrix M 
%and input and organize the parameters to find tip(s) of branch(es) 
root=[937]; %index of root cell(s)
branching=1; %detect branching? 0/1
[M, tips] = dpt_input(T, phi0, branching, 'maxdptdist', root);
% two other possibilities:
%[M, tips] = dpt_input(T, phi0, branching, 'maxdptdist'); % root unknown
%num_tips=2; [M, tips] = dpt_input(T, phi0, branching, 'manual',...
%num_tips);%, labels); %manual selection of root and tip cells

%step 4: do pseudotime ordering and separate the branches
[Branch,DPT]=dpt_analyse(M,branching,tips);
%%%%%%plotting results%%%%%%%%%%%%%%%%%%%%
%plots on diffusion map

[phi, lambda] = diffusionmap.eig_decompose_normalized(T,4);

%BrcolorOrder='kbmr';
BrcolorOrder=[0 0 0;
               0 0 1;
               1 0 1;
               1 0 0];
figure
subplot(1,2,1)
colormap jet
scatter(phi(:,2),phi(:,3),20,DPT,'fill');
title ('color by DPT')
subplot(1,2,2)
scatter(phi(:,2),phi(:,3),20,BrcolorOrder(Branch+1,:),'fill');
title ('color by branches')

% plot pseudotemporal expression dynamics
genesColOrder=[0         0    1.0000;
         0    0.5000         0;
         0.7500         0    0.7500;
        0.7500    0.7500         0;
        1.0000         0         0;
        0    0.7500    0.7500;
        0.2500    0.2500    0.2500];

figure;
for i=1:6
    scatter(DPT,data(:,i),20,genesColOrder(i,:),'fill')
    hold on
end

