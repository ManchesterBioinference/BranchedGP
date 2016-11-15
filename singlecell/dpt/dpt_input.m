function [M, tips] = dpt_input(T, phi0, branching, method, numtips_or_rootindex, labels)
% first several parameters (tipsel_params) used in suggest_tips.m are organized here.
% then dpt_input builds accumulated transition matrix M and tip(s) of branch(es).
% two types of usage is possible:
%1) [M, tips] = dpt_input(T, phi0, branching, 'maxdptdist', rootindex)%
%2) [M, tips] = dpt_input(T, phi0, branching, 'manual', numtips, labels) ,providing labels is optional. 
import dpt.suggest_tips;

n = size(T, 1);
%%%%for M in full dimensions
M = (eye(n) - T + phi0 * phi0' )^-1 - eye(n);
% %%%%for low-dim approximation of M
% l=20;
% [phi, lambda] = diffusionmap.eig_decompose_normalized(T,l);
% M=dpt.low_dim_M(phi,lambda,l);

%3%%%%%%%%find tips
if branching
    num_branches = 3;
else
    num_branches = 1;
end
    
if strcmp(method,'maxdptdist')
    tipsel_params{1}=M;
    tipsel_params{2}=phi0;
    if nargin > 4
        root = numtips_or_rootindex;
        tipsel_params{3}=root;
    end
elseif strcmp(method,'manual')  
    num_tips = numtips_or_rootindex;
    phi = diffusionmap.eig_decompose_normalized(T,10);
    tipsel_params{1}=[phi(:,2),phi(:,3)];
    tipsel_params{2}=num_tips; %num of branch tip cells to select
    if nargin>5
        tipsel_params{3}=labels;
    else
        tipsel_params{3}=ones(n,1);
    end
    
else
    error('unknown tip selection method %s', method)
end

tips = suggest_tips(method,tipsel_params,num_branches);

end
%4%%%%%%%%%%%%%%%%%%
