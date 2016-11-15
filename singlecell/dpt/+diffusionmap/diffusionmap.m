function [phi,lambda]=diffusionmap(data,method,methparam,l)
% computes l first diffusion map components of data by the one of thre three methods as below and
% the respective parameters (methparam).

% 'classic': sigma=methparam;
% 'loc': locally scaled, nsig=methparam;
% 'nn' : nearest neighboures, k=methparam(1); sigma=methparam(2);

import diffusionmap.T_classic;
import diffusionmap.T_loc;
import diffusionmap.T_nn;
import diffusionmap.eig_decompose_normalized;


if strcmp(method,'classic')
    sigma=methparam;
    [T, ~] = T_classic(data,data,sigma);   

elseif strcmp(method,'loc') 
    nsig=methparam;
    [T, ~] = T_loc(data,data,nsig);

elseif strcmp(method,'nn')
    k=methparam(1);
    sigma=methparam(2);
    [T, ~] = T_nn(data,data,k,sigma);
else
    error('unknown diffusionmap method %s', method)
end

[ phi, lambda ] = eig_decompose_normalized(T,l);