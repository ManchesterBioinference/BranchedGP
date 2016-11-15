function [T, phi0] = transition_matrix(data, method, param)
% Builds Transition matrix T by the selected method
% 'nn' , 'loc' or 'classic' correspondingly for nearest neighbours, local
% kernel width or the classic version.

if strcmp(method,'classic')
    sigma=param;
    [T,~, phi0] = diffusionmap.T_classic(data,data,sigma);   

elseif strcmp(method,'loc') 
    nsig=param;
    [T,~, phi0] = diffusionmap.T_loc(data,data,nsig);

elseif strcmp(method,'nn')
    k=param(1);
    nsig=param(2);
    %[T, phi0] = diffusionmap.T_nn(data,k,nsig);
    [T,~, phi0] = diffusionmap.T_nn(data,data,k,nsig);
else
    error('unknown diffusionmap method %s', method)
end
