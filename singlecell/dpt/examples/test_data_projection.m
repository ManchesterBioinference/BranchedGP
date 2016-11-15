 addpath(fileparts(fileparts(mfilename('fullpath'))));
 load toydata3B.mat; data=toydata3B; 

k=50; nsig=10; sigma=200;
data1=data(1:400,:);
data2=data(100:200,:);
[T1,T2,phi0] = diffusionmap.T_classic(data1,data2,sigma);
%[T1,T2,phi0] = diffusionmap.T_loc(data1,data2,nsig);
%[T1,T2,phi0] = diffusionmap.T_nn(data1,data2,k,nsig);

[phi1,phi2]=diffusionmap.data_projection(T1,T2);

figure;
scatter3(phi1(:,2),phi1(:,3),phi1(:,4),50,'fill')
hold on
scatter3(phi2(:,2),phi2(:,3),phi2(:,4),50,'r','fill')