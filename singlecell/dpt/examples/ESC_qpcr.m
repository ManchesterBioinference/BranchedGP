%This piece of code regenerates Figure1 b to c in the paper. it takes about 1.5 minutes on a normal PC to run
addpath(fileparts(fileparts(mfilename('fullpath'))));

%step 1: load data and (if applicable) cell labels and (if applicable) do preprocessing  
load ESC_qpcr_Goettgens.mat; 
%LOD=-14; k=20; data=pre_post_process.lam(data,LOD,k); %preprocessing
%data=pre_post_process.cosine_dist_normalisation(data);

%step 2: bulid the transition matrix and its first (noninformative) left
%eigenvector using one of the three methods 'classic', 'loc', 'nn'
%[T, phi0]=transition_matrix(data,'nn',[50,10]);
%[T, phi0]=transition_matrix(data,'classic',10^3);
nsig=10; [T,phi0] = transition_matrix(data,'loc',nsig);

%step 3: calculate accumulated transition matrix M 
%and input and organize the parameters to find tip(s) of branch(es) 
root=[533];% 1307,
branching=1; %detect branching? 0/1
[M, tips] = dpt_input(T, phi0, branching, 'maxdptdist', root);
%num_tips=2; [M, tips] = dpt_input(T, phi0, branching, 'manual',num_tips, labels); %manual selection of root and tip cells
%step 4: do pseudotime ordering and separate the branches
[Branch,DPT]=dpt_analyse(M,branching,tips);
%%%%%%plotting results%%%%%%%%%%%%%%%%%%%%
[phi, lambda] = diffusionmap.eig_decompose_normalized(T,10);
B1=[find(Branch==1);find(Branch==2);find(Branch==0)];%[branch{1},branch{2}];
B2=[find(Branch==1);find(Branch==3);find(Branch==0)];%[branch{1},branch{3},unassigned];
ESCcolors=[0, 126, 191; 54, 180, 79; 255, 172, 56; 255, 0, 44;135, 79, 157]/255;
figure
phi(:,3)=-phi(:,3); %rotate for nicer view
subplot(1,2,1)
colormap copper
scatter(phi(:,2),phi(:,3),5,'b','fill')
hold on
scatter(phi(B1,2),phi(B1,3),20,DPT(B1),'fill')
pre_post_process.plot_average_path(DPT,B1,phi(:,2:3))

subplot(1,2,2)
colormap copper
scatter(phi(:,2),phi(:,3),5,'b','fill')
hold on
scatter(phi(B2,2),phi(B2,3),20,DPT(B2),'fill')
pre_post_process.plot_average_path(DPT,B2,phi(:,2:3))

%%%%%%instead of the 4 steps above, it is possible to simply run the
%%%%%%following three lines for automated branch identification. By this
%you find the branchings in the data but might include
%false positive (too finegrain) branches as well depending on gstatmin
%
%gstatmin=1.01; nsig=10; [I,DPT,phi]=test_autotree(data,'loc',[nsig],root,gstatmin);
%figure; scatter(phi(:,2),phi(:,3),50,I,'fill'); colormap lines
%figure; scatter(phi(:,2),phi(:,3),50,DPT,'fill') colormap lines

%%%%%%%%%%%%plot expression of Erg and Ikaros
data(data<-11)=-11;

[valT,indT]=sort(DPT);

m1=size(B1,1);
m2=size(B2,1);
smoothL=400;
B1smooth50=zeros(m1-smoothL,42);
B2smooth50=zeros(m2-smoothL,42);

[valT1,indT1]=sort(DPT(B1));
[valT2,indT2]=sort(DPT(B2));


erg_and_ikaros=[find(strcmp('Erg',Genes_analysed)),find(strcmp('Ikaros',Genes_analysed))];

%%%%%%%%%%%%% Alexis
csvwrite('data/ESC_qpcr_Goettgens_data.csv', data)
csvwrite('data/ESC_qpcr_Goettgens_DPT.csv', DPT)  % pseudotime
csvwrite('data/ESC_qpcr_Goettgens_Branch.csv', Branch)
csvwrite('data/ESC_qpcr_Goettgens_Genes_analysed.csv', Genes_analysed.T)
sum(Branch==0)+sum(Branch==1)+sum(Branch==2)+sum(Branch==3)

ds = mat2dataset(data, 'VarNames', Genes_analysed);
export(ds, 'file', 'data/ESC_qpcr_Goettgens_dataDS.csv', 'Delimiter', ',')
%%%%%%%%%%%%%%%%%plot in DPT order

for thetwogenes=1:2
    figure;
    g=erg_and_ikaros(thetwogenes);

    B=B1(indT1);
    B1_g=pre_post_process.ksmooth(data(B,g),smoothL);
    scatter(1:m1,data(B,g),20,ESCcolors(labels(B),:),'fill')

    ylabel(Genes_analysed(g),'FontSize',20)
    hold on
 
    B=B2(indT2);
    B2_g=pre_post_process.ksmooth(data(B,g),smoothL);
    scatter(1:m2,data(B,g),20,ESCcolors(labels(B),:),'fill')

    xlim([0 4000])
    xlabel('order in DPT','FontSize',20);
    ylabel(Genes_analysed(g),'FontSize',20)
    ylim([min(data(:,g)) max(data(:,g))])
       
    plot(ceil(smoothL/2):ceil(m2-smoothL/2)-1,B2_g,'k','LineWidth',2)
    plot(ceil(smoothL/2):ceil(m1-smoothL/2)-1,B1_g,'k','LineWidth',2)
    
    set(gca,'FontSize',30)
    
end
% %%%%%%%%%%%%%%%%%plot in DPT
% 
% for thetwogenes=1:2
%     figure;
%     g=erg_and_ikaros(thetwogenes);
% 
%     B=B1(indT1);
%     %m1=size(B,1);
%     B1_g=pre_post_process.ksmooth(data(B,g),smoothL);
%     %scatter(1:m1,data(B,g),20,ESCcolors(labels(B),:),'fill')
%     scatter(valT1,data(B,g),20,ESCcolors(labels(B),:),'fill')
% 
%     ylabel(Genes_analysed(g),'FontSize',20)
%     %ylim([-20 max(data(B1,g))])
%     hold on
%  
%     B=B2(indT2);
%     %m2=size(B,1);
%     B2_g=pre_post_process.ksmooth(data(B,g),smoothL);
%     scatter(valT2,data(B,g),20,ESCcolors(labels(B),:),'fill')
% 
%     %xlim([0 4000])
%     %xlim([0 6])
%     xlabel('order in DPT','FontSize',20);
%     ylabel(Genes_analysed(g),'FontSize',20)
%     ylim([min(data(:,g)) max(data(:,g))])
%        
%     plot(valT2(ceil(smoothL/2):ceil(m2-smoothL/2)-1),B2_g,'k','LineWidth',2)
%     plot(valT1(ceil(smoothL/2):ceil(m1-smoothL/2)-1),B1_g,'k','LineWidth',2)
%     
%     set(gca,'FontSize',30)
%     
% end
