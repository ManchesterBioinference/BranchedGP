%This piece of code provides a demonstration of obtaining universal time 
%for a time-lapse data (toy toggle-switch) and generates supplementary figure 1. 
%This provides a comparison of expression versus actual time to versus universal time.
%On a normal PC takes about 5 seconds to run.

load traj_several.mat
i=1; %12;
celli=(severalTraj{i})';
figure;

%G1=fliplr(celli(:,1)');
%G2=fliplr(celli(:,2)');
%rainbowplot(G1,G2);
%colormap jet; scatter(celli(:,1),celli(:,2),20,1:size(celli,1),'fill'); hold on
plot(celli(:,1),celli(:,2));
xlabel('G1','FontSize',20)
ylabel('G2','FontSize',20)
title (['G1 G2 trajectory of cell ' num2str(i) ],'FontSize',20)


count=0;
for c=1:100
    if ~isempty(severalTraj{c})

        count=count+1;
        traj{count}=severalTraj{c};
        acttime{count}=Otime{c};

         %%%%%%%%%%%%%%%%%%%%%%%%Density estimate D%%%%
        sigma=70;
        data=traj{count}';
        [n,G]=size(data);

        d2=zeros(n,n);
        for g=1:G 
            d2_g = (bsxfun(@minus,reshape(data(:,g),n,1),reshape(data(:,g),1,n))).^2;  
            d2=d2+d2_g;
        end
        W=exp(-d2/(2*sigma^2)); %Gaussian kernel for density estimation
        D=nansum(W);

        %sumD=sum(D);
        accumT=0;
        UT=zeros(n,1);
        for i=1:n
            accumT=accumT+1/D(i);%/sum(D);
            UT(i)=accumT;
        end
        UT=UT/max(UT);
        unitime{count}=UT;

    end
end

%%%%%%%%plotting results
ColOrder=[0         0    1.0000;
         0    0.5000         0;
    1.0000         0         0;
         0    0.7500    0.7500;
    0.7500         0    0.7500;
    0.7500    0.7500         0;
    0.2500    0.2500    0.2500];
set(groot,'defaultAxesColorOrder',ColOrder)

figure
subplot(2,1,1)
for c=1:72
    plot(acttime{c},traj{c}(1,:),'Color',ColOrder(1,:))
    hold on 
    plot(acttime{c},traj{c}(2,:),'Color',ColOrder(2,:))
end
xlabel('actual time', 'FontSize',20)
ylabel('expression ', 'FontSize',20)
xlim([0 2200])

subplot(2,1,2)
for c=1:72
    plot(unitime{c},traj{c}(1,:),'Color',ColOrder(1,:))
    hold on 
    plot(unitime{c},traj{c}(2,:),'Color',ColOrder(2,:))
end
xlabel('universal time', 'FontSize',20)
ylabel('expression ', 'FontSize',20)
 