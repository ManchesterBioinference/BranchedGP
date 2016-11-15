function S=manual_select_cell(D1,D2,m,labels)
% Manually select tip cell(s) of a branch 
% D1 and D2 are plotting coordinates (e.g. phi(:,2) and phi(:,3) )
% m is the inquired number of cells to be selected
% labels (of cells) is used for colouring the diffusion map
figure
scatter(D1,D2,20,labels), title(['Select ' num2str(m) 'cells at each tip. Start from the root branch tip.'])
colormap jet;
V=[D1,D2];

[xX,yY]   = ginput(m);
S = [];

for i = 1:m
    for j = 1:size(V,1)
        NorP(j) = norm(V(j,[1 2])-[xX(i) yY(i)]);
    end
    [~, aind] = min(NorP);
    S = [S aind];
end

close