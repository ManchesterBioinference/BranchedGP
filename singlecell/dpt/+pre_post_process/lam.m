
function dat = lam(data,LOD,k) 
%Local Approximation of Missing values (e.g. Limit Of Detection), 
%for data preprocessing, replaces the missing values in data (LOD) by the
%mean of each cell's k nearest neighbours

%dat processed (approximated) data
%LOD threshold of detection for each gene (can be a vector with length G (number of genes) or a single
%value for all genes)
%k number of nearest neighbour used for local approximation

n=size(data,1);
dat=data;
id=knnsearch(data,data,'k',k);
id(:,1)=[];
for g=1:size(data,2)
  if length(LOD)==size(data,2)
    thr=LOD(g);
  else
    thr=LOD;
  end  
  for i=1:n
    if data(i,g)==thr
        dat(i,g)=mean(data(id(i,:),g));
        %dat(i,g)=median(data(id(i,:),g));
    end
  end

end
