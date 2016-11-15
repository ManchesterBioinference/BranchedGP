
function dpt=dpt_to_root(M,s)
% Finds dpt distance of all cells to the root cell

%s root cell(s)
%dpt=dpt distance to s 
%M= accumulated transition matrix

n=size(M,1);
dpt=zeros(1,n);
%M2=M^2;
if length(s)>1 %if more than one root cell has been specified
    fs=zeros(1,n);
    fs(s)=1/sqrt(length(s));
    for x=1:n
        fx=zeros(1,n);
        fx(x)=1;
        D2=(fs*M-fx*M).^2;  
        %D2=fs*M2*fs'+ fx*M2*fx' - fs*M2*fx'- fx*M2*fs' ;
        dpt(x)=sqrt(sum( D2 ));
    end

else %if more only one root cell has been specified    
    for x=1:n
        D2=(M(s,:)-M(x,:)).^2;  %D2=M(s,:)*M(:,s)+M(x,:)*M(:,x)-M(s,:)*M(:,x)-M(x,:)*M(:,s);
        %D2_=M(s,s)+M(x,x)-M(s,x)-M(x,s);
        dpt(x)=sqrt(sum( D2 ));

    end
end


