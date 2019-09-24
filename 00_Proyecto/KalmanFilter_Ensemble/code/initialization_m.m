function x=initialization_m(Nij,xmin,xmax,num_ens)
num_loc=length(Nij);
x=lhsu(xmin,xmax,num_ens);
x=x';
for i=1:num_loc
    for j=1:num_loc
        x(idx(i,j,num_loc)+1,:)=round(x(idx(i,j,num_loc)+1,:)*Nij(i,j));%S
        x(idx(i,j,num_loc)+2,:)=round(x(idx(i,j,num_loc)+2,:)*Nij(i,j));%I
    end
end
            


function a=idx(i,j,num_loc)
num_var=3;
a=(i-1)*num_var*num_loc+(j-1)*num_var;


function s=lhsu(xmin,xmax,nsample)
% s=lhsu(xmin,xmax,nsample)
% LHS from uniform distribution
% Input:
%   xmin    : min of data (1,nvar)
%   xmax    : max of data (1,nvar)
%   nsample : no. of samples
% Output:
%   s       : random sample (nsample,nvar)
%   Budiman (2003)

nvar=length(xmin);
ran=rand(nsample,nvar);
s=zeros(nsample,nvar);
for j=1: nvar
   idx=randperm(nsample);
   P =(idx'-ran(:,j))/nsample;
   s(:,j) = xmin(j) + P.* (xmax(j)-xmin(j));
end