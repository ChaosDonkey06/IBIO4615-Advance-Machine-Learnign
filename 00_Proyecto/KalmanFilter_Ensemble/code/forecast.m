function forecast(id)
%1000 instances, id from 1 to 1000
load AH6state%Absolute humidity
load initialstate_state288%initialization
load C_state288%commuting
Nijs=C;
pop=100000;%used as the baseline population of observation
num_loc=size(C,1);%number of locations
tmstep=7;%per week
dt=1;%time step in model
num_times=floor(225/tmstep);%total number of weeks
ts=288;%start date
num_ens=300;%ensemble size
lambda=1.05;%inflation
discrete=0;%continous model
%%%%%%%%%%%%%%%%%%%%initial parameter range
Sl=0.3; Su=0.9;
Il=0.00001; Iu=0.001;
Ll=200; Lu=500;
Dl=2; Du=7;
R0mxl=1.3; R0mxu=4;
R0mnl=0.8; R0mnu=1.3;
ql=0; qu=0.05;
xmin=[];xmax=[];
for l=1:num_loc*num_loc
    xmin=[xmin,Sl,Il,0];
    xmax=[xmax,Su,Iu,0];
end
xmin=[xmin,R0mxl,R0mnl,Ll,Dl,ql];
xmax=[xmax,R0mxu,R0mnu,Lu,Du,qu];

%%%prepare the commute network
Nij=Nijs(:,:,id);
%%%% prepare initial condition and parameters
x0=initialstate(:,id);
%%%%  Running the truth
[n,~]=size(x0);
xr=zeros(n,num_times+1);
xr(:,1)=x0;
obsr=zeros(num_loc,num_times+1);
tcnt=ts;
for t=1:num_times
    [xtemp,obstemp]=mixnetworkmodelp(xr(:,t),tcnt,dt,tmstep,Nij,AH,discrete);
    xr(:,t+1)=xtemp;
    obsr(:,t+1)=obstemp;
    tcnt=tcnt+tmstep;
end
% plot(obsr');
% hold on
%%%%%%%%% add noise
obs=zeros(num_loc,num_times+1);
oev=zeros(num_loc,num_times+1);
for l=1:num_loc
    for t=1:num_times+1
        ave=obsr(l,max(1,t-1));
        ave=ave+obsr(l,max(1,t-2));
        ave=ave+obsr(l,max(1,t-3));
        ave=ave/3;    
        oev(l,t)=1e4+ave^2/5;
        obs_sqrt=sqrt(oev(l,t));
        obs(l,t)=max(0,obsr(l,t)+randn(1,1)*obs_sqrt);
    end
end
% plot(1:num_times+1,obs,'x');

forecastens=NaN(num_ens,num_times,num_loc,num_times);%forecast ensembles
xpostens=NaN(n,num_ens,num_times,num_times);%posterior ensembles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% start EAKF of network model
%%%initialization
x=initialization_m(Nij,xmin,xmax,num_ens);
%%%run for one week
tcnt=ts;
for t=1:1 %1 weeks
    [x,obs_ens]=mixnetworkmodelp(x,tcnt,dt,tmstep,Nij,AH,discrete);
    tcnt=tcnt+tmstep;
end
%%%% Begin looping through observations
xprior=NaN(n,num_ens,num_times);
xpost=xprior;
obsprior=NaN(num_loc,num_ens,num_times);
obspost=NaN(num_loc,num_ens,num_times);
tcnt=ts;
for tt = 1:num_times%assimilate from week 1
    tt
    %%% inflation of x before assimilation
    x=mean(x,2)*ones(1,num_ens)+lambda*(x-mean(x,2)*ones(1,num_ens));
    obs_ens=mean(obs_ens,2)*ones(1,num_ens)+lambda*(obs_ens-mean(obs_ens,2)*ones(1,num_ens));
    %%%%save prior
    xprior(:,:,tt)=x;
    obsprior(:,:,tt)=obs_ens;
    %loop through local observations
    for l=1:num_loc
        %%%%%  Get the variance of the ensemble
        obs_var = oev(l,tt);
        prior_var = var(obs_ens(l,:));
        post_var = prior_var*obs_var/(prior_var+obs_var);
        if prior_var==0
            post_var=0;
            prior_var=1e-3;
        end
        prior_mean = mean(obs_ens(l,:));
        post_mean = post_var*(prior_mean/prior_var + obs(l,tt)/obs_var);
        %%%% Compute alpha and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);
        %%%  Getting the covariance of the prior state space and
        %%%  observations  (which could be part of state space, e.g. infections)
        %%%  Loop over each state variable
        rr=zeros(1,size(x,1));
        for j=1:size(x,1)
            A=cov(x(j,:),obs_ens(l,:));
            rr(j)=A(2,1)/prior_var;
        end
        dx=rr'*dy;
        %%%  Get the adjusted ensemble and obs_ens
        x = x + dx;
        obs_ens(l,:)=obs_ens(l,:)+dy;
        obs_ens(l,obs_ens(l,:)<0)=0;
        %%%  Corrections to DA produced aphysicalities
        x = checkbound(x,Nij);
        x(size(x,1),x(size(x,1),:)<0)=1e-3;
    end
    xnew = x;
    %%%%%%%%%%%%%
    xpost(:,:,tt)=xnew;
    xpostens(:,:,:,tt)=xpost;
    obspost(:,:,tt)=obs_ens;
    %%%  Integrate forward one time step
    [x,obs_ens]=mixnetworkmodelp(xnew,tcnt,dt,tmstep,Nij,AH,discrete);
    tcnt=tcnt+tmstep;
    %forecast
    xpred=NaN(n,num_ens,num_times);
    obspred=NaN(num_loc,num_ens,num_times);
    %assign previous observation as true observation
    for t=1:tt
        obspred(:,:,t)=obs(:,t)*ones(1,num_ens);
%             obspred(:,:,t)=obspost(:,:,t);
        xpred(:,:,t)=xpost(:,:,t);
    end
    %run forward to make forecast
    tpred=ts+(tt-1)*tmstep;
    for t=tt:num_times-1
        [xpred(:,:,t+1),obspred(:,:,t+1)]=mixnetworkmodelp(xpred(:,:,t),tpred,dt,tmstep,Nij,AH,discrete);
        xpred(:,:,t+1) = checkbound(xpred(:,:,t+1),Nij);
        tpred=tpred+tmstep;
    end
    obspred=shiftdim(obspred,1);
    forecastens(:,:,:,tt)=obspred;
end

save(['forecast',num2str(id)],'forecastens','xpostens','obs')

function x = checkbound(x,Nij)
num_loc=length(Nij);
num_var=3;
%R0max
Rxidx=num_loc*num_loc*num_var+1;
%R0min
Rnidx=Rxidx+1;
%L
Lidx=Rnidx+1;
%D
Didx=Lidx+1;
%q
qidx=Didx+1;
for i=1:num_loc
    for j=1:num_loc
        temp=idx(i,j,num_loc)+1;%S
        x(temp,x(temp,:)<0)=mean(x(temp,:));
        x(temp,x(temp,:)>Nij(i,j))=Nij(i,j)-1;
        temp=temp+1;%I
        x(temp,x(temp,:)<0)=mean(x(temp,:));
        x(temp,x(temp,:)>Nij(i,j))=median(x(temp,:));
        temp=temp+1;%incidence
        x(temp,x(temp,:)<0)=mean(x(temp,:));
        x(temp,x(temp,:)>Nij(i,j))=mean(x(temp,:));
    end
end
x(Rxidx,x(Rxidx,:)>4)=4;
x(Rnidx,x(Rnidx,:)>1.3)=1.3;
%if R0max < R0min
x(Rxidx,x(Rxidx,:)<0)=mean(x(Rxidx,:));
x(Rnidx,x(Rnidx,:)<0)=mean(x(Rnidx,:));
x(Rxidx,x(Rxidx,:)<x(Rnidx,:))=x(Rnidx,x(Rxidx,:)<x(Rnidx,:))+0.01;
%L
x(Lidx,x(Lidx,:)>500)=500;
x(Lidx,x(Lidx,:)<0)=mean(x(Lidx,:));
x(Lidx,x(Lidx,:)<200)=median(x(Lidx,:));
%D
x(Didx,x(Didx,:)>12)=12;
x(Didx,x(Didx,:)<0)=mean(x(Didx,:));
x(Didx,x(Didx,:)<0.5)=median(x(Didx,:));
if num_loc>1
    %q
    x(qidx,x(qidx,:)<0)=0;
    x(qidx,x(qidx,:)>1)=1;
end

function a=idx(i,j,num_loc)
%find the index of subpopulation (i,j)
num_var=3;
a=(i-1)*num_var*num_loc+(j-1)*num_var;