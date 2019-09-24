function [x,obs]=mixnetworkmodelp(x,ts,dt,tmstep,Nij,AH,discrete)
%x:state vector,num_var:number of variables for each location,num_loc:
%number of locations,ts:integration start time,dt:minimal integration
%timestep,tmstep:total integration timestep
%Nij(i,j): i<-j, live in j and work at i
%discrete=0: continuous, 1: stochastic
Ntotal=sum(sum(Nij));%total population
Ctotal=Ntotal-trace(Nij);%total commuters
Nave=(Nij+Nij')/2;%average commuting flux
Cave=zeros(length(Nij),1);%average commuting in each location
for i=1:length(Nij)
    Cave(i)=sum(Nave(:,i))-Nave(i,i);
end
%continuous
num_var=3;
num_loc=length(Nij);
[~,num_ens]=size(x);
N=sum(Nij,2);%day time population
N1=sum(Nij,1);%night time population
AH(length(AH)+1:2*length(AH),:)=AH;
%R0max
Rxidx=idx(num_loc,num_loc,num_loc)+num_var+1;
%R0min
Rnidx=Rxidx+1;
%L
Lidx=Rnidx+1;
%D
Didx=Lidx+1;
%q
qidx=Didx+1;
%prepare BETA
BT1=zeros(length(AH),num_ens,num_loc);
BETA=zeros(length(AH),num_ens,num_loc);
for i=1:num_loc
    for j=1:num_ens
        Rx=x(Rxidx,j);Rn=x(Rnidx,j);D=x(Didx,j);
        b=log(Rx-Rn); a=-180;
        BT1(:,j,i)=exp(a*AH(:,i)+b)+Rn;
        BETA(:,j,i)=BT1(:,j,i)/D;
    end
end
%transform format from state space vector to matrix
I=zeros(num_loc*num_loc,num_ens,abs(tmstep)+1);
S=zeros(num_loc*num_loc,num_ens,abs(tmstep)+1);
Incidence=zeros(num_loc*num_loc,num_ens,abs(tmstep)+1);
obs=zeros(num_loc,num_ens);
for i=1:num_loc
    for j=1:num_loc
        I((i-1)*num_loc+j,:,1)=x(idx(i,j,num_loc)+2,:);
        S((i-1)*num_loc+j,:,1)=x(idx(i,j,num_loc)+1,:);
    end
end
L=x(Lidx,:);
D=x(Didx,:);
q=x(qidx,:);
sk1=zeros(num_loc*num_loc,num_ens);
ik1=sk1;ik1i=sk1;
sk2=sk1;ik2=sk1;ik2i=sk1;
sk3=sk1;ik3=sk1;ik3i=sk1;
sk4=sk1;ik4=sk1;ik4i=sk1;
%start integration
tcnt=0;
for t=ts+dt:dt:ts+tmstep
    tcnt=tcnt+1;
    %daytime transmission
    dt1=dt/3;
    %first step
    daytimeI=zeros(num_loc,num_ens);
    daytimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            daytimeI(i,:)=daytimeI(i,:)+I((i-1)*num_loc+k,:,tcnt);
            daytimeS(i,:)=daytimeS(i,:)+S((i-1)*num_loc+k,:,tcnt);
        end
    end
    dayIincome=zeros(num_loc,num_ens);
    daySincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                dayIincome(i,:)=dayIincome(i,:)+Nave(i,k).*daytimeI(k,:)/N(k);
                daySincome(i,:)=daySincome(i,:)+Nave(i,k).*daytimeS(k,:)/N(k);
            end
        end
    end
    for i=1:num_loc
        temp=daytimeI(i,:);
        for j=1:num_loc
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-S((i-1)*num_loc+j,:,tcnt)-I((i-1)*num_loc+j,:,tcnt))./L);
            Einf=min(dt1*(BETA(t,:,i).*S((i-1)*num_loc+j,:,tcnt).*temp/N(i)),S((i-1)*num_loc+j,:,tcnt));
            Erecov=min(dt1*(I((i-1)*num_loc+j,:,tcnt)./D),I((i-1)*num_loc+j,:,tcnt));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*I((i-1)*num_loc+j,:,tcnt);
            EIleft=min(temp1,I((i-1)*num_loc+j,:,tcnt));
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*S((i-1)*num_loc+j,:,tcnt);
            ESleft=min(temp1,S((i-1)*num_loc+j,:,tcnt));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*dayIincome(i,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*daySincome(i,:);
            if discrete==0
                sk1((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik1((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik1i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk1((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik1((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik1i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    %second step
    Ts1=S(:,:,tcnt)+round(sk1/2);
    Ti1=I(:,:,tcnt)+round(ik1/2);
    daytimeI=zeros(num_loc,num_ens);
    daytimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            daytimeI(i,:)=daytimeI(i,:)+Ti1((i-1)*num_loc+k,:);
            daytimeS(i,:)=daytimeS(i,:)+Ts1((i-1)*num_loc+k,:);
        end
    end
    dayIincome=zeros(num_loc,num_ens);
    daySincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                dayIincome(i,:)=dayIincome(i,:)+Nave(i,k).*daytimeI(k,:)/N(k);
                daySincome(i,:)=daySincome(i,:)+Nave(i,k).*daytimeS(k,:)/N(k);
            end
        end
    end
    for i=1:num_loc
        temp=daytimeI(i,:);
        for j=1:num_loc
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-Ts1((i-1)*num_loc+j,:)-Ti1((i-1)*num_loc+j,:))./L);
            Einf=min(dt1*(BETA(t,:,i).*Ts1((i-1)*num_loc+j,:).*temp/N(i)),Ts1((i-1)*num_loc+j,:));
            Erecov=min(dt1*(Ti1((i-1)*num_loc+j,:)./D),Ti1((i-1)*num_loc+j,:));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);  
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*Ti1((i-1)*num_loc+j,:);
            EIleft=min(temp1,Ti1((i-1)*num_loc+j,:));
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*Ts1((i-1)*num_loc+j,:);
            ESleft=min(temp1,Ts1((i-1)*num_loc+j,:));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*dayIincome(i,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*daySincome(i,:);
            if discrete==0
                sk2((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik2((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik2i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk2((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik2((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik2i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    %third step
    Ts2=S(:,:,tcnt)+round(sk2/2);
    Ti2=I(:,:,tcnt)+round(ik2/2);
    daytimeI=zeros(num_loc,num_ens);
    daytimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            daytimeI(i,:)=daytimeI(i,:)+Ti2((i-1)*num_loc+k,:);
            daytimeS(i,:)=daytimeS(i,:)+Ts2((i-1)*num_loc+k,:);
        end
    end
    dayIincome=zeros(num_loc,num_ens);
    daySincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                dayIincome(i,:)=dayIincome(i,:)+Nave(i,k).*daytimeI(k,:)/N(k);
                daySincome(i,:)=daySincome(i,:)+Nave(i,k).*daytimeS(k,:)/N(k);
            end
        end
    end
    for i=1:num_loc
        temp=daytimeI(i,:);
        for j=1:num_loc
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-Ts2((i-1)*num_loc+j,:)-Ti2((i-1)*num_loc+j,:))./L);
            Einf=min(dt1*(BETA(t,:,i).*Ts2((i-1)*num_loc+j,:).*temp/N(i)),Ts2((i-1)*num_loc+j,:));
            Erecov=min(dt1*(Ti2((i-1)*num_loc+j,:)./D),Ti2((i-1)*num_loc+j,:));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*Ti2((i-1)*num_loc+j,:);
            EIleft=min(temp1,Ti2((i-1)*num_loc+j,:));
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*Ts2((i-1)*num_loc+j,:);
            ESleft=min(temp1,Ts2((i-1)*num_loc+j,:));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*dayIincome(i,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*daySincome(i,:);
            if discrete==0
                sk3((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik3((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik3i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk3((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik3((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik3i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    %fourth step
    Ts3=S(:,:,tcnt)+round(sk3);
    Ti3=I(:,:,tcnt)+round(ik3);
    daytimeI=zeros(num_loc,num_ens);
    daytimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            daytimeI(i,:)=daytimeI(i,:)+Ti3((i-1)*num_loc+k,:);
            daytimeS(i,:)=daytimeS(i,:)+Ts3((i-1)*num_loc+k,:);
        end
    end
    dayIincome=zeros(num_loc,num_ens);
    daySincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                dayIincome(i,:)=dayIincome(i,:)+Nave(i,k).*daytimeI(k,:)/N(k);
                daySincome(i,:)=daySincome(i,:)+Nave(i,k).*daytimeS(k,:)/N(k);
            end
        end
    end
    for i=1:num_loc
        temp=daytimeI(i,:);
        for j=1:num_loc
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-Ts3((i-1)*num_loc+j,:)-Ti3((i-1)*num_loc+j,:))./L);
            Einf=min(dt1*(BETA(t,:,i).*Ts3((i-1)*num_loc+j,:).*temp/N(i)),Ts3((i-1)*num_loc+j,:));
            Erecov=min(dt1*(Ti3((i-1)*num_loc+j,:)./D),Ti3((i-1)*num_loc+j,:));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*Ti3((i-1)*num_loc+j,:);
            EIleft=min(temp1,Ti3((i-1)*num_loc+j,:));
            temp1=dt1*Ntotal/Ctotal*Cave(i)/N(i)*q.*Ts3((i-1)*num_loc+j,:);
            ESleft=min(temp1,Ts3((i-1)*num_loc+j,:));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*dayIincome(i,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N(i)*q.*daySincome(i,:);
            if discrete==0
                sk4((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik4((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik4i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk4((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik4((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik4i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    S(:,:,tcnt+1)=S(:,:,tcnt)+round(sk1/6+sk2/3+sk3/3+sk4/6);
    I(:,:,tcnt+1)=I(:,:,tcnt)+round(ik1/6+ik2/3+ik3/3+ik4/6);
    Incidence(:,:,tcnt+1)=round(ik1i/6+ik2i/3+ik3i/3+ik4i/6);
    %nighttime transmission
    dt1=2*dt/3;
    %first step
    nighttimeI=zeros(num_loc,num_ens);
    nighttimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            nighttimeI(i,:)=nighttimeI(i,:)+I((k-1)*num_loc+i,:,tcnt+1);
            nighttimeS(i,:)=nighttimeS(i,:)+S((k-1)*num_loc+i,:,tcnt+1);
        end
    end
    nightIincome=zeros(num_loc,num_ens);
    nightSincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                nightIincome(i,:)=nightIincome(i,:)+Nave(i,k).*nighttimeI(k,:)/N1(k);
                nightSincome(i,:)=nightSincome(i,:)+Nave(i,k).*nighttimeS(k,:)/N1(k);
            end
        end
    end
    for i=1:num_loc
        for j=1:num_loc
            temp=nighttimeI(j,:);
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-S((i-1)*num_loc+j,:,tcnt+1)-I((i-1)*num_loc+j,:,tcnt+1))./L);
            Einf=min(dt1*(BETA(t,:,j).*S((i-1)*num_loc+j,:,tcnt+1).*temp/N1(j)),S((i-1)*num_loc+j,:,tcnt+1));
            Erecov=min(dt1*(I((i-1)*num_loc+j,:,tcnt+1)./D),I((i-1)*num_loc+j,:,tcnt+1));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*I((i-1)*num_loc+j,:,tcnt+1);
            EIleft=min(temp1,I((i-1)*num_loc+j,:,tcnt+1));
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*S((i-1)*num_loc+j,:,tcnt+1);
            ESleft=min(temp1,S((i-1)*num_loc+j,:,tcnt+1));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightIincome(j,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightSincome(j,:);
            if discrete==0
                sk1((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik1((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik1i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk1((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik1((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik1i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    %second step
    Ts1=S(:,:,tcnt+1)+round(sk1/2);
    Ti1=I(:,:,tcnt+1)+round(ik1/2);
    nighttimeI=zeros(num_loc,num_ens);
    nighttimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            nighttimeI(i,:)=nighttimeI(i,:)+Ti1((k-1)*num_loc+i,:);
            nighttimeS(i,:)=nighttimeS(i,:)+Ts1((k-1)*num_loc+i,:);
        end
    end
    nightIincome=zeros(num_loc,num_ens);
    nightSincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                nightIincome(i,:)=nightIincome(i,:)+Nave(i,k).*nighttimeI(k,:)/N1(k);
                nightSincome(i,:)=nightSincome(i,:)+Nave(i,k).*nighttimeS(k,:)/N1(k);
            end
        end
    end
    for i=1:num_loc
        for j=1:num_loc
            temp=nighttimeI(j,:);
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-Ts1((i-1)*num_loc+j,:)-Ti1((i-1)*num_loc+j,:))./L);
            Einf=min(dt1*(BETA(t,:,j).*Ts1((i-1)*num_loc+j,:).*temp/N1(j)),Ts1((i-1)*num_loc+j,:));
            Erecov=min(dt1*(Ti1((i-1)*num_loc+j,:)./D),Ti1((i-1)*num_loc+j,:));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*Ti1((i-1)*num_loc+j,:);
            EIleft=min(temp1,Ti1((i-1)*num_loc+j,:));
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*Ts1((i-1)*num_loc+j,:);
            ESleft=min(temp1,Ts1((i-1)*num_loc+j,:));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightIincome(j,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightSincome(j,:);
            if discrete==0
                sk2((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik2((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik2i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk2((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik2((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik2i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    %third step
    Ts2=S(:,:,tcnt+1)+round(sk2/2);
    Ti2=I(:,:,tcnt+1)+round(ik2/2);
    nighttimeI=zeros(num_loc,num_ens);
    nighttimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            nighttimeI(i,:)=nighttimeI(i,:)+Ti2((k-1)*num_loc+i,:);
            nighttimeS(i,:)=nighttimeS(i,:)+Ts2((k-1)*num_loc+i,:);
        end
    end
    nightIincome=zeros(num_loc,num_ens);
    nightSincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                nightIincome(i,:)=nightIincome(i,:)+Nave(i,k).*nighttimeI(k,:)/N1(k);
                nightSincome(i,:)=nightSincome(i,:)+Nave(i,k).*nighttimeS(k,:)/N1(k);
            end
        end
    end
    for i=1:num_loc
        for j=1:num_loc
            temp=nighttimeI(j,:);
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-Ts2((i-1)*num_loc+j,:)-Ti2((i-1)*num_loc+j,:))./L);
            Einf=min(dt1*(BETA(t,:,j).*Ts2((i-1)*num_loc+j,:).*temp/N1(j)),Ts2((i-1)*num_loc+j,:));
            Erecov=min(dt1*(Ti2((i-1)*num_loc+j,:)./D),Ti2((i-1)*num_loc+j,:));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*Ti2((i-1)*num_loc+j,:);
            EIleft=min(temp1,Ti2((i-1)*num_loc+j,:));
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*Ts2((i-1)*num_loc+j,:);
            ESleft=min(temp1,Ts2((i-1)*num_loc+j,:));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightIincome(j,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightSincome(j,:);
            if discrete==0
                sk3((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik3((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik3i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk3((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik3((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik3i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    %fourth step
    Ts3=S(:,:,tcnt+1)+round(sk3);
    Ti3=I(:,:,tcnt+1)+round(ik3);
    nighttimeI=zeros(num_loc,num_ens);
    nighttimeS=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            nighttimeI(i,:)=nighttimeI(i,:)+Ti3((k-1)*num_loc+i,:);
            nighttimeS(i,:)=nighttimeS(i,:)+Ts3((k-1)*num_loc+i,:);
        end
    end
    nightIincome=zeros(num_loc,num_ens);
    nightSincome=zeros(num_loc,num_ens);
    for i=1:num_loc
        for k=1:num_loc
            if k~=i
                nightIincome(i,:)=nightIincome(i,:)+Nave(i,k).*nighttimeI(k,:)/N1(k);
                nightSincome(i,:)=nightSincome(i,:)+Nave(i,k).*nighttimeS(k,:)/N1(k);
            end
        end
    end
    for i=1:num_loc
        for j=1:num_loc
            temp=nighttimeI(j,:);
            Eimmloss=dt1*((Nij(i,j)*ones(1,num_ens)-Ts3((i-1)*num_loc+j,:)-Ti3((i-1)*num_loc+j,:))./L);
            Einf=min(dt1*(BETA(t,:,j).*Ts3((i-1)*num_loc+j,:).*temp/N1(j)),Ts3((i-1)*num_loc+j,:));
            Erecov=min(dt1*(Ti3((i-1)*num_loc+j,:)./D),Ti3((i-1)*num_loc+j,:));
            Eimmloss=max(Eimmloss,0);Einf=max(Einf,0);Erecov=max(Erecov,0);
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*Ti3((i-1)*num_loc+j,:);
            EIleft=min(temp1,Ti3((i-1)*num_loc+j,:));
            temp1=dt1*Ntotal/Ctotal*Cave(j)/N1(j)*q.*Ts3((i-1)*num_loc+j,:);
            ESleft=min(temp1,Ts3((i-1)*num_loc+j,:));
            EIenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightIincome(j,:);
            ESenter=dt1*Ntotal/Ctotal*Nij(i,j)/N1(j)*q.*nightSincome(j,:);
            if discrete==0
                sk4((i-1)*num_loc+j,:)=Eimmloss-Einf+ESenter-ESleft;
                ik4((i-1)*num_loc+j,:)=Einf-Erecov+EIenter-EIleft;
                ik4i((i-1)*num_loc+j,:)=Einf;
            end
            if discrete==1
                l=poissrnd([Eimmloss;Einf;Erecov;ESenter;ESleft;EIenter;EIleft]);
                sk4((i-1)*num_loc+j,:)=l(1,:)-l(2,:)+l(4,:)-l(5,:);
                ik4((i-1)*num_loc+j,:)=l(2,:)-l(3,:)+l(6,:)-l(7,:);
                ik4i((i-1)*num_loc+j,:)=l(2,:);
            end
        end
    end
    S(:,:,tcnt+1)=S(:,:,tcnt+1)+round(sk1/6+sk2/3+sk3/3+sk4/6);
    I(:,:,tcnt+1)=I(:,:,tcnt+1)+round(ik1/6+ik2/3+ik3/3+ik4/6);
    Incidence(:,:,tcnt+1)=Incidence(:,:,tcnt+1)+round(ik1i/6+ik2i/3+ik3i/3+ik4i/6);
    
    %observation
    for i=1:num_loc
        for j=1:num_loc
            obs(i,:)=obs(i,:)+Incidence((j-1)*num_loc+i,:,tcnt+1)/sum(Nij(:,i))*1e5;
        end
    end
end

for i=1:num_loc
    for j=1:num_loc
        %S
        x(idx(i,j,num_loc)+1,:)=S((i-1)*num_loc+j,:,tcnt+1);
        %I
        x(idx(i,j,num_loc)+2,:)=I((i-1)*num_loc+j,:,tcnt+1);
        %Incidence
        x(idx(i,j,num_loc)+3,:)=0;
        for t=1:tcnt
            x(idx(i,j,num_loc)+3,:)=x(idx(i,j,num_loc)+3,:)+Incidence((i-1)*num_loc+j,:,t+1)/sum(Nij(:,i))*1e5;
        end
    end
end

function a=idx(i,j,num_loc)
num_var=3;
a=(i-1)*num_var*num_loc+(j-1)*num_var;