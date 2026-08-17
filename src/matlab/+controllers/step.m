function [c,rate,info] = step(c,tMin,glucoseCGM)
%STEP Advance a controller by one five-minute CGM sample.

info=struct('safetyIntervened',false);
switch c.method
    case "normal"
        rate=0;
    case "fixed"
        [c,rate,info]=fixedStep(c,glucoseCGM);
    case "sl_lqg"
        [c,rate,info]=lqgStep(c,glucoseCGM);
    case {"admm","proposed"}
        [c,rate,info]=adaptiveStep(c,tMin,glucoseCGM,c.method=="proposed",c.method=="proposed");
    case "direct_ppo"
        [c,rate,info]=directPolicyStep(c,tMin,glucoseCGM);
    case "ppo_pid"
        [c,rate,info]=policyPIDStep(c,tMin,glucoseCGM);
    otherwise
        error('controllers:UnknownMethod','Unknown method: %s',c.method);
end
end

function [c,rate,info]=fixedStep(c,y)
[c,error,candidate]=pidFeatures(c,y,10/(10+c.dt));
gains=c.gainNominal; raw=c.subject.basalRate+gains*[error;candidate;c.derivative];
rate=min(max(raw,0),c.subject.maxRate);
if ~((raw>c.subject.maxRate && error>0)||(raw<0 && error<0)), c.integral=candidate; end
c.prevError=error; c.first=false;
info=struct('gains',gains,'safetyIntervened',false);
end

function [c,rate,info]=lqgStep(c,y)
[c.observer,xhat]=observerUpdate(c.observer,y);
model=c.observer.model; g=xhat(1); remote=xhat(2);
dsec=model.secretionGain*(g>model.gb);
A=[-model.p1-remote,-g,0;0,-model.p2,model.p3;dsec,0,-model.n];
B=[0;0;model.insulinGain]; aug=zeros(4); aug(1:3,1:3)=A; aug(1:3,4)=B;
disc=expm(aug*c.dt); Ad=disc(1:3,1:3); Bd=disc(1:3,4);
Q=diag([1/35^2,1/0.012^2,1/70^2]); R=0.040; P=Q;
for k=1:200
    K=(R+Bd'*P*Bd)\(Bd'*P*Ad);
    Pnew=Ad'*P*Ad-Ad'*P*Bd*K+Q;
    if norm(Pnew-P,'fro')<1e-10, P=Pnew; break; end
    P=Pnew;
end
K=(R+Bd'*P*Bd)\(Bd'*P*Ad);
if all(isfinite(K)), c.lastK=K; end
deviation=[xhat(1)-c.reference;xhat(2);xhat(3)-model.ib];
rate=min(max(c.subject.basalRate-c.lastK*deviation,0),c.subject.maxRate);
c.observer.uPrev=rate; info=struct('safetyIntervened',false);
end

function [c,rate,info]=adaptiveStep(c,tMin,y,robust,safety)
[c.observer,xhat]=observerUpdate(c.observer,y);
[c,error,candidate]=pidFeatures(c,y,0.67);
if mod(c.stepCount,2)==0
    theta=admmUpdate(c,xhat,candidate,error,tMin,robust);
else
    theta=c.thetaPrev;
end
gains=c.gainMin+min(max(theta,0),1).*(c.gainMax-c.gainMin);
raw=c.subject.basalRate+gains*[error;candidate;c.derivative];
nominalRate=min(max(raw,0),c.subject.maxRate);
rate=safetyProjection(c,xhat,nominalRate,tMin,safety);
intervened=rate<nominalRate-1e-4;
if ~((raw>c.subject.maxRate && error>0)||(raw<0 && error<0)||intervened)
    c.integral=candidate;
end
c.prevError=error; c.thetaPrev=theta; c.first=false; c.stepCount=c.stepCount+1;
c.observer.uPrev=rate;
info=struct('gains',gains,'safetyIntervened',intervened);
end

function theta=admmUpdate(c,xhat,integral,error,tMin,robust)
z=c.thetaPrev; split=c.thetaPrev; dual=zeros(1,3);
rho=7; smooth=2.5; nominal=0.5;
if robust, stepSize=0.060; radius=0.12; else, stepSize=0.055; radius=0.14; end
for iteration=1:3
    grad=finiteDifference(c,z,xhat,integral,error,tMin,robust);
    z=clip01(z-stepSize*(grad+rho*(z-split+dual)));
    split=clip01((rho*(z+dual)+smooth*c.thetaPrev+nominal*c.thetaNominal)/(rho+smooth+nominal));
    dual=dual+z-split;
end
theta=clip01(0.5*(z+split));
theta=min(max(theta,c.thetaPrev-radius),c.thetaPrev+radius); theta=clip01(theta);
end

function grad=finiteDifference(c,theta,xhat,integral,error,tMin,robust)
grad=zeros(1,3); delta=0.018;
for index=1:3
    plus=theta; minus=theta; plus(index)=min(1,plus(index)+delta); minus(index)=max(0,minus(index)-delta);
    denominator=plus(index)-minus(index);
    if denominator>1e-12
        jp=rolloutCost(c,plus,xhat,integral,error,tMin,robust);
        jm=rolloutCost(c,minus,xhat,integral,error,tMin,robust);
        grad(index)=(jp-jm)/denominator;
    end
end
grad=min(max(grad,-20),20);
end

function cost=rolloutCost(c,theta,x0,integral0,error0,tMin,robust)
gains=c.gainMin+clip01(theta).*(c.gainMax-c.gainMin);
if robust, scenarios=[0,1.25;2.0,1.0;4.2,0.82]; else, scenarios=[0,1.0]; end
scenarioCosts=zeros(size(scenarios,1),1);
for sidx=1:size(scenarios,1)
    x=x0; integ=integral0; deriv=c.derivative; prevError=error0; value=0;
    for j=0:7
        e=x(1)-c.reference; integ=min(max(integ+c.dt*e,-12000),12000);
        deriv=0.67*deriv+0.33*(e-prevError)/c.dt;
        u=c.observer.model.basalRate+gains*[e;integ;deriv];
        u=min(max(u,0),c.observer.model.maxRate);
        appearance=scenarios(sidx,1)*exp(-j*c.dt/45);
        x=bgc.advanceInterval(tMin+j*c.dt,x,u,c.observer.model,[],c.dt,c.dt,appearance,scenarios(sidx,2));
        g=x(1); track=((g-c.reference)/34)^2;
        tightHigh=0.90*(max(g-140,0)/40)^2; hypo=9*(max(75-g,0)/18)^2;
        hyper=2*(max(g-180,0)/60)^2;
        effort=0.010*((u-c.observer.model.basalRate)/c.observer.model.maxRate)^2;
        value=value+track+tightHigh+hypo+hyper+effort; prevError=e;
    end
    scenarioCosts(sidx)=value/8;
end
cost=mean(scenarioCosts); if robust, cost=cost+0.20*max(scenarioCosts); end
end

function rate=safetyProjection(c,xhat,nominalRate,tMin,enabled)
if ~enabled, rate=nominalRate; return; end
if xhat(1)<=78, rate=0; return; end
basalIOB=c.observer.model.basalRate*c.observer.model.tauIOB/60;
if xhat(1)>=125 || (xhat(1)>=100 && c.derivative>=0 && xhat(4)<basalIOB+1.5)
    rate=nominalRate; return;
end
floorValue=75;
if minimumPrediction(c,xhat,nominalRate,tMin)>=floorValue, rate=nominalRate; return; end
if minimumPrediction(c,xhat,0,tMin)<floorValue, rate=0; return; end
low=0; high=nominalRate;
for k=1:12
    mid=0.5*(low+high);
    if minimumPrediction(c,xhat,mid,tMin)>=floorValue, low=mid; else, high=mid; end
end
rate=low;
end

function minimum=minimumPrediction(c,xhat,rate,tMin)
model=c.observer.model; model.p3=1.18*model.p3; model.insulinGain=1.10*model.insulinGain;
x=xhat; minimum=x(1);
for j=0:8
    x=bgc.advanceInterval(tMin+j*c.dt,x,rate,model,[],c.dt,c.dt,0,1.18);
    minimum=min(minimum,x(1));
end
end

function [c,rate,info]=directPolicyStep(c,tMin,y)
[c,error,~]=policyFeatures(c,y); c.integral=min(max(c.integral+error*c.dt,-12000),12000);
obs=policyObservation(c,y,tMin); action=policyAction(c.policy,obs);
rate=min(max(action(1),0),1)*c.subject.maxRate; c.rate=rate;
c.prevError=error; c.first=false; info=struct('safetyIntervened',false);
end

function [c,rate,info]=policyPIDStep(c,tMin,y)
[c,error,candidate]=policyFeatures(c,y); obs=policyObservation(c,y,tMin,candidate);
action=policyAction(c.policy,obs); gains=c.gainMin+min(max(action(:)',0),1).*(c.gainMax-c.gainMin);
raw=c.subject.basalRate+gains*[error;candidate;c.derivative];
rate=min(max(raw,0),c.subject.maxRate); c.rate=rate;
if ~((raw>c.subject.maxRate && error>0)||(raw<0 && error<0)), c.integral=candidate; end
c.prevError=error; c.first=false; info=struct('gains',gains,'safetyIntervened',false);
end

function [c,error,candidate]=policyFeatures(c,y)
error=y-c.reference; if c.first, rawD=0; else, rawD=(error-c.prevError)/c.dt; end
c.derivative=0.67*c.derivative+0.33*rawD;
candidate=min(max(c.integral+error*c.dt,-12000),12000);
decay=exp(-c.dt/c.subject.tauIOB);
c.iob=decay*c.iob+c.rate*c.subject.tauIOB/60*(1-decay);
end

function obs=policyObservation(c,y,tMin,integralValue)
if nargin<4, integralValue=c.integral; end
oneHot=[strcmpi(c.subject.cohort,'t1d'),strcmpi(c.subject.cohort,'t2d')];
obs=[min(max((y-110)/100,-1.5),3), min(max(c.derivative/8,-2),2), ...
     min(max(integralValue/8000,-1.5),1.5), min(max(c.iob/6,0),3), ...
     min(max(c.rate/max(c.subject.maxRate,1),0),1), ...
     sin(2*pi*tMin/1440),cos(2*pi*tMin/1440),c.subject.basalRate/2,oneHot];
end

function action=policyAction(policy,obs)
assert(~isempty(policy),'A trained policy structure is required.');
hidden=tanh(obs*policy.w1+reshape(policy.b1,1,[]));
latent=hidden*policy.w2+reshape(policy.b2,1,[]);
latent=min(max(latent,-30),30); action=1./(1+exp(-latent));
end

function [c,error,candidate]=pidFeatures(c,y,derivativeMemory)
error=y-c.reference; if c.first, rawD=0; else, rawD=(error-c.prevError)/c.dt; end
c.derivative=derivativeMemory*c.derivative+(1-derivativeMemory)*rawD;
candidate=min(max(c.integral+error*c.dt,-12000),12000);
end

function [o,xhat]=observerUpdate(o,y)
if ~o.initialized
    o.x(1)=y; o.initialized=true; xhat=[o.x;o.iob;o.x(1)]; return;
end
model=o.model; g=o.x(1); remote=o.x(2); dsec=model.secretionGain*(g>model.gb);
F=eye(3)+o.dt*[-model.p1-remote,-g,0;0,-model.p2,model.p3;dsec,0,-model.n];
o.x=observerRK4(o.x,o.uPrev,model,o.dt); o.P=F*o.P*F'+o.Q;
H=[1,0,0]; innovation=y-H*o.x; S=H*o.P*H'+o.R; K=o.P*H'/S;
o.x=o.x+K*innovation; o.P=(eye(3)-K*H)*o.P;
o.x(1)=min(max(o.x(1),35),500); o.x(2)=min(max(o.x(2),-0.03),0.08); o.x(3)=min(max(o.x(3),0),600);
decay=exp(-o.dt/model.tauIOB); o.iob=decay*o.iob+o.uPrev*model.tauIOB/60*(1-decay);
xhat=[o.x;o.iob;o.x(1)];
end

function out=observerRK4(x,u,m,dt)
f=@(z) [-m.p1*(z(1)-m.gb)-z(2)*z(1); ...
          -m.p2*z(2)+m.p3*(z(3)-m.ib); ...
          -m.n*(z(3)-m.ib)+m.insulinGain*(u-m.basalRate)+m.secretionGain*max(z(1)-m.gb,0)];
k1=f(x); k2=f(x+dt*k1/2); k3=f(x+dt*k2/2); k4=f(x+dt*k3);
out=x+dt*(k1+2*k2+2*k3+k4)/6;
out(1)=min(max(out(1),35),500); out(2)=min(max(out(2),-0.03),0.08); out(3)=min(max(out(3),0),600);
end

function value=clip01(value)
value=min(max(value,0),1);
end
