function out = advanceInterval(tMin, x, rate, subject, day, controlDt, integrationDt, appearanceOverride, sensitivityScale)
%ADVANCEINTERVAL RK4 integration under a zero-order-held pump command.

if nargin < 6, controlDt=5; end
if nargin < 7, integrationDt=1; end
if nargin < 8, appearanceOverride=[]; end
if nargin < 9, sensitivityScale=1; end
n = round(controlDt/integrationDt);
assert(abs(n*integrationDt-controlDt)<1e-10, 'controlDt must divide integrationDt.');
out = x(:);
for j=0:n-1
    tt=tMin+j*integrationDt; h=integrationDt;
    f=@(time,state) bgc.rhs(time,state,rate,subject,day,appearanceOverride,sensitivityScale);
    k1=f(tt,out); k2=f(tt+h/2,out+h*k1/2);
    k3=f(tt+h/2,out+h*k2/2); k4=f(tt+h,out+h*k3);
    out=out+h*(k1+2*k2+2*k3+k4)/6;
    out(1)=min(max(out(1),35),500); out(2)=min(max(out(2),-0.03),0.08);
    out(3)=min(max(out(3),0),600); out(4)=min(max(out(4),0),30);
    out(5)=min(max(out(5),35),500);
end
end
