function tr = simulate(subject, day, controller, horizonMin, controlDt, integrationDt)
%SIMULATE Run one deterministic CGM-to-pump closed-loop experiment.

if nargin < 4, horizonMin=24*60; end
if nargin < 5, controlDt=5; end
if nargin < 6, integrationDt=1; end
nSteps=round(horizonMin/controlDt)+1;
assert(numel(day.cgmNoise)>=nSteps && numel(day.processNoise)>=nSteps, ...
    'Noise sequences are shorter than the simulation horizon.');
tr.timeMin=(0:nSteps-1)'*controlDt;
tr.state=zeros(nSteps,5); tr.glucoseCGM=zeros(nSteps,1);
tr.insulinRate=zeros(nSteps,1); tr.gains=nan(nSteps,3);
tr.reward=zeros(nSteps,1); tr.safetyInterventions=false(nSteps,1);
g0=subject.gb+4*day.initialZ(1); i0=max(0,subject.ib+day.initialZ(2));
tr.state(1,:)=[g0,0,i0,subject.basalRate*subject.tauIOB/60,g0];
for k=1:nSteps-1
    measurement=tr.state(k,5)+subject.cgmBias+subject.cgmNoiseSD*day.cgmNoise(k);
    tr.glucoseCGM(k)=min(max(measurement,40),400);
    [controller,rate,info]=controllers.step(controller,tr.timeMin(k),tr.glucoseCGM(k));
    if subject.maxRate>0, rate=min(max(rate,0),subject.maxRate); else, rate=0; end
    tr.insulinRate(k)=rate;
    if isfield(info,'gains'), tr.gains(k,:)=info.gains(:)'; end
    if isfield(info,'safetyIntervened'), tr.safetyInterventions(k)=info.safetyIntervened; end
    xnext=bgc.advanceInterval(tr.timeMin(k),tr.state(k,:)',rate,subject,day,controlDt,integrationDt);
    xnext(1)=min(max(xnext(1)+day.processNoise(k+1),35),500);
    tr.state(k+1,:)=xnext';
    tr.reward(k)=clinicalReward(tr.state(k,1),rate,subject);
end
tr.glucoseCGM(end)=min(max(tr.state(end,5)+subject.cgmBias+subject.cgmNoiseSD*day.cgmNoise(end),40),400);
tr.insulinRate(end)=tr.insulinRate(end-1); tr.reward(end)=clinicalReward(tr.state(end,1),tr.insulinRate(end),subject);
if all(isfinite(tr.gains(end-1,:))), tr.gains(end,:)=tr.gains(end-1,:); end
tr.safetyInterventions(end)=tr.safetyInterventions(end-1);
tr.glucoseTrue=tr.state(:,1);

    function value=clinicalReward(g,u,s)
        track=((g-110)/45)^2; low=8*(max(70-g,0)/20)^2;
        veryLow=14*(max(54-g,0)/16)^2; high=1.6*(max(g-180,0)/60)^2;
        effort=0.015*((u-s.basalRate)/max(s.maxRate,1))^2;
        value=-(track+low+veryLow+high+effort);
    end
end
