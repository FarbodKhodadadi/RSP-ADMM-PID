function c = create(method, subject, dt, policy)
%CREATE Initialize one benchmark controller.
% Methods: fixed, sl_lqg, direct_ppo, ppo_pid, admm, proposed, normal.

if nargin < 3, dt=5; end
if nargin < 4, policy=[]; end
c.method=lower(string(method)); c.subject=subject; c.dt=dt; c.reference=110;
c.integral=0; c.prevError=0; c.derivative=0; c.first=true;
c.rate=subject.basalRate; c.iob=subject.basalRate*subject.tauIOB/60;
c.gainMin=[0,0,0]; c.gainMax=[0.080,0.00080,0.250];
if strcmpi(subject.cohort,'t1d'), c.gainNominal=[0.028,0.00012,0.045];
else, c.gainNominal=[0.024,0.00010,0.040]; end
c.thetaNominal=(c.gainNominal-c.gainMin)./(c.gainMax-c.gainMin);
c.thetaPrev=c.thetaNominal; c.stepCount=0; c.policy=policy;
if any(c.method==["sl_lqg","admm","proposed"])
    c.observer=makeObserver(subject,dt);
end
if c.method=="sl_lqg", c.lastK=zeros(1,3); end

    function o=makeObserver(s,h)
        model=bgc.nominalSubject(s.cohort); model.basalRate=s.basalRate; model.maxRate=s.maxRate;
        o.model=model; o.dt=h; o.initialized=false; o.x=[model.gb;0;model.ib];
        o.P=diag([36,2.5e-5,25]); o.Q=diag([7,2e-7,2]);
        o.R=max(model.cgmNoiseSD^2,9); o.iob=model.basalRate*model.tauIOB/60;
        o.uPrev=model.basalRate;
    end
end
