function run_smoke
%RUN_SMOKE Short deterministic comparison and controller sanity plot.

root=fileparts(mfilename('fullpath')); addpath(root);
subject=bgc.sampleSubject('t1d',900000); day=bgc.sampleDay(1000000);
fixed=controllers.create('fixed',subject,5);
proposed=controllers.create('proposed',subject,5);
trFixed=bgc.simulate(subject,day,fixed,6*60);
trProposed=bgc.simulate(subject,day,proposed,6*60);
assert(all(isfinite(trFixed.state),'all') && all(isfinite(trProposed.state),'all'));
assert(all(trProposed.insulinRate>=0 & trProposed.insulinRate<=subject.maxRate));

figure('Color','w'); tiledlayout(2,1,'TileSpacing','compact');
nexttile; plot(trFixed.timeMin/60,trFixed.glucoseTrue,'LineWidth',1.4); hold on;
plot(trProposed.timeMin/60,trProposed.glucoseTrue,'LineWidth',1.4);
yline(70,'--'); yline(180,'--'); ylabel('Glucose (mg/dL)'); grid on;
legend('Fixed PID','RSP-ADMM-PID','Location','best');
nexttile; stairs(trFixed.timeMin/60,trFixed.insulinRate,'LineWidth',1.2); hold on;
stairs(trProposed.timeMin/60,trProposed.insulinRate,'LineWidth',1.2);
xlabel('Time (h)'); ylabel('Pump rate (U/h)'); grid on;
disp(bgc.metrics(trFixed)); disp(bgc.metrics(trProposed));
end
