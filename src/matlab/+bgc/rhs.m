function dx = rhs(tMin, x, rate, subject, day, appearanceOverride, sensitivityScale)
%RHS Extended Bergman dynamics for [G; X; I; IOB; G_CGM].

if nargin < 6, appearanceOverride = []; end
if nargin < 7, sensitivityScale = 1; end
g=x(1); remote=x(2); insulin=x(3); iob=x(4); gs=x(5);
if subject.maxRate > 0, u=min(max(rate,0),subject.maxRate); else, u=0; end
if ~isempty(appearanceOverride)
    ra = appearanceOverride;
elseif isempty(day)
    ra = 0;
else
    ra = bgc.mealAppearance(tMin, subject, day);
end
if isempty(day), exercise = 1; else, exercise = bgc.exerciseMultiplier(tMin,subject,day); end
circadian = 1 + 0.12*sin(2*pi*(tMin-15*60)/(24*60));
alpha = sensitivityScale*exercise*circadian;
secretion = subject.secretionGain*max(g-subject.gb,0);
dx = [-subject.p1*(g-subject.gb)-alpha*remote*g+ra; ...
      -subject.p2*remote+subject.p3*(insulin-subject.ib); ...
      -subject.n*(insulin-subject.ib)+subject.insulinGain*(u-subject.basalRate)+secretion; ...
      u/60-iob/subject.tauIOB; ...
      (g-gs)/subject.tauCGM];
end
