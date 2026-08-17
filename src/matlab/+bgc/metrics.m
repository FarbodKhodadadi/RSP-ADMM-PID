function m = metrics(tr, normalReference)
%METRICS Compute sampled glucose-range and tracking outcomes.

if nargin < 2, normalReference=[]; end
g=tr.glucoseTrue(:); u=tr.insulinRate(:);
dtH=median(diff(tr.timeMin))/60;
m.tir_70_180=100*mean(g>=70 & g<=180);
m.tight_80_140=100*mean(g>=80 & g<=140);
m.tbr_70=100*mean(g<70); m.tbr_54=100*mean(g<54);
m.tar_180=100*mean(g>180); m.mean_glucose=mean(g);
m.sd_glucose=std(g,0); m.cv_glucose=100*std(g,0)/max(mean(g),eps);
m.rmse_110=sqrt(mean((g-110).^2));
m.total_insulin_u=sum(u(1:end-1))*dtH;
m.safety_interventions=sum(tr.safetyInterventions(1:end-1));
m.return=sum(tr.reward(1:end-1));
if isempty(normalReference)
    m.normality_rmse=NaN;
else
    ref=normalReference(:); assert(numel(ref)==numel(g),'Reference size mismatch.');
    m.normality_rmse=sqrt(mean((g-ref).^2));
end
end
