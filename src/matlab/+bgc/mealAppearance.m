function ra = mealAppearance(tMin, subject, day)
%MEALAPPEARANCE Gamma (two-compartment) meal appearance, mg/dL/min.

lag = tMin - day.mealTimes;
active = lag >= 0;
if ~any(active), ra = 0; return; end
lag = lag(active); tau = day.mealTaus(active);
absorbed = 0.90*1000*day.mealCarbsG(active)/subject.glucoseVolumeDL;
ra = sum(absorbed .* lag .* exp(-lag./tau) ./ (tau.^2));
end
