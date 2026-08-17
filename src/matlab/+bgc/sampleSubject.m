function s = sampleSubject(cohort, seed)
%SAMPLESUBJECT Reproducible lognormal domain randomization of one subject.
% The spreads are robustness-test choices, not clinically fitted prevalence.

rng(double(seed), 'twister');
s = bgc.nominalSubject(cohort);
bodyMass = min(max(logMultiplier(0.12), 0.72), 1.35);
s.gb = s.gb + 3*randn;
s.ib = max(5, s.ib*logMultiplier(0.12));
s.p1 = s.p1*logMultiplier(0.22);
s.p2 = s.p2*logMultiplier(0.18);
s.p3 = s.p3*logMultiplier(0.28);
s.n = s.n*logMultiplier(0.15);
if s.secretionGain > 0, s.secretionGain = s.secretionGain*logMultiplier(0.30); end
if s.basalRate > 0, s.basalRate = s.basalRate*logMultiplier(0.12); end
s.insulinGain = s.insulinGain*logMultiplier(0.18);
s.tauIOB = s.tauIOB*logMultiplier(0.18);
s.tauCGM = min(max(s.tauCGM*logMultiplier(0.20), 5), 18);
s.cgmBias = 3*randn;
s.cgmNoiseSD = min(max(s.cgmNoiseSD*logMultiplier(0.15), 2.5), 8);
s.glucoseVolumeDL = s.glucoseVolumeDL*bodyMass;
s.exerciseGain = s.exerciseGain*logMultiplier(0.25);

    function value = logMultiplier(cv)
        sigma = sqrt(log(1 + cv^2));
        value = exp(sigma*randn - 0.5*sigma^2);
    end
end
