function s = nominalSubject(cohort)
%NOMINALSUBJECT Nominal extended-Bergman phenotype used by the benchmark.
%   Units: glucose mg/dL, insulin mU/L, time min, pump rate U/h.
%   This code is an in-silico research benchmark, not a dosing tool.

cohort = lower(string(cohort));
s = struct('cohort', char(cohort), 'gb', 110, 'ib', 15, ...
    'p1', 0.012, 'p2', 0.025, 'p3', 1.30e-5, 'n', 5/54, ...
    'secretionGain', 0, 'basalRate', 0.90, 'maxRate', 5.0, ...
    'insulinGain', 0.90, 'tauIOB', 60, 'tauCGM', 10, ...
    'cgmBias', 0, 'cgmNoiseSD', 5, 'glucoseVolumeDL', 130, ...
    'exerciseGain', 0.75);

switch cohort
    case "normal"
        s.gb = 100; s.p1 = 0.028; s.secretionGain = 0.060;
        s.basalRate = 0; s.maxRate = 0; s.cgmNoiseSD = 4;
        s.exerciseGain = 0.60;
    case "t1d"
        % Defaults above.
    case "t2d"
        s.ib = 20; s.p1 = 0.018; s.p3 = 0.62*1.30e-5;
        s.secretionGain = 0.020; s.basalRate = 0.45;
        s.maxRate = 4.0; s.insulinGain = 0.85; s.tauIOB = 70;
        s.glucoseVolumeDL = 135; s.exerciseGain = 0.55;
    otherwise
        error('bgc:UnknownCohort', 'Unknown cohort: %s', cohort);
end
end
