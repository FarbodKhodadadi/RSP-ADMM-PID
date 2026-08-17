function alpha = exerciseMultiplier(tMin, subject, day)
%EXERCISEMULTIPLIER Smooth activity response with post-exercise tail.

if ~isfinite(day.exerciseStart) || tMin < day.exerciseStart
    alpha = 1; return;
end
elapsed = tMin - day.exerciseStart;
rise = 1 - exp(-elapsed/10);
if elapsed <= day.exerciseDuration
    activity = rise;
else
    endLevel = 1 - exp(-day.exerciseDuration/10);
    activity = endLevel*exp(-(elapsed-day.exerciseDuration)/100);
end
alpha = 1 + subject.exerciseGain*day.exerciseIntensity*activity;
end
