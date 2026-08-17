function day = sampleDay(seed, nSteps)
%SAMPLEDAY Draw three meals, optional snack, exercise, and matched noise.

if nargin < 2, nSteps = 289; end
rng(double(seed), 'twister');
means = 60*[8, 13, 19]; sds = [25, 35, 40];
lows = 60*[6.5, 11, 17]; highs = 60*[10, 15.5, 21.5];
day.mealTimes = zeros(1,3);
for k = 1:3
    day.mealTimes(k) = truncatedNormal(means(k), sds(k), lows(k), highs(k));
end
medians = [48, 68, 74];
day.mealCarbsG = arrayfun(@(m) min(max(m*logMultiplier(0.25),20),120), medians);
day.mealTaus = arrayfun(@(~) min(max(45*logMultiplier(0.22),25),80), 1:3);
if rand < 0.60
    day.mealTimes(end+1) = truncatedNormal(22*60, 25, 20.5*60, 23.5*60);
    day.mealCarbsG(end+1) = min(max(24*logMultiplier(0.30),10),45);
    day.mealTaus(end+1) = min(max(38*logMultiplier(0.20),22),65);
end
[day.mealTimes, order] = sort(day.mealTimes);
day.mealCarbsG = day.mealCarbsG(order); day.mealTaus = day.mealTaus(order);
if rand < 0.78
    day.exerciseStart = truncatedNormal(17.2*60, 50, 14.5*60, 20.5*60);
    day.exerciseDuration = 30 + 45*rand;
    day.exerciseIntensity = 0.35 + 0.50*rand;
else
    day.exerciseStart = inf; day.exerciseDuration = 0; day.exerciseIntensity = 0;
end
day.cgmNoise = randn(nSteps,1);
day.processNoise = 0.70*randn(nSteps,1);
day.initialZ = randn(2,1);

    function x = truncatedNormal(mu, sd, low, high)
        x = mu;
        for attempt = 1:100
            x = mu + sd*randn;
            if x >= low && x <= high, return; end
        end
        x = min(max(x, low), high);
    end
    function value = logMultiplier(cv)
        sigma = sqrt(log(1 + cv^2));
        value = exp(sigma*randn - 0.5*sigma^2);
    end
end
