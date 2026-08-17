function [raw,summary] = run_monte_carlo(nSubjects)
%RUN_MONTE_CARLO MATLAB mirror of the paired virtual-cohort benchmark.
%   [RAW,SUMMARY]=RUN_MONTE_CARLO(30) evaluates all paper controllers.
%   MATLAB's Twister differs from NumPy's PCG64, so use the included Python
%   runner to reproduce the published CSV bit-for-bit; this implementation
%   independently samples the same declared distributions and equations.

if nargin<1, nSubjects=30; end
root=fileparts(mfilename('fullpath')); addpath(root);
resultDir=fullfile(root,'results'); if ~exist(resultDir,'dir'), mkdir(resultDir); end
directData=load(fullfile(root,'policies','direct_ppo_selected.mat'));
pidData=load(fullfile(root,'policies','pid_ppo_selected.mat'));
methodNames={'Fixed PID','SL-LQG','Direct PPO','PPO-PID','ADMM-PID','RSP-ADMM-PID (proposed)'};
methodKeys={'fixed','sl_lqg','direct_ppo','ppo_pid','admm','proposed'};
rows=struct([]); rowIndex=0;

for index=0:nSubjects-1
    subjectSeed=900000+17*index; daySeed=1000000+31*index;
    day=bgc.sampleDay(daySeed); normal=bgc.sampleSubject('normal',subjectSeed);
    tr=bgc.simulate(normal,day,controllers.create('normal',normal,5));
    rowIndex=rowIndex+1; rows(rowIndex)=makeRow(bgc.metrics(tr,tr.glucoseTrue), ...
        'normal','Normal physiology',index,subjectSeed,daySeed);
    normalReference=tr.glucoseTrue;
    for cohortCell={'t1d','t2d'}
        cohort=cohortCell{1}; subject=bgc.sampleSubject(cohort,subjectSeed);
        for m=1:numel(methodKeys)
            if strcmp(methodKeys{m},'direct_ppo'), policy=directData.policy;
            elseif strcmp(methodKeys{m},'ppo_pid'), policy=pidData.policy;
            else, policy=[]; end
            controller=controllers.create(methodKeys{m},subject,5,policy);
            tic; trajectory=bgc.simulate(subject,day,controller); runtime=toc;
            values=bgc.metrics(trajectory,normalReference); values.runtime_s=runtime;
            rowIndex=rowIndex+1; rows(rowIndex)=makeRow(values,cohort,methodNames{m},index,subjectSeed,daySeed);
        end
    end
end
raw=struct2table(rows); writetable(raw,fullfile(resultDir,'matlab_per_subject_metrics.csv'));
summary=summarize(raw); writetable(summary,fullfile(resultDir,'matlab_summary_mean_std.csv'));
disp(summary(:,{'cohort','method','n','tir_70_180_mean','tir_70_180_std','normality_rmse_mean','normality_rmse_std'}));

    function row=makeRow(values,cohort,method,seedIndex,subjectSeed,daySeed)
        row=values; if ~isfield(row,'runtime_s'), row.runtime_s=NaN; end
        row.cohort=string(cohort); row.method=string(method); row.seed_index=seedIndex;
        row.subject_seed=subjectSeed; row.day_seed=daySeed;
    end

    function out=summarize(input)
        metrics={'tir_70_180','tight_80_140','tbr_70','tbr_54','tar_180', ...
            'mean_glucose','sd_glucose','cv_glucose','rmse_110','normality_rmse', ...
            'total_insulin_u','safety_interventions','runtime_s'};
        pairs=unique(input(:,{'cohort','method'}),'rows','stable'); aggregate=struct([]);
        for p=1:height(pairs)
            mask=input.cohort==pairs.cohort(p) & input.method==pairs.method(p);
            aggregate(p).cohort=pairs.cohort(p); aggregate(p).method=pairs.method(p); aggregate(p).n=sum(mask);
            for q=1:numel(metrics)
                name=metrics{q}; values=input.(name)(mask); values=values(isfinite(values));
                aggregate(p).([name '_mean'])=mean(values);
                aggregate(p).([name '_std'])=std(values,0);
                aggregate(p).([name '_ci95'])=1.96*std(values,0)/sqrt(numel(values));
            end
        end
        out=struct2table(aggregate);
    end
end
