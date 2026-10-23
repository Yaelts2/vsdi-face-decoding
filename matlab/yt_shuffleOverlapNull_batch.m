%% yt_shuffleOverlapNull_batch.m
% Shuffle-model null for the phase-1 / phase-2 extreme-weight overlap, all datasets + group test.
%
% For every model: real overlap (yt_phaseOverlap on Weights_acrossT at the two peaks) vs the
% overlap of the 100 shuffled-label models at the SAME peak windows (NullWeights_acrossT).
% Group level: two-tailed exact Wilcoxon signed-rank, real vs null mean, sessions as the unit (n = 16).
% The 4 sign combinations are tested the same way, BH-FDR across the 5 tests.
%
% Needs on path: yt_phaseOverlap.m, yt_shuffleOverlapNull.m, yt_bhFDR.m (and plotModelFigures.m if a peak is left empty)

clear; clc;

MAT_DIR = 'C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\saveDataModels\all';
OUT_DIR = 'C:\Users\USER\OneDrive - Bar-Ilan University - Students\Documents\school\2nd\lab\new project\new project results\overlapShuffleNull';

% ------------------------------------------------------------------------------------------
% Models + the peak windows (CENTER frames) you use for the real overlap of each model.
%   - leave MODELS = {} to take EVERY .mat in MAT_DIR that has NullWeights_acrossT
%   - a peak left [] is chosen with plotModelFigures (interactive, as before)   <- EDIT
% ------------------------------------------------------------------------------------------
MODELS = {
%   .mat file name (in MAT_DIR)                                                    first peak   second peak
%   'sliding_window__110209a24_frame1-100__SVM_10foldCV____2026-07-19_13-24-35.mat', 34:36,       48:50
};

MAKE_MODEL_FIGS = true;     % per-model figures (3 per model)

% ------------------------------------------------------------------------------------------
if isempty(MODELS)
    files = dir(fullfile(MAT_DIR, 'sliding_window__*.mat'));
    files = files(cellfun(@isempty, strfind({files.name}, '__backup_before_perm')));
    MODELS = cell(0, 3);
    for k = 1:numel(files)
        vars = whos('-file', fullfile(MAT_DIR, files(k).name));
        if any(strcmp({vars.name}, 'NullWeights_acrossT'))
            MODELS(end+1, :) = {files(k).name, [], []}; %#ok<SAGROW>
        end
    end
    fprintf('MODELS empty -> %d models with NullWeights_acrossT found in MAT_DIR\n', size(MODELS, 1));
    if isempty(MODELS); error('no model .mat with NullWeights_acrossT in %s', MAT_DIR); end
end

n_models = size(MODELS, 1);
R = cell(n_models, 1);
for m = 1:n_models
    matpath = fullfile(MAT_DIR, MODELS{m,1});
    f1 = MODELS{m,2};  f2 = MODELS{m,3};
    if isempty(f1) || isempty(f2)
        fprintf('\n--- %s: choose the peaks ---\n', MODELS{m,1});
        [f1, f2, ov_plot] = plotModelFigures(matpath);
        MODELS{m,2} = f1;  MODELS{m,3} = f2;
    else
        ov_plot = [];
    end
    R{m} = yt_shuffleOverlapNull(matpath, f1, f2, MAKE_MODEL_FIGS);
    % same procedure check: must equal plotModelFigures' overlap
    if ~isempty(ov_plot) && abs(ov_plot - R{m}.real) > 1e-9
        warning('%s: yt_phaseOverlap (%.3f) differs from plotModelFigures (%.3f)!', MODELS{m,1}, R{m}.real, ov_plot);
    end
end
R = [R{:}];

% ------------------------------------------------------------------------------------------
% Group level
% ------------------------------------------------------------------------------------------
real_ov   = [R.real];
null_mean = [R.null_mean];
null_std  = [R.null_std];
z_ov      = [R.z];
p_sess    = [R.p_two];
names     = {R.name};
short     = cellfun(@(s) regexprep(s, '^sliding_window__(\w+?)_frame.*$', '$1'), names, 'UniformOutput', false);

combo_real = reshape([R.combo_real], 4, [])';        % n_models x 4
combo_null = reshape([R.combo_null_mean], 4, [])';
combo_names = R(1).combo_names;

test_names = [{'Overall'}, combo_names];
X = [real_ov(:), combo_real];
Y = [null_mean(:), combo_null];
p_group = nan(1, 5);
for t = 1:5
    if n_models >= 2
        p_group(t) = signrank(X(:,t), Y(:,t), 'method', 'exact');
    end
end
q_group = yt_bhFDR(p_group);

fprintf('\n================ SHUFFLE-MODEL OVERLAP NULL: %d models ================\n', n_models);
fprintf('%-12s %8s %8s %8s %7s %8s\n', 'model', 'real%', 'null%', 'null_sd', 'z', 'p(two)');
for m = 1:n_models
    fprintf('%-12s %8.1f %8.1f %8.1f %7.2f %8.3f\n', short{m}, real_ov(m), null_mean(m), null_std(m), z_ov(m), p_sess(m));
end
fprintf('\nsessions with real > null (p<0.05, two-tailed): %d / %d\n', sum(p_sess < 0.05 & real_ov > null_mean), n_models);
fprintf('\nGroup (Wilcoxon signed-rank, exact, two-tailed, real vs null mean; BH-FDR over 5 tests):\n');
for t = 1:5
    fprintf('  %-24s real %5.1f%% | null %5.1f%% | median diff %+5.1f | p %.4g | q %.4g\n', test_names{t}, ...
        mean(X(:,t)), mean(Y(:,t)), median(X(:,t) - Y(:,t)), p_group(t), q_group(t));
end

% ------------------------------------------------------------------------------------------
% Group figures
% ------------------------------------------------------------------------------------------
% 1) real vs null mean, one line per session
figure('Name', 'Overlap: real vs shuffle null', 'Color', 'w'); hold on;
for m = 1:n_models
    plot([1 2], [null_mean(m) real_ov(m)], '-', 'Color', [0.6 0.6 0.6]);
end
errorbar(ones(1,n_models), null_mean, null_std, 'o', 'Color', [0.4 0.4 0.4], 'MarkerFaceColor', [0.7 0.7 0.7]);
plot(2*ones(1,n_models), real_ov, 'o', 'Color', 'r', 'MarkerFaceColor', 'r');
set(gca, 'XTick', [1 2], 'XTickLabel', {'shuffle null (mean \pm sd)', 'real'}, 'XLim', [0.5 2.5]);
ylabel('Phase-1 / phase-2 overlap (%)');
title(sprintf('%d sessions | Wilcoxon p = %.4g', n_models, p_group(1)));
box off; hold off;

% 2) per-session z
figure('Name', 'Overlap z vs shuffle null', 'Color', 'w');
bar(z_ov, 'FaceColor', [0.5 0.5 0.8]); hold on;
xl = xlim;
plot(xl, [1.96 1.96], '--k'); plot(xl, [-1.96 -1.96], '--k'); plot(xl, [0 0], 'k');
set(gca, 'XTick', 1:n_models, 'XTickLabel', short, 'XTickLabelRotation', 45);
ylabel('z = (real - null mean) / null sd'); title('Real overlap relative to its own shuffle null');
box off; hold off;

% 3) the 4 sign combinations
figure('Name', 'Sign combos: real vs shuffle null', 'Color', 'w');
for i = 1:4
    subplot(2,2,i); hold on;
    for m = 1:n_models
        plot([1 2], [combo_null(m,i) combo_real(m,i)], '-', 'Color', [0.6 0.6 0.6]);
    end
    plot(ones(1,n_models), combo_null(:,i), 'o', 'Color', [0.4 0.4 0.4], 'MarkerFaceColor', [0.7 0.7 0.7]);
    plot(2*ones(1,n_models), combo_real(:,i), 'o', 'Color', 'r', 'MarkerFaceColor', 'r');
    set(gca, 'XTick', [1 2], 'XTickLabel', {'null', 'real'}, 'XLim', [0.5 2.5]);
    ylabel('Overlap (%)');
    title(sprintf('%s\np = %.3g, q = %.3g', combo_names{i}, p_group(i+1), q_group(i+1)));
    box off; hold off;
end

% ------------------------------------------------------------------------------------------
% Save (tempdir -> movefile)
% ------------------------------------------------------------------------------------------
if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end
results = struct();
results.R          = R;            % per-model results (real, null, z, p, combos)
results.models     = MODELS;       % file names + the peak windows used
results.test_names = test_names;
results.p_group    = p_group;
results.q_group    = q_group;
fname = sprintf('overlapShuffleNull_%s.mat', datestr(now, 'yyyy-mm-dd_HH-MM'));
tmp = fullfile(tempdir, fname);
save(tmp, 'results', '-v7');
movefile(tmp, fullfile(OUT_DIR, fname));
fprintf('\nsaved %s\n', fullfile(OUT_DIR, fname));

