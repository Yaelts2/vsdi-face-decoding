function r = yt_shuffleOverlapNull(matpath, first_frames, second_frames, make_figs)
%YT_SHUFFLEOVERLAPNULL  Shuffle-model null for the phase-1 / phase-2 overlap of ONE model.
%   r = yt_shuffleOverlapNull(matpath, first_frames, second_frames)
%   r = yt_shuffleOverlapNull(matpath, first_frames, second_frames, make_figs)
%
%   matpath        : model .mat (after open_permutation_sliding.py added NullWeights_acrossT)
%   first_frames   : CENTER frames of the first peak  (same as plotModelFigures, e.g. 33:35)
%   second_frames  : CENTER frames of the second peak (e.g. 46:48)
%   make_figs      : true (default) -> per-model figures
%
%   The overlap is computed with yt_overlapFromMaps on
%       real : mean(Weights_acrossT(:, peak windows), 2)
%       null : mean(NullWeights_acrossT(:, same peak windows, p), 2)   for every shuffle p
%   i.e. the SAME windows and the SAME procedure, only the weights come from a model trained on
%   shuffled labels.
%
%   r.real, r.null (1 x n_perm), r.null_mean, r.null_std, r.z, r.p_greater, r.p_two
%   r.combo_real (1x4), r.combo_null (n_perm x 4), r.combo_null_mean, r.combo_z, r.combo_p_two
%   p_greater = (#null >= real + 1) / (n_perm + 1);  p_two = min(1, 2*min(p_greater, p_less))

if nargin < 4 || isempty(make_figs); make_figs = true; end

S = load(matpath, 'Weights_acrossT', 'centers', 'NullWeights_acrossT', 'null_w_centers', 'model_id');
if ~isfield(S, 'NullWeights_acrossT')
    error('%s has no NullWeights_acrossT - run open_permutation_sliding.py for this model first.', matpath);
end
W        = double(S.Weights_acrossT);
centers  = S.centers(:)';
NullW    = S.NullWeights_acrossT;                  % 10000 x n_null_win x n_perm (single)
null_c   = S.null_w_centers(:)';
n_perm   = size(NullW, 3);
if size(W,1) ~= 10000 && size(W,2) == 10000; W = W'; end

[~, name] = fileparts(char(matpath));

% --- peak windows: positions in the real model and in the null ---
pos1 = find(ismember(centers, first_frames));
pos2 = find(ismember(centers, second_frames));
if numel(pos1) ~= numel(first_frames) || numel(pos2) ~= numel(second_frames)
    error('%s: peak centers not found in centers.', name);
end
npos1 = find(ismember(null_c, first_frames));
npos2 = find(ismember(null_c, second_frames));
if numel(npos1) ~= numel(first_frames) || numel(npos2) ~= numel(second_frames)
    error(['%s: peak windows %s / %s are outside the windows that were shuffled (centers %d-%d).\n' ...
           'Rerun the permutation for this model with a wider T_MS.'], ...
           name, mat2str(first_frames), mat2str(second_frames), min(null_c), max(null_c));
end

% --- real overlap ---
firstpeak  = mean(W(:, pos1), 2);
secondpeak = mean(W(:, pos2), 2);

% if the null was made on a smaller ROI than the real model (030209a15), compute the real overlap
% on that same smaller ROI, so real and null use the same pixels and the same 10% thresholds
null_roi  = ~isnan(NullW(:, 1, 1));
real_roi  = ~isnan(firstpeak) & firstpeak ~= 0;
r.roi_restricted = false;
r.n_roi_real_model = sum(real_roi);
if any(real_roi & ~null_roi)
    if any(null_roi & ~real_roi)
        error('%s: the null ROI is not inside the real-model ROI.', name);
    end
    firstpeak(~null_roi)  = NaN;
    secondpeak(~null_roi) = NaN;
    r.roi_restricted = true;
    fprintf('NOTE %s: real overlap computed on the null ROI (%d of %d pixels)\n', ...
            name, sum(null_roi), r.n_roi_real_model);
end
r.n_roi = sum(~isnan(firstpeak) & firstpeak ~= 0);
ov_real = yt_overlapFromMaps(firstpeak, secondpeak);

% --- null overlaps ---
null_overall = zeros(1, n_perm);
null_combo   = zeros(n_perm, 4);
null_nvalid  = zeros(1, n_perm);
for p = 1:n_perm
    f1 = mean(double(NullW(:, npos1, p)), 2);
    f2 = mean(double(NullW(:, npos2, p)), 2);
    ov = yt_overlapFromMaps(f1, f2);
    null_overall(p)  = ov.overall;
    null_combo(p, :) = ov.combo;
    null_nvalid(p)   = sum(~isnan(f1) & f1 ~= 0);
    if p == 1; ov_null_example = ov; end
end

% the null maps must now cover exactly the same pixels as the real map
if any(null_nvalid ~= r.n_roi)
    warning('%s: null maps have a different number of valid pixels (%d-%d) than the real map (%d).', ...
            name, min(null_nvalid), max(null_nvalid), r.n_roi);
end

% --- stats ---
r.name          = name;
r.matpath       = char(matpath);
r.first_frames  = first_frames;
r.second_frames = second_frames;
r.n_perm        = n_perm;
r.real          = ov_real.overall;
r.null          = null_overall;
r.null_mean     = mean(null_overall);
r.null_std      = std(null_overall);
r.z             = (r.real - r.null_mean) / r.null_std;
[r.p_greater, r.p_two] = pstats(r.real, null_overall);

r.combo_names     = ov_real.combo_names;
r.combo_real      = ov_real.combo;
r.combo_null      = null_combo;
r.combo_null_mean = mean(null_combo, 1);
r.combo_null_std  = std(null_combo, 0, 1);
r.combo_z         = (r.combo_real - r.combo_null_mean) ./ r.combo_null_std;
r.combo_p_two     = zeros(1,4);
for i = 1:4
    [~, r.combo_p_two(i)] = pstats(r.combo_real(i), null_combo(:, i));
end

fprintf('%-60s real %5.1f%% | null %5.1f +- %4.1f%% | z %6.2f | p(two) %.3f\n', ...
        name, r.real, r.null_mean, r.null_std, r.z, r.p_two);

if ~make_figs; return; end

% ===================== figures =====================
% 1) null distribution of the overall overlap
figure('Name', ['Overlap null - ' name], 'Color', 'w');
histogram(null_overall, 20, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); hold on;
yl = ylim;
plot([r.real r.real], yl, 'r', 'LineWidth', 2.5);
plot([r.null_mean r.null_mean], yl, '--k', 'LineWidth', 1.2);
xlabel('Overlap (%)'); ylabel('Shuffled models');
title(sprintf('%s\nreal %.1f%% vs null %.1f +/- %.1f%%  (z = %.2f, p = %.3f, %d shuffles)', ...
      name, r.real, r.null_mean, r.null_std, r.z, r.p_two, n_perm), 'Interpreter', 'none');
legend({'null (shuffled labels)', 'real', 'null mean'}, 'Location', 'best'); box off; hold off;

% 2) the 4 sign combinations
figure('Name', ['Sign-combo null - ' name], 'Color', 'w');
for i = 1:4
    subplot(2,2,i);
    histogram(null_combo(:, i), 20, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none'); hold on;
    yl = ylim;
    plot([r.combo_real(i) r.combo_real(i)], yl, 'r', 'LineWidth', 2.5);
    title(sprintf('%s\nreal %.1f%% | null %.1f%% | p = %.3f', r.combo_names{i}, ...
          r.combo_real(i), r.combo_null_mean(i), r.combo_p_two(i)));
    xlabel('Overlap (%)'); box off; hold off;
end

% 3) real vs one shuffled model: the masks themselves
grey = [0.95 0.95 0.95]; white = [1 1 1];
c1 = [0.40 0.80 0.40]; c2 = [1.00 0.71 1.00]; cov = [0 0 0];
figure('Name', ['Real vs shuffled masks - ' name], 'Color', 'w');
both = {ov_real, ov_null_example};
ttl  = {sprintf('Real: overlap %.1f%%', r.real), sprintf('Shuffle #1: overlap %.1f%%', null_overall(1))};
valid = ~isnan(firstpeak) & firstpeak ~= 0;
for k = 1:2
    o  = both{k};
    e1 = o.top1 | o.bot1; e2 = o.top2 | o.bot2;
    m  = zeros(10000,1);
    m(valid) = 1; m(e1 & ~e2) = 2; m(e2 & ~e1) = 3; m(e1 & e2) = 4;
    subplot(1,2,k);
    imagesc(reshape(m,100,100)');
    colormap(gca, [white; grey; c1; c2; cov]); caxis([0 4]); axis image; axis off;
    title(ttl{k});
end
end


function [p_greater, p_two] = pstats(real, null)
% empirical p-values with the +1 correction (the real model counts as one draw under H0)
n = numel(null);
p_greater = (sum(null >= real) + 1) / (n + 1);
p_less    = (sum(null <= real) + 1) / (n + 1);
p_two     = min(1, 2 * min(p_greater, p_less));
end
