function [figEx, figGrp] = yt_plotOverlapNullFigures(R, p_group, q_group, example)
%YT_PLOTOVERLAPNULLFIGURES  Thesis figures for the shuffle-model overlap null.
%   [figEx, figGrp] = yt_plotOverlapNullFigures(R, p_group, q_group, example)
%
%   R, p_group, q_group : left in the workspace by yt_shuffleOverlapNull_batch
%   example             : session tag of the example session, e.g. '110209a15'
%
%   Figure 1 - example session
%     A  real model: phase-1 / phase-2 extreme-weight pixels and their overlap (map)
%     B  one shuffled-label model, same windows (the shuffle whose overlap is closest to the null mean)
%     C  overall overlap: real vs the distribution of the 100 shuffled models
%     D  the 4 sign combinations: real vs null (mean, 95% range)
%   Figure 2 - all sessions
%     A  overall overlap, real vs own null mean, one line per session (Wilcoxon over sessions)
%     B  each session's real overlap in null-sd units (z)
%     C  the 4 sign combinations: real - null mean per session (dots), mean +/- SEM, BH-FDR q
%
%   Colours as in plotModelFigures: phase 1 green (dark = top 10% / positive, light = bottom 10% /
%   negative), phase 2 pink (dark = top, light = bottom), overlap black.

if nargin < 4 || isempty(example); example = '110209a15'; end

% ---------- colours (plotModelFigures) ----------
C.p1top = [0.00 0.50 0.20];  C.p1bot = [0.60 0.90 0.60];
C.p2top = [1.00 0.00 1.00];  C.p2bot = [1.00 0.50 0.80];
C.p1    = [0.40 0.80 0.40];  C.p2    = [1.00 0.71 1.00];
C.ov    = [0 0 0];           C.roi   = [0.93 0.93 0.93];  C.bg = [1 1 1];
C.null  = [0.62 0.62 0.62];  C.real  = [0.85 0.10 0.10];
combo_cols = {C.p1top, C.p2top; C.p1top, C.p2bot; C.p1bot, C.p2top; C.p1bot, C.p2bot};
combo_lbl  = {'P1 + / P2 +', 'P1 + / P2 -', 'P1 - / P2 +', 'P1 - / P2 -'};

tags = cellfun(@(s) regexprep(s, '^.*?(\d{6}[a-z]\d{2}).*$', '$1'), {R.name}, 'UniformOutput', false);
n = numel(R);

% =====================================================================================
% FIGURE 1 - example session
% =====================================================================================
e = find(strcmp(tags, example), 1);
if isempty(e); error('example session %s not in R (have: %s)', example, strjoin(tags, ', ')); end
ex = R(e);

S = load(ex.matpath, 'Weights_acrossT', 'centers', 'NullWeights_acrossT', 'null_w_centers');
W = double(S.Weights_acrossT);  cen = S.centers(:)';  nc = S.null_w_centers(:)';
f1 = mean(W(:, ismember(cen, ex.first_frames)), 2);
f2 = mean(W(:, ismember(cen, ex.second_frames)), 2);
null_roi = ~isnan(S.NullWeights_acrossT(:, 1, 1));
if isfield(ex, 'roi_restricted') && ex.roi_restricted
    f1(~null_roi) = NaN;  f2(~null_roi) = NaN;
end
ov_real = yt_overlapFromMaps(f1, f2);
[~, pk] = min(abs(ex.null - ex.null_mean));                       % representative shuffle
g1 = mean(double(S.NullWeights_acrossT(:, ismember(nc, ex.first_frames), pk)), 2);
g2 = mean(double(S.NullWeights_acrossT(:, ismember(nc, ex.second_frames), pk)), 2);
ov_null = yt_overlapFromMaps(g1, g2);
valid = ~isnan(f1) & f1 ~= 0;

figEx = figure('Name', ['Overlap null - example ' example], 'Color', 'w', 'Position', [80 80 1250 760]);

% A / B: maps
ax = subplot(2, 3, 1);
overlap_map(ax, ov_real, valid, C);
title(ax, {sprintf('A   Real decoder (%s)', example), sprintf('overlap %.1f%%', ex.real)}, 'FontWeight', 'normal');
ax = subplot(2, 3, 2);
overlap_map(ax, ov_null, valid, C);
title(ax, {'B   Shuffled-label decoder', sprintf('overlap %.1f%%  (shuffle #%d)', ex.null(pk), pk)}, 'FontWeight', 'normal');
% map legend
ax = subplot(2, 3, 3); axis(ax, 'off'); hold(ax, 'on');
items = {C.p1, 'phase 1 only'; C.p2, 'phase 2 only'; C.ov, 'both phases (overlap)'; C.roi, 'other ROI pixels'};
for k = 1:size(items, 1)
    y = 0.85 - 0.11 * k;
    patch(ax, [0.05 0.13 0.13 0.05], [y y y+0.07 y+0.07], items{k,1}, 'EdgeColor', [0.6 0.6 0.6]);
    text(ax, 0.17, y + 0.035, items{k,2}, 'FontSize', 10);
end
text(ax, 0.05, 0.86, 'Extreme-weight pixels (top + bottom 10%)', 'FontSize', 10, 'FontWeight', 'bold');
text(ax, 0.05, 0.22, {sprintf('phase 1: centers %s (%g ms)', range_txt(ex.first_frames), center_ms(ex.first_frames)), ...
                      sprintf('phase 2: centers %s (%g ms)', range_txt(ex.second_frames), center_ms(ex.second_frames))}, ...
     'FontSize', 9, 'Color', [0.3 0.3 0.3]);
set(ax, 'XLim', [0 1], 'YLim', [0 1]);

% C: overall null distribution
ax = subplot(2, 3, 4); hold(ax, 'on');
edges = linspace(min([ex.null ex.real]) - 0.5, max([ex.null ex.real]) + 0.5, 26);
histogram(ax, ex.null, edges, 'FaceColor', C.null, 'EdgeColor', 'w');
yl = ylim(ax);
plot(ax, [ex.null_mean ex.null_mean], yl, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2);
plot(ax, [ex.real ex.real], yl, '-', 'Color', C.real, 'LineWidth', 2.5);
text(ax, ex.real, yl(2) * 0.97, sprintf(' real\n %.1f%%', ex.real), 'Color', C.real, 'VerticalAlignment', 'top', 'FontSize', 9);
xlabel(ax, 'Phase-1 / phase-2 overlap (%)'); ylabel(ax, 'Shuffled-label decoders');
title(ax, {'C   Overall overlap vs shuffle null', sprintf('null %.1f \\pm %.1f%%,  z = %.1f,  %d shuffles', ...
      ex.null_mean, ex.null_std, ex.z, ex.n_perm)}, 'FontWeight', 'normal');
box(ax, 'off'); set(ax, 'TickDir', 'out');

% D: sign combinations
ax = subplot(2, 3, [5 6]); hold(ax, 'on');
for i = 1:4
    nd = ex.combo_null(:, i);
    lo = prctile_(nd, 2.5);  hi = prctile_(nd, 97.5);
    patch(ax, [i-0.18 i+0.18 i+0.18 i-0.18], [lo lo hi hi], C.null, 'EdgeColor', 'none', 'FaceAlpha', 0.55);
    plot(ax, [i-0.18 i+0.18], [1 1] * mean(nd), '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 2);
    plot(ax, i, ex.combo_real(i), 'd', 'MarkerSize', 10, 'MarkerFaceColor', C.real, 'MarkerEdgeColor', 'k');
end
set(ax, 'XLim', [0.4 4.6], 'XTick', []);
yl = [0 max([ex.combo_real(:); ex.combo_null(:)]) * 1.15];  ylim(ax, yl);
combo_axis_labels(ax, combo_cols, combo_lbl, yl);
ylabel(ax, 'Overlap (%)');
title(ax, 'D   Overlap by weight sign:  real (red)  vs  shuffle null (grey: 95% range, line: mean)', 'FontWeight', 'normal');
box(ax, 'off'); set(ax, 'TickDir', 'out');

% =====================================================================================
% FIGURE 2 - all sessions
% =====================================================================================
real_ov = [R.real];  null_mean = [R.null_mean];  null_std = [R.null_std];  z = [R.z];
restricted = false(1, n);
if isfield(R, 'roi_restricted'); restricted = [R.roi_restricted]; end
day = cellfun(@(s) s(1:6), tags, 'UniformOutput', false);
combo_real = reshape([R.combo_real], 4, [])';           % n x 4
combo_null = reshape([R.combo_null_mean], 4, [])';
d = combo_real - combo_null;

figGrp = figure('Name', 'Overlap null - all sessions', 'Color', 'w', 'Position', [100 100 1250 760]);

% A: paired real vs null
ax = subplot(2, 2, 1); hold(ax, 'on');
for m = 1:n
    plot(ax, [1 2], [null_mean(m) real_ov(m)], '-', 'Color', [0.75 0.75 0.75]);
end
errorbar(ax, ones(1, n), null_mean, null_std, 'o', 'Color', C.null, 'MarkerFaceColor', C.null, 'MarkerSize', 5, 'CapSize', 0);
plot(ax, 2 * ones(1, n), real_ov, 'o', 'MarkerFaceColor', C.real, 'MarkerEdgeColor', 'none', 'MarkerSize', 6);
plot(ax, [0.8 1.2], [1 1] * mean(null_mean), 'k-', 'LineWidth', 2.5);
plot(ax, [1.8 2.2], [1 1] * mean(real_ov), 'k-', 'LineWidth', 2.5);
set(ax, 'XTick', [1 2], 'XTickLabel', {'shuffle null', 'real'}, 'XLim', [0.5 2.5], 'TickDir', 'out');
ylabel(ax, 'Phase-1 / phase-2 overlap (%)');
title(ax, {'A   Overall overlap, each session vs its own null', ...
      sprintf('%d sessions,  Wilcoxon p = %s', n, p_txt(p_group(1)))}, 'FontWeight', 'normal');
box(ax, 'off');

% B: z per session
ax = subplot(2, 2, 2); hold(ax, 'on');
days = unique(day, 'stable');
day_col = {[0.35 0.35 0.35], [0.70 0.70 0.70]};
for m = 1:n
    col = day_col{find(strcmp(days, day{m}), 1)};
    bar(ax, m, z(m), 0.75, 'FaceColor', col, 'EdgeColor', 'none');
end
plot(ax, [0.4 n + 0.6], [1.96 1.96], '--', 'Color', C.real);
plot(ax, [0.4 n + 0.6], [0 0], 'k-');
lbl = tags;  lbl(restricted) = strcat(lbl(restricted), '*');
set(ax, 'XTick', 1:n, 'XTickLabel', lbl, 'XTickLabelRotation', 50, 'XLim', [0.4 n + 0.6], 'TickDir', 'out', 'FontSize', 8);
ylabel(ax, 'z  (real - null mean) / null sd', 'FontSize', 10);
title(ax, {'B   Real overlap relative to each session''s null', ...
      sprintf('%d / %d sessions above z = 1.96 (dashed);  dark / light = recording day', sum(z > 1.96), n)}, ...
      'FontWeight', 'normal', 'FontSize', 10);
box(ax, 'off');

% C: sign combinations, real - null
ax = subplot(2, 2, [3 4]); hold(ax, 'on');
plot(ax, [0.4 4.6], [0 0], 'k-');
for i = 1:4
    jit = (rand_(n, i) - 0.5) * 0.25;
    plot(ax, i + jit, d(:, i), 'o', 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
    mu = mean(d(:, i));  se = std(d(:, i)) / sqrt(n);
    patch(ax, [i-0.22 i+0.22 i+0.22 i-0.22], [0 0 mu mu], combo_cols{i, 2}, 'FaceAlpha', 0.35, 'EdgeColor', combo_cols{i, 2}, 'LineWidth', 1.5);
    plot(ax, [i i], mu + [-se se], 'k-', 'LineWidth', 1.5);
    yq = max(abs(d(:)));
    text(ax, i, sign(mu) * yq * 1.12, sprintf('q = %s', p_txt(q_group(i + 1))), 'HorizontalAlignment', 'center', 'FontSize', 9);
end
yl = [-1 1] * max(abs(d(:))) * 1.25;  ylim(ax, yl);
set(ax, 'XLim', [0.4 4.6], 'XTick', [], 'TickDir', 'out');
combo_axis_labels(ax, combo_cols, combo_lbl, yl);
ylabel(ax, 'Real - null overlap (%)');
text(ax, 4.55, yl(2) * 0.95, 'more than chance', 'HorizontalAlignment', 'right', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
text(ax, 4.55, yl(1) * 0.95, 'less than chance', 'HorizontalAlignment', 'right', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
title(ax, {'C   Overlap by weight sign (bars: mean \pm SEM over sessions, dots: sessions)', ...
      'same sign in both phases  vs  sign reversed between phases'}, 'FontWeight', 'normal');
box(ax, 'off');
if any(restricted)
    annotation(figGrp, 'textbox', [0.01 0.005 0.9 0.03], 'String', ...
        '* real and null overlap computed on the permutation ROI (smaller than the real-model ROI)', ...
        'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
end
end


% =====================================================================================
function overlap_map(ax, ov, valid, C)
e1 = ov.top1 | ov.bot1;  e2 = ov.top2 | ov.bot2;
m = zeros(10000, 1);
m(valid) = 1;  m(e1 & ~e2) = 2;  m(e2 & ~e1) = 3;  m(e1 & e2) = 4;
imagesc(ax, reshape(m, 100, 100)');
colormap(ax, [C.bg; C.roi; C.p1; C.p2; C.ov]);
caxis(ax, [0 4]);  axis(ax, 'image');  axis(ax, 'off');
end

function combo_axis_labels(ax, combo_cols, combo_lbl, yl)
% two coloured squares (phase-1 sign, phase-2 sign) + text under each group
h = diff(yl);
for i = 1:4
    y0 = yl(1) - 0.10 * h;
    patch(ax, [i-0.20 i-0.04 i-0.04 i-0.20], [y0 y0 y0+0.06*h y0+0.06*h], combo_cols{i,1}, 'EdgeColor', 'none', 'Clipping', 'off');
    patch(ax, [i+0.04 i+0.20 i+0.20 i+0.04], [y0 y0 y0+0.06*h y0+0.06*h], combo_cols{i,2}, 'EdgeColor', 'none', 'Clipping', 'off');
    text(ax, i, yl(1) - 0.17 * h, combo_lbl{i}, 'HorizontalAlignment', 'center', 'FontSize', 10, 'Clipping', 'off');
end
text(ax, 2.5, yl(1) - 0.27 * h, '+ = top 10% (face-preferring) weights,  - = bottom 10% weights;  left square = phase 1, right = phase 2', ...
     'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.4 0.4 0.4], 'Clipping', 'off');
end

function s = p_txt(p)
if p < 0.001; s = sprintf('%.1e', p); else; s = sprintf('%.3f', p); end
end

function s = range_txt(f)
s = sprintf('%d-%d', min(f), max(f));
end

function t = center_ms(f)
t = (mean(f) - 27) * 10;      % zero_frame 27, 10 ms frames (plotModelFigures time axis)
end

function v = prctile_(x, p)
% percentile without the Statistics Toolbox (linear interpolation)
x = sort(x(:));  n = numel(x);
r = 1 + (n - 1) * p / 100;
lo = floor(r);  hi = ceil(r);
v = x(lo) + (x(hi) - x(lo)) * (r - lo);
end

function r = rand_(n, seed)
% fixed jitter so the figure looks the same every time
r = mod((1:n)' * 0.6180339887 + seed * 0.37, 1);
end
