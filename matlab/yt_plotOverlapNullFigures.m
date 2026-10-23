function fig = yt_plotOverlapNullFigures(R, p_group, q_group, example)
%YT_PLOTOVERLAPNULLFIGURES  One thesis figure for the shuffle-model overlap null.
%   fig = yt_plotOverlapNullFigures(R, p_group, q_group, example)
%
%   R, p_group, q_group : left in the workspace by yt_shuffleOverlapNull_batch
%   example             : session tag of the example session, e.g. '110209a15'
%
%   Top row - example session
%     A  extreme-weight overlap map of one shuffled-label decoder (the shuffle whose overlap is
%        closest to the null mean), same peak windows as the real decoder
%     B  overall overlap: real vs the 100 shuffled-label decoders
%     C  overlap by weight sign: real vs null (95% range + mean)
%   Bottom row - all sessions
%     D  per session: real - null-mean overlap (percentage points), with the spread of that
%        session's own shuffles (grey, 95% range around 0); group Wilcoxon p
%     E  overlap by weight sign: real - null mean per session (dots), mean +/- SEM, BH-FDR q
%
%   Colours as in plotModelFigures: phase 1 green (dark = top 10%, light = bottom 10%),
%   phase 2 pink (dark = top, light = bottom), overlap black.

if nargin < 4 || isempty(example); example = '110209a15'; end

% ---------- colours (plotModelFigures) ----------
C.p1top = [0.00 0.50 0.20];  C.p1bot = [0.60 0.90 0.60];
C.p2top = [1.00 0.00 1.00];  C.p2bot = [1.00 0.50 0.80];
C.p1    = [0.40 0.80 0.40];  C.p2    = [1.00 0.71 1.00];
C.ov    = [0 0 0];           C.roi   = [0.95 0.95 0.95];  C.bg = [1 1 1];
C.null  = [0.70 0.70 0.70];  C.real  = [0.85 0.10 0.10];
combo_cols = {C.p1top, C.p2top; C.p1top, C.p2bot; C.p1bot, C.p2top; C.p1bot, C.p2bot};
combo_lbl  = {'P1 + / P2 +', 'P1 + / P2 -', 'P1 - / P2 +', 'P1 - / P2 -'};
FS = 9;                                                   % base font size

tags = cellfun(@(s) regexprep(s, '^.*?(\d{6}[a-z]\d{2}).*$', '$1'), {R.name}, 'UniformOutput', false);
n = numel(R);
e = find(strcmp(tags, example), 1);
if isempty(e); error('example session %s not in R (have: %s)', example, strjoin(tags, ', ')); end
ex = R(e);

fig = figure('Name', 'Overlap vs shuffle null', 'Color', 'w', 'Position', [60 40 1350 900]);

% row headers
annotation(fig, 'textbox', [0.01 0.955 0.6 0.04], 'String', sprintf('Example session  %s', example), ...
           'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold');
annotation(fig, 'textbox', [0.01 0.465 0.6 0.04], 'String', sprintf('All sessions  (n = %d)', n), ...
           'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold');

% =====================================================================================
% A - one shuffled-label decoder map
% =====================================================================================
S = load(ex.matpath, 'NullWeights_acrossT', 'null_w_centers');
nc = S.null_w_centers(:)';
[~, pk] = min(abs(ex.null - ex.null_mean));               % representative shuffle
g1 = mean(double(S.NullWeights_acrossT(:, ismember(nc, ex.first_frames), pk)), 2);
g2 = mean(double(S.NullWeights_acrossT(:, ismember(nc, ex.second_frames), pk)), 2);
ov_null = yt_overlapFromMaps(g1, g2);
valid = ~isnan(g1) & g1 ~= 0;

ax = axes('Position', [0.03 0.585 0.20 0.28]);
overlap_map(ax, ov_null, valid, C);
title(ax, {'A   Shuffled-label decoder', sprintf('overlap %.1f%%  (example shuffle)', ex.null(pk))}, ...
      'FontWeight', 'normal', 'FontSize', FS + 1);
% compact colour key under the map
axk = axes('Position', [0.03 0.53 0.20 0.045]); axis(axk, 'off'); hold(axk, 'on');
keys = {C.p1, 'phase 1'; C.p2, 'phase 2'; C.ov, 'both'};
for k = 1:3
    x0 = (k - 1) * 0.34;
    patch(axk, x0 + [0 0.07 0.07 0], [0.3 0.3 0.8 0.8], keys{k,1}, 'EdgeColor', [0.6 0.6 0.6]);
    text(axk, x0 + 0.09, 0.55, keys{k,2}, 'FontSize', FS);
end
set(axk, 'XLim', [0 1], 'YLim', [0 1]);

% =====================================================================================
% B - overall overlap, real vs null distribution
% =====================================================================================
ax = axes('Position', [0.31 0.60 0.22 0.265]); hold(ax, 'on');
edges = linspace(min([ex.null ex.real]) - 0.5, max([ex.null ex.real]) + 0.8, 26);
histogram(ax, ex.null, edges, 'FaceColor', C.null, 'EdgeColor', 'w');
yl = ylim(ax);
plot(ax, [ex.null_mean ex.null_mean], yl, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2);
plot(ax, [ex.real ex.real], yl, '-', 'Color', C.real, 'LineWidth', 2.5);
text(ax, ex.real, yl(2) * 0.98, sprintf('real %.1f%% ', ex.real), 'Color', C.real, ...
     'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', 'FontSize', FS);
xlabel(ax, 'Phase-1 / phase-2 overlap (%)', 'FontSize', FS);
ylabel(ax, 'Shuffled-label decoders', 'FontSize', FS);
title(ax, {'B   Overall overlap', sprintf('real %.1f%%  vs  null %.1f \\pm %.1f%%  (%d shuffles)', ...
      ex.real, ex.null_mean, ex.null_std, ex.n_perm)}, 'FontWeight', 'normal', 'FontSize', FS + 1);
set(ax, 'TickDir', 'out', 'FontSize', FS); box(ax, 'off');

% =====================================================================================
% C - example session by weight sign
% =====================================================================================
ax = axes('Position', [0.60 0.60 0.38 0.265]); hold(ax, 'on');
for i = 1:4
    nd = ex.combo_null(:, i);
    lo = prctile_(nd, 2.5);  hi = prctile_(nd, 97.5);
    patch(ax, [i-0.2 i+0.2 i+0.2 i-0.2], [lo lo hi hi], C.null, 'EdgeColor', 'none');
    plot(ax, [i-0.2 i+0.2], [1 1] * mean(nd), '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 2);
    plot(ax, i, ex.combo_real(i), 'd', 'MarkerSize', 10, 'MarkerFaceColor', C.real, 'MarkerEdgeColor', 'k');
end
yl = [0 max([ex.combo_real(:); ex.combo_null(:)]) * 1.15];
set(ax, 'XLim', [0.4 4.6], 'YLim', yl, 'XTick', [], 'TickDir', 'out', 'FontSize', FS);
combo_axis_labels(ax, combo_cols, combo_lbl, yl, FS);
ylabel(ax, 'Overlap (%)', 'FontSize', FS);
title(ax, {'C   Overlap by weight sign', 'red: real   |   grey: shuffle null (95% range, line = mean)'}, ...
      'FontWeight', 'normal', 'FontSize', FS + 1);
box(ax, 'off');

% =====================================================================================
% D - all sessions: real - null, against each session's own null spread
% =====================================================================================
real_ov = [R.real];  null_mean = [R.null_mean];
restricted = false(1, n);
if isfield(R, 'roi_restricted'); restricted = [R.roi_restricted]; end
day = cellfun(@(s) s(1:6), tags, 'UniformOutput', false);

% order: by recording day, then as in R
[~, ord] = sort(day);  ord = ord(:)';
ax = axes('Position', [0.06 0.17 0.38 0.225]); hold(ax, 'on');
n_above = 0;
for j = 1:n
    m = ord(j);
    dn = R(m).null - R(m).null_mean;                      % this session's shuffles, around 0
    lo = prctile_(dn, 2.5);  hi = prctile_(dn, 97.5);
    patch(ax, [j-0.3 j+0.3 j+0.3 j-0.3], [lo lo hi hi], C.null, 'EdgeColor', 'none');
    dv = real_ov(m) - null_mean(m);
    plot(ax, j, dv, 'o', 'MarkerSize', 7, 'MarkerFaceColor', C.real, 'MarkerEdgeColor', 'k');
    n_above = n_above + (dv > hi);
end
plot(ax, [0.4 n + 0.6], [0 0], 'k-');
% day separator
days = unique(day);
if numel(days) > 1
    xb = sum(strcmp(day, days{1})) + 0.5;
    plot(ax, [xb xb], [-100 100], ':', 'Color', [0.5 0.5 0.5]);
end
lbl = tags(ord);  lbl(restricted(ord)) = strcat(lbl(restricted(ord)), '*');
ylv = [min([-3, real_ov - null_mean]) - 1, max(real_ov - null_mean) + 1.5];
set(ax, 'XTick', 1:n, 'XTickLabel', lbl, 'XTickLabelRotation', 50, 'XLim', [0.4 n + 0.6], ...
        'YLim', ylv, 'TickDir', 'out', 'FontSize', FS - 1);
ylabel(ax, 'Real - null overlap (% points)', 'FontSize', FS);
title(ax, {'D   Overall overlap, each session vs its own shuffles', ...
      sprintf('red: real  |  grey: 95%% of shuffles  |  %d / %d above the range  |  Wilcoxon p = %s', ...
      n_above, n, p_txt(p_group(1)))}, 'FontWeight', 'normal', 'FontSize', FS + 1);
box(ax, 'off');

% =====================================================================================
% E - all sessions by weight sign
% =====================================================================================
combo_real = reshape([R.combo_real], 4, [])';             % n x 4
combo_null = reshape([R.combo_null_mean], 4, [])';
d = combo_real - combo_null;
ax = axes('Position', [0.53 0.17 0.45 0.225]); hold(ax, 'on');
plot(ax, [0.4 4.6], [0 0], 'k-');
ym = max(abs(d(:)));
for i = 1:4
    mu = mean(d(:, i));  se = std(d(:, i)) / sqrt(n);
    patch(ax, [i-0.25 i+0.25 i+0.25 i-0.25], [0 0 mu mu], [0.85 0.85 0.85], ...
          'EdgeColor', [0.35 0.35 0.35], 'LineWidth', 1);
    jit = (rand_(n, i) - 0.5) * 0.3;
    plot(ax, i + jit, d(:, i), 'o', 'MarkerFaceColor', [0.45 0.45 0.45], 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
    plot(ax, [i i], mu + [-se se], 'k-', 'LineWidth', 2);
    text(ax, i, sign(mu) * ym * 1.13, sprintf('q = %s', p_txt(q_group(i + 1))), ...
         'HorizontalAlignment', 'center', 'FontSize', FS);
end
yl = [-1 1] * ym * 1.3;
set(ax, 'XLim', [0.4 4.6], 'YLim', yl, 'XTick', [], 'TickDir', 'out', 'FontSize', FS);
combo_axis_labels(ax, combo_cols, combo_lbl, yl, FS);
ylabel(ax, 'Real - null overlap (% points)', 'FontSize', FS);
title(ax, {'E   Overlap by weight sign', 'dots: sessions  |  bars: mean \pm SEM  |  above 0 = more than chance'}, ...
      'FontWeight', 'normal', 'FontSize', FS + 1);
box(ax, 'off');

% footnotes
note = '+ = top 10% (face-preferring) weights,  - = bottom 10% weights;  left square = phase 1, right square = phase 2.';
if any(restricted)
    note = [note '   * real and null overlap computed on the permutation ROI.'];
end
annotation(fig, 'textbox', [0.02 0.008 0.96 0.03], 'String', note, 'EdgeColor', 'none', ...
           'FontSize', FS - 1, 'Color', [0.4 0.4 0.4], 'HorizontalAlignment', 'center');
end


% =====================================================================================
function overlap_map(ax, ov, valid, C)
e1 = ov.top1 | ov.bot1;  e2 = ov.top2 | ov.bot2;
m = zeros(10000, 1);
m(valid) = 1;  m(e1 & ~e2) = 2;  m(e2 & ~e1) = 3;  m(e1 & e2) = 4;
imagesc(ax, reshape(m, 100, 100)');
colormap(ax, [C.bg; C.roi; C.p1; C.p2; C.ov]);
caxis(ax, [0 4]);  axis(ax, 'image');  axis(ax, 'off');
% overlap pixels drawn again as outlined squares - exactly as in plotModelFigures
hold(ax, 'on');
[oy, ox] = find(reshape(e1 & e2, 100, 100)');
for k = 1:numel(ox)
    rectangle(ax, 'Position', [ox(k)-0.5, oy(k)-0.5, 1, 1], ...
              'FaceColor', C.ov, 'EdgeColor', 'k', 'LineWidth', 0.3);
end
hold(ax, 'off');
end

function combo_axis_labels(ax, combo_cols, combo_lbl, yl, FS)
% two coloured squares (phase-1 sign, phase-2 sign) + text under each group
h = diff(yl);
for i = 1:4
    y0 = yl(1) - 0.11 * h;
    patch(ax, [i-0.20 i-0.03 i-0.03 i-0.20], [y0 y0 y0+0.06*h y0+0.06*h], combo_cols{i,1}, 'EdgeColor', 'none', 'Clipping', 'off');
    patch(ax, [i+0.03 i+0.20 i+0.20 i+0.03], [y0 y0 y0+0.06*h y0+0.06*h], combo_cols{i,2}, 'EdgeColor', 'none', 'Clipping', 'off');
    text(ax, i, yl(1) - 0.19 * h, combo_lbl{i}, 'HorizontalAlignment', 'center', 'FontSize', FS, 'Clipping', 'off');
end
end

function s = p_txt(p)
if p < 0.001; s = sprintf('%.1e', p); else; s = sprintf('%.3f', p); end
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
