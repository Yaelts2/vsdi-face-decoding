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
%     D  5 paired panels - overall overlap and the 4 sign combinations: each session's null mean
%        connected to its real overlap, group mean +/- SEM; Wilcoxon p (overall) / BH-FDR q (signs)
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
% D - all sessions: 5 paired panels (overall + 4 sign combinations), null -> real per session
% =====================================================================================
real_ov = [R.real];  null_mean = [R.null_mean];
restricted = false(1, n);
if isfield(R, 'roi_restricted'); restricted = [R.roi_restricted]; end
combo_real = reshape([R.combo_real], 4, [])';             % n x 4
combo_null = reshape([R.combo_null_mean], 4, [])';

vals_null = [null_mean(:), combo_null];
vals_real = [real_ov(:),  combo_real];
p_show    = [p_group(1), q_group(2:5)];
ttl       = [{'Overall'}, combo_lbl];
w = 0.155;  gap = 0.04;  x0 = 0.06;
for k = 1:5
    ax = axes('Position', [x0 + (k - 1) * (w + gap), 0.17, w, 0.225]);
    paired_panel(ax, vals_null(:, k), vals_real(:, k), p_show(k), C, FS);
    if k == 1
        ylabel(ax, 'Phase-1 / phase-2 overlap (%)', 'FontSize', FS);
        stat = 'Wilcoxon p';
    else
        stat = 'q';
        yl = get(ax, 'YLim');
        combo_squares(ax, combo_cols(k - 1, :), yl);
    end
    n_up = sum(vals_real(:, k) > vals_null(:, k));
    title(ax, {ttl{k}, sprintf('%d / %d up,  %s = %s', n_up, n, stat, p_txt(p_show(k)))}, ...
          'FontWeight', 'normal', 'FontSize', FS);
end
annotation(fig, 'textbox', [0.20 0.465 0.78 0.04], 'String', ...
    ['D   grey: shuffle-null mean, red: real overlap, one line per session;  black: mean \pm SEM;  ' ...
     'overall: Wilcoxon over sessions, sign combinations: BH-FDR q'], ...
    'EdgeColor', 'none', 'FontSize', FS, 'Color', [0.25 0.25 0.25]);

% footnotes
note = '+ = top 10% (face-preferring) weights,  - = bottom 10% weights;  left square = phase 1, right square = phase 2.';
if any(restricted)
    note = [note sprintf('   %s: real and null overlap computed on the permutation ROI.', strjoin(tags(restricted), ', '))];
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

function paired_panel(ax, vnull, vreal, p, C, FS)
% one paired panel: null mean (x=1) -> real (x=2) per session, group mean +/- SEM, significance mark
hold(ax, 'on');
n = numel(vnull);
for m = 1:n
    if vreal(m) > vnull(m); lc = [0.55 0.55 0.55]; else; lc = [0.82 0.82 0.82]; end
    plot(ax, [1 2], [vnull(m) vreal(m)], '-', 'Color', lc, 'LineWidth', 1);
end
plot(ax, ones(1, n), vnull, 'o', 'MarkerSize', 5, 'MarkerFaceColor', C.null, 'MarkerEdgeColor', [0.4 0.4 0.4]);
plot(ax, 2 * ones(1, n), vreal, 'o', 'MarkerSize', 5, 'MarkerFaceColor', C.real, 'MarkerEdgeColor', 'k');
mn = mean(vnull);  mr = mean(vreal);
sn = std(vnull) / sqrt(n);  sr = std(vreal) / sqrt(n);
plot(ax, [0.62 0.80], [mn mn], 'k-', 'LineWidth', 2.5);  plot(ax, [0.71 0.71], mn + [-sn sn], 'k-', 'LineWidth', 1.5);
plot(ax, [2.20 2.38], [mr mr], 'k-', 'LineWidth', 2.5);  plot(ax, [2.29 2.29], mr + [-sr sr], 'k-', 'LineWidth', 1.5);
lo = min([vnull(:); vreal(:)]);  hi = max([vnull(:); vreal(:)]);  rg = max(hi - lo, 1);
yl = [max(0, lo - 0.08 * rg), hi + 0.22 * rg];
yb = hi + 0.08 * rg;
plot(ax, [1 1 2 2], [yb - 0.03 * rg, yb, yb, yb - 0.03 * rg], 'k-');
text(ax, 1.5, yb + 0.01 * rg, p_star(p), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', FS + 1);
set(ax, 'XTick', [1 2], 'XTickLabel', {'null', 'real'}, 'XLim', [0.5 2.5], 'YLim', yl, 'TickDir', 'out', 'FontSize', FS);
box(ax, 'off');
end

function combo_squares(ax, cols, yl)
% phase-1 / phase-2 sign squares under the x axis
h = diff(yl);  y0 = yl(1) - 0.24 * h;
patch(ax, [1.30 1.47 1.47 1.30], [y0 y0 y0 + 0.07 * h y0 + 0.07 * h], cols{1}, 'EdgeColor', 'none', 'Clipping', 'off');
patch(ax, [1.53 1.70 1.70 1.53], [y0 y0 y0 + 0.07 * h y0 + 0.07 * h], cols{2}, 'EdgeColor', 'none', 'Clipping', 'off');
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

function s = p_star(p)
if p < 0.001; s = '***'; elseif p < 0.01; s = '**'; elseif p < 0.05; s = '*'; else; s = 'n.s.'; end
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
