function fig = yt_plotOverlapNullFigures(R, p_group, q_group, example)
%YT_PLOTOVERLAPNULLFIGURES  One thesis figure for the shuffle-model overlap null: 2 rows x 5 columns.
%   fig = yt_plotOverlapNullFigures(R, p_group, q_group, example)
%
%   R, p_group, q_group : left in the workspace by yt_shuffleOverlapNull_batch
%   example             : session tag of the example session, e.g. '110209a15'
%
%   Columns: overall overlap, then the 4 sign combinations (P1+/P2+, P1+/P2-, P1-/P2+, P1-/P2-).
%   Top row - example session: its 100 shuffled-label decoders (grey dots, black = mean) and the real
%             decoder (red); stars = that session's own empirical test vs its null (two-tailed;
%             with 100 shuffles the smallest possible p is 0.02).
%   Bottom row - all sessions: each session's null mean (grey) connected to its real overlap (red),
%             black = mean +/- SEM; stars = Wilcoxon over sessions (overall) / BH-FDR q (signs).
%   Each column uses the same y-axis in both rows.
%
%   Colours as in plotModelFigures: phase 1 green (dark = top 10%, light = bottom 10%),
%   phase 2 pink (dark = top, light = bottom).

if nargin < 4 || isempty(example); example = '110209a15'; end

C.p1top = [0.00 0.50 0.20];  C.p1bot = [0.60 0.90 0.60];
C.p2top = [1.00 0.00 1.00];  C.p2bot = [1.00 0.50 0.80];
C.null  = [0.70 0.70 0.70];  C.real  = [0.85 0.10 0.10];
combo_cols = {C.p1top, C.p2top; C.p1top, C.p2bot; C.p1bot, C.p2top; C.p1bot, C.p2bot};
col_ttl = {'Overall', 'P1+ / P2+', 'P1+ / P2-', 'P1- / P2+', 'P1- / P2-'};
FS = 9;

tags = cellfun(@(s) regexprep(s, '^.*?(\d{6}[a-z]\d{2}).*$', '$1'), {R.name}, 'UniformOutput', false);
n = numel(R);
e = find(strcmp(tags, example), 1);
if isempty(e); error('example session %s not in R (have: %s)', example, strjoin(tags, ', ')); end
ex = R(e);

% ---------- data per column ----------
ex_null = [ex.null(:), ex.combo_null];                    % n_perm x 5
ex_real = [ex.real, ex.combo_real];                       % 1 x 5
ex_p    = [ex.p_two, ex.combo_p_two];                     % 1 x 5
combo_real = reshape([R.combo_real], 4, [])';
combo_null = reshape([R.combo_null_mean], 4, [])';
g_null = [[R.null_mean]', combo_null];                    % n x 5
g_real = [[R.real]', combo_real];                         % n x 5
g_p    = [p_group(1), q_group(2:5)];
restricted = false(1, n);
if isfield(R, 'roi_restricted'); restricted = [R.roi_restricted]; end

fig = figure('Name', 'Overlap vs shuffle null', 'Color', 'w', 'Position', [60 60 1350 760]);
annotation(fig, 'textbox', [0.01 0.950 0.6 0.04], 'String', sprintf('Example session  %s', example), ...
           'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold');
annotation(fig, 'textbox', [0.01 0.480 0.6 0.04], 'String', sprintf('All sessions  (n = %d)', n), ...
           'EdgeColor', 'none', 'FontSize', 12, 'FontWeight', 'bold');

w = 0.155;  gap = 0.04;  x0 = 0.06;
for k = 1:5
    % same y-range for both rows of this column
    allv = [ex_null(:, k); ex_real(k); g_null(:, k); g_real(:, k)];
    lo = min(allv);  hi = max(allv);  rg = max(hi - lo, 1);
    yl = [max(0, lo - 0.08 * rg), hi + 0.22 * rg];
    xk = x0 + (k - 1) * (w + gap);

    % ---- top: example session ----
    ax = axes('Position', [xk, 0.575, w, 0.33]);
    example_panel(ax, ex_null(:, k), ex_real(k), ex_p(k), yl, C, FS);
    title(ax, col_ttl{k}, 'FontWeight', 'bold', 'FontSize', FS + 2);
    if k == 1; ylabel(ax, 'Phase-1 / phase-2 overlap (%)', 'FontSize', FS); end

    % ---- bottom: all sessions ----
    ax = axes('Position', [xk, 0.135, w, 0.33]);
    paired_panel(ax, g_null(:, k), g_real(:, k), g_p(k), yl, C, FS);
    if k == 1
        ylabel(ax, 'Phase-1 / phase-2 overlap (%)', 'FontSize', FS);
    else
        combo_squares(ax, combo_cols(k - 1, :), yl);
    end
end

note = ['Top: grey = 100 shuffled-label decoders of this session (black = mean), red = real decoder; ' ...
        'stars = test vs its own null.   Bottom: one line per session, null mean (grey) to real (red), ' ...
        'black = mean \pm SEM; stars = Wilcoxon over sessions (overall) / BH-FDR q (signs).'];
note2 = ['+ / - = top / bottom 10% weights;  squares: left = phase 1, right = phase 2.   ' ...
         '* p < 0.05, ** p < 0.01, *** p < 0.001'];
if any(restricted)
    note2 = [note2 sprintf('.   %s: computed on the permutation ROI.', strjoin(tags(restricted), ', '))];
end
annotation(fig, 'textbox', [0.02 0.028 0.96 0.035], 'String', note, 'EdgeColor', 'none', ...
           'FontSize', FS - 1, 'Color', [0.35 0.35 0.35], 'HorizontalAlignment', 'center');
annotation(fig, 'textbox', [0.02 0.002 0.96 0.03], 'String', note2, 'EdgeColor', 'none', ...
           'FontSize', FS - 1, 'Color', [0.35 0.35 0.35], 'HorizontalAlignment', 'center');
end


% =====================================================================================
function example_panel(ax, vnull, vreal, p, yl, C, FS)
% one session: null distribution as jittered grey dots (x = 1), real as red dot (x = 2)
hold(ax, 'on');
np = numel(vnull);
jit = (mod((1:np)' * 0.6180339887, 1) - 0.5) * 0.36;
plot(ax, 1 + jit, vnull, 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', C.null, 'MarkerEdgeColor', 'none');
plot(ax, [0.72 1.28], [1 1] * mean(vnull), 'k-', 'LineWidth', 2.5);
plot(ax, 2, vreal, 'o', 'MarkerSize', 8, 'MarkerFaceColor', C.real, 'MarkerEdgeColor', 'k');
sig_bracket(ax, max([vnull(:); vreal]), yl, p, FS);
set(ax, 'XTick', [1 2], 'XTickLabel', {'null', 'real'}, 'XLim', [0.5 2.5], 'YLim', yl, 'TickDir', 'out', 'FontSize', FS);
box(ax, 'off');
end

function paired_panel(ax, vnull, vreal, p, yl, C, FS)
% all sessions: null mean (x = 1) -> real (x = 2) per session, group mean +/- SEM
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
sig_bracket(ax, max([vnull(:); vreal(:)]), yl, p, FS);
set(ax, 'XTick', [1 2], 'XTickLabel', {'null', 'real'}, 'XLim', [0.5 2.5], 'YLim', yl, 'TickDir', 'out', 'FontSize', FS);
box(ax, 'off');
end

function sig_bracket(ax, top, yl, p, FS)
rg = diff(yl);
yb = top + 0.07 * rg;
plot(ax, [1 1 2 2], [yb - 0.03 * rg, yb, yb, yb - 0.03 * rg], 'k-');
text(ax, 1.5, yb + 0.005 * rg, p_star(p), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', FS + 2);
end

function combo_squares(ax, cols, yl)
% phase-1 / phase-2 sign squares under the x axis
h = diff(yl);  y0 = yl(1) - 0.20 * h;
patch(ax, [1.30 1.47 1.47 1.30], [y0 y0 y0 + 0.06 * h y0 + 0.06 * h], cols{1}, 'EdgeColor', 'none', 'Clipping', 'off');
patch(ax, [1.53 1.70 1.70 1.53], [y0 y0 y0 + 0.06 * h y0 + 0.06 * h], cols{2}, 'EdgeColor', 'none', 'Clipping', 'off');
end

function s = p_star(p)
if p < 0.001; s = '***'; elseif p < 0.01; s = '**'; elseif p < 0.05; s = '*'; else; s = 'n.s.'; end
end
