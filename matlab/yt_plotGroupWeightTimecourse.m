function [fig, TC] = yt_plotGroupWeightTimecourse(R, varargin)
%YT_PLOTGROUPWEIGHTTIMECOURSE  Mean weight time course of each extreme-weight pixel group, averaged
%   over sessions. Every session's groups are defined by ITS OWN phase-1 / phase-2 peak windows.
%
%   [fig, TC] = yt_plotGroupWeightTimecourse(R)
%   [fig, TC] = yt_plotGroupWeightTimecourse(R, 'Normalize', false, 'TimeMs', [-100 400])
%
%   R : from yt_shuffleOverlapNull_batch (uses matpath, first_frames, second_frames, roi_restricted)
%
%   Per session: masks from yt_overlapFromMaps on the mean of Weights_acrossT over the phase windows
%   (top / bottom 10% signed weights - the same masks as plotModelFigures and the overlap analysis);
%   for each group the mean weight over its pixels in every window (as plotModelFigures section 4c).
%   Then mean +/- SEM over sessions.
%
%   Groups
%     A  P1 top, P1 bottom, P2 top, P2 bottom                      (the 4 single-sign masks)
%     B  phase 1 only, phase 2 only, overlap (both phases)
%     C  overlap by sign: P1+/P2+, P1-/P2-  (same sign, grey)   P1+/P2-, P1-/P2+  (reversed, black)
%
%   Options
%     'Normalize'  true   divide each session's weights by the SD of all its ROI weights in TimeMs, so
%                         sessions with larger weight scales do not dominate the average
%                         (false = raw weights, as in plotModelFigures)
%     'TimeMs'     [-100 400]   time range shown (ms from stimulus onset)
%
%   TC : struct with t_ms, and for every group the per-session curves (n_sessions x n_windows)

opt = struct('Normalize', true, 'TimeMs', [-100 400]);
for k = 1:2:numel(varargin)
    if ~isfield(opt, varargin{k}); error('unknown option %s', varargin{k}); end
    opt.(varargin{k}) = varargin{k+1};
end

% ---------- colours (plotModelFigures) ----------
c.p1top = [0.00 0.50 0.20];  c.p1bot = [0.60 0.90 0.60];
c.p2top = [1.00 0.00 1.00];  c.p2bot = [1.00 0.50 0.80];
c.p1    = [0.40 0.80 0.40];  c.p2    = [1.00 0.71 1.00];  c.ov = [0 0 0];

groups = { ...
  % name                 panel  colour                 style
    'P1 top',              1,   c.p1top,               '-';
    'P1 bottom',           1,   c.p1bot,               '-';
    'P2 top',              1,   c.p2top,               '--';
    'P2 bottom',           1,   c.p2bot,               '--';
    'phase 1 only',        2,   c.p1,                  '-';
    'phase 2 only',        2,   c.p2,                  '-';
    'overlap',             2,   c.ov,                  '-';
    'P1+ / P2+',           3,   [0.60 0.60 0.60],      '-';
    'P1- / P2-',           3,   [0.60 0.60 0.60],      '--';
    'P1+ / P2-',           3,   [0 0 0],               '--';
    'P1- / P2+',           3,   [0 0 0],               '-'};
ng = size(groups, 1);
n  = numel(R);

curves = [];  t_ms = [];  p1_ms = zeros(n, 2);  p2_ms = zeros(n, 2);  npix = zeros(n, ng);
for m = 1:n
    S = load(R(m).matpath, 'Weights_acrossT', 'centers', 'zero_frame', 'frame_duration_ms', 'NullWeights_acrossT');
    W = double(S.Weights_acrossT);
    cen = S.centers(:)';
    t = (cen - S.zero_frame) * S.frame_duration_ms;
    keep = t >= opt.TimeMs(1) & t <= opt.TimeMs(2);
    if isempty(t_ms); t_ms = t(keep); curves = nan(n, numel(t_ms), ng); end
    if ~isequal(t(keep), t_ms); error('%s: windows differ from the first session', R(m).name); end

    f1 = mean(W(:, ismember(cen, R(m).first_frames)), 2);
    f2 = mean(W(:, ismember(cen, R(m).second_frames)), 2);
    if isfield(R, 'roi_restricted') && R(m).roi_restricted && isfield(S, 'NullWeights_acrossT')
        nr = ~isnan(S.NullWeights_acrossT(:, 1, 1));
        f1(~nr) = NaN;  f2(~nr) = NaN;
    end
    ov = yt_overlapFromMaps(f1, f2);
    e1 = ov.top1 | ov.bot1;  e2 = ov.top2 | ov.bot2;
    masks = {ov.top1, ov.bot1, ov.top2, ov.bot2, ...
             e1 & ~e2, e2 & ~e1, e1 & e2, ...
             ov.top1 & ov.top2, ov.bot1 & ov.bot2, ov.top1 & ov.bot2, ov.bot1 & ov.top2};

    Wk = W(:, keep);
    if opt.Normalize
        roi = ~isnan(f1) & f1 ~= 0;
        sc = std(reshape(Wk(roi, :), [], 1));
        Wk = Wk / sc;
    end
    for g = 1:ng
        npix(m, g) = sum(masks{g});
        if npix(m, g) > 0
            curves(m, :, g) = mean(Wk(masks{g}, :), 1);
        end
    end
    p1_ms(m, :) = ([min(R(m).first_frames)  max(R(m).first_frames)]  - S.zero_frame) * S.frame_duration_ms;
    p2_ms(m, :) = ([min(R(m).second_frames) max(R(m).second_frames)] - S.zero_frame) * S.frame_duration_ms;
end

% ---------- figure ----------
fig = figure('Name', 'Weight time course by pixel group (all sessions)', 'Color', 'w', 'Position', [80 80 1350 430]);
ttl = {'A   Single-sign masks', 'B   Phase-specific vs overlapping pixels', 'C   Overlapping pixels by sign'};
if opt.Normalize; ylab = 'Mean weight (SD units)'; else; ylab = 'Mean weight'; end
allc = curves(:);  allc = allc(~isnan(allc));
yl = [min(allc) max(allc)] + [-0.15 0.15] * max(range_(allc), eps);
for pnl = 1:3
    ax = subplot(1, 3, pnl); hold(ax, 'on');
    % phase windows (range over sessions)
    patch(ax, [min(p1_ms(:,1)) max(p1_ms(:,2)) max(p1_ms(:,2)) min(p1_ms(:,1))], yl([1 1 2 2]), c.p1bot, ...
          'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    patch(ax, [min(p2_ms(:,1)) max(p2_ms(:,2)) max(p2_ms(:,2)) min(p2_ms(:,1))], yl([1 1 2 2]), c.p2, ...
          'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(ax, t_ms([1 end]), [0 0], 'k:', 'HandleVisibility', 'off');
    plot(ax, [0 0], yl, 'k--', 'HandleVisibility', 'off');
    hl = [];  ll = {};
    for g = find(cell2mat(groups(:, 2))' == pnl)
        Y = curves(:, :, g);
        ok = ~any(isnan(Y), 2);                       % sessions where this group has pixels
        Y = Y(ok, :);  ns = size(Y, 1);
        mu = mean(Y, 1);  se = std(Y, 0, 1) / sqrt(ns);
        patch(ax, [t_ms fliplr(t_ms)], [mu - se fliplr(mu + se)], groups{g, 3}, 'FaceAlpha', 0.15, ...
              'EdgeColor', 'none', 'HandleVisibility', 'off');
        lbl = groups{g, 1};
        if ns < n; lbl = sprintf('%s (n = %d)', lbl, ns); end
        hl(end+1) = plot(ax, t_ms, mu, groups{g, 4}, 'Color', groups{g, 3}, 'LineWidth', 2);  %#ok<AGROW>
        ll{end+1} = lbl;                                                                      %#ok<AGROW>
    end
    set(ax, 'XLim', t_ms([1 end]), 'YLim', yl, 'TickDir', 'out', 'FontSize', 9);
    xlabel(ax, 'Time from stimulus onset (ms)');
    if pnl == 1; ylabel(ax, ylab); end
    title(ax, ttl{pnl}, 'FontWeight', 'normal', 'FontSize', 11);
    lg = legend(ax, hl, ll, 'Location', 'southwest');      % explicit handles + labels (R2017b)
    set(lg, 'Box', 'off', 'FontSize', 8);
    box(ax, 'off');
end
annotation(fig, 'textbox', [0.02 0.0 0.96 0.04], 'EdgeColor', 'none', 'FontSize', 8, 'Color', [0.35 0.35 0.35], ...
    'HorizontalAlignment', 'center', 'String', sprintf(['Groups defined per session by its own phase windows ' ...
    '(top / bottom 10%% signed weights); mean \\pm SEM over %d sessions. Shaded: range of phase-1 (green) and ' ...
    'phase-2 (pink) windows across sessions.'], n));

TC.t_ms = t_ms;  TC.names = groups(:, 1)';  TC.curves = curves;  TC.npix = npix;  TC.normalize = opt.Normalize;
end

function r = range_(x)
r = max(x) - min(x);
end
