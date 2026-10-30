function fig = yt_plotShuffleAccuracy(R)
%YT_PLOTSHUFFLEACCURACY  Grand-average decoding accuracy over time: real decoders vs the shuffled-label
%   decoders used for the overlap null.
%   fig = yt_plotShuffleAccuracy(R)        % R from yt_shuffleOverlapNull_batch
%
%   Real     : trial_acc_mean of every session, mean +/- SEM over sessions (all windows).
%   Shuffled : null_trial_acc_perm of every session (100 shuffles x the windows 0-300 ms).
%              Shuffle k is averaged over the sessions -> 100 group-average shuffle curves;
%              shown as their mean and the 2.5-97.5% range.
%   Phase windows: grey bars = the range of the phase-1 / phase-2 windows used across sessions.
%   Time = (center - zero_frame) * frame_duration_ms, as in plotModelFigures.

n = numel(R);
t_real = [];  acc_real = [];
t_null = [];  acc_null = [];        % acc_null: n_perm x n_win x n_sessions
p1_ms = zeros(n, 2);  p2_ms = zeros(n, 2);
for m = 1:n
    S = load(R(m).matpath, 'centers', 'trial_acc_mean', 'zero_frame', 'frame_duration_ms', ...
             'null_trial_acc_perm', 'null_w_centers');
    if ~isfield(S, 'null_trial_acc_perm')
        error('%s has no null_trial_acc_perm - run open_permutation_sliding.py for it.', R(m).name);
    end
    c  = S.centers(:)';  z = S.zero_frame;  fd = S.frame_duration_ms;
    t  = (c - z) * fd;
    tn = (S.null_w_centers(:)' - z) * fd;
    if isempty(t_real); t_real = t; t_null = tn; end
    if ~isequal(t, t_real) || ~isequal(tn, t_null)
        error('%s: windows differ from the first session', R(m).name);
    end
    acc_real(m, :) = S.trial_acc_mean(:)';                       %#ok<AGROW>
    acc_null(:, :, m) = S.null_trial_acc_perm;                   %#ok<AGROW>
    p1_ms(m, :) = ([min(R(m).first_frames)  max(R(m).first_frames)]  - z) * fd;
    p2_ms(m, :) = ([min(R(m).second_frames) max(R(m).second_frames)] - z) * fd;
end

mr = mean(acc_real, 1);  sr = std(acc_real, 0, 1) / sqrt(n);
gnull = mean(acc_null, 3);                       % 100 x n_win: group-average shuffle curves
mn = mean(gnull, 1);
lo = prct(gnull, 2.5);  hi = prct(gnull, 97.5);
outside = mr(ismember(t_real, t_null)) > hi;     % windows where the real mean is above all but 2.5%

C_real = [0.13 0.33 0.68];  C_null = [0.55 0.55 0.55];
fig = figure('Name', 'Accuracy: real vs shuffled decoders', 'Color', 'w', 'Position', [100 100 820 430]);
ax = axes('Position', [0.09 0.14 0.86 0.74]); hold(ax, 'on');
yl = [0.35 1.02];
% phase windows (range over sessions)
patch(ax, [min(p1_ms(:,1)) max(p1_ms(:,2)) max(p1_ms(:,2)) min(p1_ms(:,1))], yl([1 1 2 2]), [0.60 0.90 0.60], ...
      'FaceAlpha', 0.25, 'EdgeColor', 'none');
patch(ax, [min(p2_ms(:,1)) max(p2_ms(:,2)) max(p2_ms(:,2)) min(p2_ms(:,1))], yl([1 1 2 2]), [1.00 0.71 1.00], ...
      'FaceAlpha', 0.25, 'EdgeColor', 'none');
plot(ax, [t_real(1) t_real(end)], [0.5 0.5], ':', 'Color', [0.4 0.4 0.4]);
plot(ax, [0 0], yl, '--k');
% shuffled: 95% range of the group-average curves + mean
patch(ax, [t_null fliplr(t_null)], [lo fliplr(hi)], C_null, 'FaceAlpha', 0.35, 'EdgeColor', 'none');
h_null = plot(ax, t_null, mn, '-', 'Color', C_null, 'LineWidth', 2);
% real: mean +/- SEM
patch(ax, [t_real fliplr(t_real)], [mr - sr fliplr(mr + sr)], C_real, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
h_real = plot(ax, t_real, mr, '-', 'Color', C_real, 'LineWidth', 2.2);
set(ax, 'YLim', yl, 'XLim', [t_real(1) t_real(end)], 'TickDir', 'out', 'FontSize', 10);
xlabel(ax, 'Time from stimulus onset (ms)');  ylabel(ax, 'Trial-level decoding accuracy');
legend(ax, [h_real h_null], {sprintf('real decoders (mean \\pm SEM, n = %d)', n), ...
       sprintf('shuffled-label decoders (group average, %d shuffles, 95%% range)', size(gnull, 1))}, ...
       'Location', 'northeast', 'Box', 'off');
title(ax, sprintf('Real vs shuffled-label decoders   |   real above the shuffle range in %d / %d windows (0-%g ms)', ...
      sum(outside), numel(outside), max(t_null)), 'FontWeight', 'normal');
box(ax, 'off');

fprintf('shuffled decoders, group average over %d sessions: mean accuracy %.3f (range of window means %.3f-%.3f)\n', ...
        n, mean(mn), min(mn), max(mn));
end

function v = prct(X, p)
% column-wise percentile without the Statistics Toolbox
X = sort(X, 1);  k = size(X, 1);
r = 1 + (k - 1) * p / 100;  a = floor(r);  b = ceil(r);
v = X(a, :) + (X(b, :) - X(a, :)) * (r - a);
end
