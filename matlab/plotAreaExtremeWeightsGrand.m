function out = plotAreaExtremeWeightsGrand(db_path_v1, db_path_v2, varargin)
%PLOTAREAEXTREMEWEIGHTSGRAND  Across-session version of plotAreaExtremeWeights (figure 2):
%   each extreme-weight group's mean weight across time, averaged over ALL models of the V1
%   database and, separately, over all models of the V2 database.
%
%   out = plotAreaExtremeWeightsGrand(db_path_v1, db_path_v2)
%   out = plotAreaExtremeWeightsGrand(..., 'QuantileFrac', 0.10, 'Normalize', false, 'Rows', {[], []})
%
%   For every model (database row) of an area - exactly as in plotAreaExtremeWeights:
%     phase windows = that row's OWN peak1_frames / peak2_frames
%     weights       = Weights_acrossT of its model file (mat_path, or <model_id>.mat next to the database)
%     groups        = P1 top / P1 bottom / P2 top / P2 bottom: top / bottom QuantileFrac of the
%                     mean weight over the phase window (nonzero, non-NaN pixels)
%     curve         = mean weight over the group's pixels at every window
%   Then: mean +/- SEM over the models (sessions) of that area. Grey bands = the range of the
%   phase-1 / phase-2 windows across the area's models.
%
%   Options
%     'QuantileFrac' 0.10        tail size (top 10% + bottom 10%)
%     'Normalize'    false       true -> divide each model's weights by the SD of its ROI weights,
%                                so models with larger weight scales do not dominate the average
%     'Rows'         {[], []}    database rows to use for V1 / V2 ([] = all rows)
%
%   out.V1 / out.V2 : t_ms, curves (n_models x n_windows x 4), names, model_ids, frames

opt = struct('QuantileFrac', 0.10, 'Normalize', false, 'Rows', {{[], []}});
for k = 1:2:numel(varargin)
    if ~isfield(opt, varargin{k}); error('unknown option %s', varargin{k}); end
    opt.(varargin{k}) = varargin{k+1};
end

db_paths = {db_path_v1, db_path_v2};
areas    = {'V1', 'V2'};
col_top  = {[0.00 0.50 0.20], [1.00 0.00 1.00]};       % phase 1, phase 2 (plotModelFigures)
col_bot  = {[0.60 0.90 0.60], [1.00 0.50 0.80]};
grp_names = {'P1 top', 'P1 bottom', 'P2 top', 'P2 bottom'};
grp_cols  = {col_top{1}, col_bot{1}, col_top{2}, col_bot{2}};
grp_style = {'-', '--', '-', '--'};

out = struct();
for a = 1:2
    DB = local_load_db(db_paths{a});
    cols = local_cols(DB);
    nrow = numel(DB.(cols{1}));
    rows = opt.Rows{a};
    if isempty(rows); rows = 1:nrow; end
    ids = local_col_text(DB, 'model_id');
    p1c = local_col_cell(DB, 'peak1_frames');
    p2c = local_col_cell(DB, 'peak2_frames');
    mpc = {};
    if any(strcmp(cols, 'mat_path')); mpc = local_col_text(DB, 'mat_path'); end

    t_ms = [];  curves = [];  used = {};  frames_all = {};  p_ms = [];
    for r = rows
        f1 = local_parse_frames(p1c{r});  f2 = local_parse_frames(p2c{r});
        if isempty(f1) || isempty(f2)
            fprintf('%s row %d (%s): no phase frames - skipped\n', areas{a}, r, ids{r});
            continue
        end
        mp = '';
        if ~isempty(mpc); mp = mpc{r}; end
        if isempty(mp) || exist(mp, 'file') ~= 2
            alt = fullfile(fileparts(db_paths{a}), [ids{r} '.mat']);
            if exist(alt, 'file') == 2; mp = alt; end
        end
        if exist(mp, 'file') ~= 2
            fprintf('%s row %d (%s): model file not found - skipped\n', areas{a}, r, ids{r});
            continue
        end
        S = load(mp, 'Weights_acrossT', 'centers', 'zero_frame', 'frame_duration_ms');
        W = double(S.Weights_acrossT);
        if size(W,1) ~= 10000 && size(W,2) == 10000; W = W'; end
        cen = S.centers(:)';
        zf = 27;  fd = 10;
        if isfield(S, 'zero_frame');        zf = S.zero_frame;        end
        if isfield(S, 'frame_duration_ms'); fd = S.frame_duration_ms; end
        t = (cen - zf) * fd;
        if isempty(t_ms); t_ms = t; end
        if ~isequal(t, t_ms)
            fprintf('%s row %d (%s): different windows than the first model - skipped\n', areas{a}, r, ids{r});
            continue
        end

        w1 = mean(W(:, ismember(cen, f1)), 2);
        w2 = mean(W(:, ismember(cen, f2)), 2);
        [t1, b1, v1] = local_tails(w1, opt.QuantileFrac);
        [t2, b2]     = local_tails(w2, opt.QuantileFrac);
        if opt.Normalize
            W = W / std(reshape(W(v1, :), [], 1));
        end
        groups = {t1, b1, t2, b2};
        cv = nan(1, numel(t), 4);
        for g = 1:4
            if any(groups{g}); cv(1, :, g) = mean(W(groups{g}, :), 1); end
        end
        curves = cat(1, curves, cv);
        used{end+1} = ids{r};                                             %#ok<AGROW>
        frames_all(end+1, :) = {f1, f2};                                  %#ok<AGROW>
        p_ms(end+1, :) = ([min(f1) max(f1) min(f2) max(f2)] - zf) * fd;   %#ok<AGROW>
    end
    if isempty(used); error('%s: no usable models', areas{a}); end
    out.(areas{a}).t_ms = t_ms;
    out.(areas{a}).curves = curves;
    out.(areas{a}).names = grp_names;
    out.(areas{a}).model_ids = used;
    out.(areas{a}).frames = frames_all;
    out.(areas{a}).phase_ms = p_ms;
    fprintf('%s: %d models averaged\n', areas{a}, numel(used));
end

%% Figure: mean +/- SEM over sessions, one panel per area -------------------------------
figure('Name', 'Extreme-weight groups across time - all sessions', 'Color', 'w', 'Position', [150 120 1100 800]);
ax = gobjects(1, 2);
for a = 1:2
    A = out.(areas{a});
    ns = size(A.curves, 1);
    ax(a) = subplot(2, 1, a); hold on
    hl = gobjects(1, 4);
    for g = 1:4
        Y = A.curves(:, :, g);
        Y = Y(~any(isnan(Y), 2), :);
        m  = mean(Y, 1);
        se = std(Y, 0, 1) / sqrt(size(Y, 1));
        fill([A.t_ms fliplr(A.t_ms)], [m+se fliplr(m-se)], grp_cols{g}, 'EdgeColor', 'none', 'FaceAlpha', 0.25);
        hl(g) = plot(A.t_ms, m, grp_style{g}, 'Color', grp_cols{g}, 'LineWidth', 2);
    end
    plot(A.t_ms([1 end]), [0 0], 'k:');
    yl = ylim;
    % phase windows: range over this area's models
    P = A.phase_ms;
    patch([min(P(:,1)) max(P(:,2)) max(P(:,2)) min(P(:,1))], yl([1 1 2 2]), [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.12);
    patch([min(P(:,3)) max(P(:,4)) max(P(:,4)) min(P(:,3))], yl([1 1 2 2]), [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.12);
    plot([0 0], yl, 'k--');
    ylim(yl);
    xlabel('Time from stimulus onset (ms)');
    if opt.Normalize; ylabel('Mean weight (SD units)'); else; ylabel('Mean weight'); end
    title(sprintf('%s: extreme-weight groups across time, mean \\pm SEM over %d sessions (grey bands = phase 1, phase 2)', ...
          areas{a}, ns));
    legend(hl, grp_names, 'Location', 'eastoutside');
    hold off
end
linkaxes(ax, 'x');
end


%% ======================== local helpers ================================

function DB = local_load_db(db_path)
S  = load(db_path);
fn = fieldnames(S);
fn = fn(strncmp(fn, 'DB', 2));
assert(~isempty(fn), 'No variable starting with ''DB'' in %s', db_path);
DB = S.(fn{1});
end

function c = local_cols(DB)
if isa(DB, 'table'); c = DB.Properties.VariableNames; else; c = fieldnames(DB)'; end
end

function c = local_col_cell(DB, name)
v = DB.(name);
if iscell(v); c = v(:);
elseif ischar(v); c = cellstr(v);
elseif isa(v, 'string'); c = cellstr(v);
else; c = num2cell(v, 2);
end
end

function c = local_col_text(DB, name)
c = cellfun(@local_str, local_col_cell(DB, name), 'UniformOutput', false);
end

function f = local_parse_frames(v)
% '33:35' (char / string / cell) or a numeric vector -> numeric vector
if iscell(v); v = v{1}; end
if isnumeric(v); f = v(:)'; return; end
v = strtrim(char(v));
if isempty(v); f = []; return; end
nums = str2double(regexp(v, '\d+', 'match'));
if numel(nums) == 2 && ~isempty(regexp(v, '\d\s*[:\-]\s*\d', 'once'))
    f = nums(1):nums(2);          % '33:35' or '33-35' -> range
else
    f = nums;                     % '[33 34 35]', '33,34,35', '34' -> list
end
end

function s = local_str(v)
if iscell(v); v = v{1}; end
if isnumeric(v) || islogical(v); s = mat2str(v); else; s = strtrim(char(v)); end
end

function [top_mask, bot_mask, valid_idx] = local_tails(w, quantile_frac)
valid_idx = find(w ~= 0 & ~isnan(w));
top_mask = false(10000, 1);
bot_mask = false(10000, 1);
if numel(valid_idx) < 2; return; end
thr_top = quantile(w(valid_idx), 1 - quantile_frac);
thr_bot = quantile(w(valid_idx), quantile_frac);
top_mask(valid_idx(w(valid_idx) >= thr_top)) = true;
bot_mask(valid_idx(w(valid_idx) <= thr_bot)) = true;
end
