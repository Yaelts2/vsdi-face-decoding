function P = yt_findPhasePeaks(db_path, model_dir, varargin)
%YT_FINDPHASEPEAKS  Find the phase-1 / phase-2 accuracy peaks of every model in the database,
%   plot them for checking, and (only when asked) write them to the database.
%
%   STEP 1 - look (nothing is written):
%       P = yt_findPhasePeaks(db_path, model_dir);
%   STEP 2 - after checking the figures, fix what is wrong and write:
%       P = yt_findPhasePeaks(db_path, model_dir, 'Manual', {3, 34, 47; 9, 35, 46}, 'Save', true);
%
%   db_path   : models_database .mat (the variable whose name starts with 'DB' is the table)
%   model_dir : folder of the model .mat files; the model is fullfile(model_dir, [model_id '.mat']),
%               falling back to the stored mat_path with a warning
%
%   Algorithm (trial-level accuracy, only windows with center time in [0, MaxMs] ms,
%   time = (centers - zero_frame) * frame_duration_ms, as in plotModelFigures):
%     1. PHASE 1 = the FIRST local maximum (>= both neighbours) with accuracy >= MinPeakAcc
%     2. TROUGH  = the first local minimum after phase 1 that is at least MinDip below phase 1
%     3. PHASE 2 = the first local maximum after the trough that is at least MinDip above the trough
%   Each phase = its peak center frame +/- 1  (3 frames, e.g. '34:36'), as in the database.
%
%   Options (name/value):
%     'MaxMs'      250    end of the search window (ms after stimulus onset)
%     'MinPeakAcc' 0.70   phase 1 must reach this accuracy (skips noise bumps near onset)
%     'MinDip'     0.02   minimum dip / rise between peak and trough (skips 1-window wiggles);
%                         0 = the plain rule with no tolerance
%     'Manual'     {}     rows to set by hand: {db_row, phase1_center, phase2_center; ...}
%                         (a center may also be given as the 3 frames, e.g. 33:35)
%     'Save'       false  true -> write peak1_frames / peak1_time_ms / peak1_acc and the same for
%                         peak2 into the database (backup first, tempdir -> movefile)
%     'PerFig'     16     models per figure
%
%   P : struct array, one per database row: model, p1, p2, trough (center frames), frames1, frames2,
%       t1_ms, t2_ms, acc1, acc2, old1, old2 (what the database had), changed, source, note

opt = struct('MaxMs', 250, 'MinPeakAcc', 0.70, 'MinDip', 0.02, 'Manual', {{}}, 'Save', false, 'PerFig', 16);
for k = 1:2:numel(varargin)
    if ~isfield(opt, varargin{k}); error('unknown option %s', varargin{k}); end
    opt.(varargin{k}) = varargin{k+1};
end

% ---------------- load the database ----------------
S = load(db_path);
fn = fieldnames(S);
kDB = find(strncmp(fn, 'DB', 2), 1);
if isempty(kDB); error('no variable starting with ''DB'' in %s', db_path); end
db_var = fn{kDB};
DB = S.(db_var);
cols = colnames(DB);
n_rows = numel(getcol(DB, cols{1}));
ids   = cellfun(@as_text, as_cell(getcol(DB, 'model_id')), 'UniformOutput', false);
paths = {};
if any(strcmp(cols, 'mat_path')); paths = cellfun(@as_text, as_cell(getcol(DB, 'mat_path')), 'UniformOutput', false); end
old1 = repmat({''}, n_rows, 1);  old2 = repmat({''}, n_rows, 1);
old_t1 = nan(n_rows, 1); old_t2 = nan(n_rows, 1); old_a1 = nan(n_rows, 1); old_a2 = nan(n_rows, 1);
if any(strcmp(cols, 'peak1_time_ms')); old_t1 = num_col(getcol(DB, 'peak1_time_ms')); end
if any(strcmp(cols, 'peak2_time_ms')); old_t2 = num_col(getcol(DB, 'peak2_time_ms')); end
if any(strcmp(cols, 'peak1_acc'));     old_a1 = num_col(getcol(DB, 'peak1_acc'));     end
if any(strcmp(cols, 'peak2_acc'));     old_a2 = num_col(getcol(DB, 'peak2_acc'));     end
if any(strcmp(cols, 'peak1_frames')); old1 = cellfun(@as_text, as_cell(getcol(DB, 'peak1_frames')), 'UniformOutput', false); end
if any(strcmp(cols, 'peak2_frames')); old2 = cellfun(@as_text, as_cell(getcol(DB, 'peak2_frames')), 'UniformOutput', false); end
fprintf('database %s: %d models (variable %s)\n', db_path, n_rows, db_var);

% manual rows
man_rows = [];
if ~isempty(opt.Manual); man_rows = cell2mat(opt.Manual(:, 1))'; end

% ---------------- every model ----------------
fields = {'row', 'model', 'tag', 'time_ms', 'acc', 'centers', 'p1', 'p2', 'trough', 'frames1', 'frames2', ...
          't1_ms', 't2_ms', 'acc1', 'acc2', 'old1', 'old2', 'changed', 'source', 'note'};
tmpl = cell2struct(cell(numel(fields), 1), fields, 1);     % every q has the same fields in the same order
P = repmat(tmpl, 1, 0);
for r = 1:n_rows
    q = tmpl;
    q.row = r;  q.model = ids{r};
    tok = regexp(ids{r}, '\d{6}[a-z]\d{2}', 'match', 'once');
    if isempty(tok); tok = sprintf('row%d', r); end
    q.tag = tok;
    q.old1 = old1{r};  q.old2 = old2{r};
    q.note = '';

    % --- model file (model_dir/model_id.mat first, then stored mat_path) ---
    mp = fullfile(model_dir, [ids{r} '.mat']);
    if ~exist(mp, 'file') && ~isempty(paths) && exist(paths{r}, 'file')
        warning('row %d: %s not in model_dir, using stored mat_path', r, ids{r});
        mp = paths{r};
    end
    if ~exist(mp, 'file')
        % no model file -> keep what the database already has for this row
        q = empty_peaks(q);
        q.frames1 = q.old1;  q.frames2 = q.old2;
        q.t1_ms = old_t1(r); q.t2_ms = old_t2(r); q.acc1 = old_a1(r); q.acc2 = old_a2(r);
        q.source = 'KEPT';
        q.note = 'model .mat not found -> database values kept';
        P(end+1) = q; %#ok<AGROW>
        fprintf('row %2d %-12s  model .mat NOT FOUND -> keeping the database values\n', r, tok);
        continue
    end
    M = load(mp, 'centers', 'trial_acc_mean', 'zero_frame', 'frame_duration_ms');
    c   = M.centers(:)';
    acc = M.trial_acc_mean(:)';
    t   = (c - M.zero_frame) * M.frame_duration_ms;
    q.time_ms = t;  q.acc = acc;  q.centers = c;

    % --- automatic detection ---
    [i1, it, i2, note] = detect(acc, t, opt);
    q.note = note;
    q.source = 'auto';

    % --- manual override ---
    mrow = find(man_rows == r, 1);
    if ~isempty(mrow)
        i1 = center_index(c, opt.Manual{mrow, 2});
        i2 = center_index(c, opt.Manual{mrow, 3});
        it = [];
        q.source = 'MANUAL';
        q.note = '';
    end

    q.p1 = pick(c, i1);  q.p2 = pick(c, i2);  q.trough = pick(c, it);
    if isempty(i1) || isempty(i2)
        q.frames1 = '';  q.frames2 = '';  q.t1_ms = NaN;  q.t2_ms = NaN;  q.acc1 = NaN;  q.acc2 = NaN;
    else
        q.frames1 = sprintf('%d:%d', q.p1 - 1, q.p1 + 1);
        q.frames2 = sprintf('%d:%d', q.p2 - 1, q.p2 + 1);
        q.t1_ms = t(i1);  q.t2_ms = t(i2);
        q.acc1 = acc(i1); q.acc2 = acc(i2);
    end
    q.changed = ~strcmp(norm_frames(q.frames1), norm_frames(q.old1)) || ...
                ~strcmp(norm_frames(q.frames2), norm_frames(q.old2));
    P(end+1) = q; %#ok<AGROW>
end

% ---------------- print ----------------
fprintf('\n%-4s %-11s %-8s %-11s %-11s %-10s %-10s %-8s %s\n', 'row', 'dataset', 'source', ...
        'phase 1', 'phase 2', 'db before', '', 'changed', 'note');
for q = P
    fprintf('%-4d %-11s %-8s %-11s %-11s %-10s %-10s %-8s %s\n', q.row, q.tag, q.source, ...
        sprintf('%s(%gms)', q.frames1, q.t1_ms), sprintf('%s(%gms)', q.frames2, q.t2_ms), ...
        q.old1, q.old2, yesno(q.changed), q.note);
end
bad = find(cellfun(@isempty, {P.frames1}) | cellfun(@isempty, {P.frames2}));
if ~isempty(bad)
    fprintf('\n%d model(s) without two peaks -> set them with ''Manual'': rows %s\n', numel(bad), mat2str([P(bad).row]));
end

% ---------------- figures ----------------
plot_all(P, opt);

% ---------------- save ----------------
if ~opt.Save
    fprintf('\nNothing written (Save = false). Check the figures, then run again with ''Save'', true\n');
    fprintf('and ''Manual'', {row, phase1_center, phase2_center; ...} for the rows you want to change.\n');
    return
end
if ~isempty(bad)
    error('not saving: rows %s have no peaks - give them in ''Manual''.', mat2str([P(bad).row]));
end

DB = setcol(DB, 'peak1_frames',  {P.frames1}');
DB = setcol(DB, 'peak1_time_ms', [P.t1_ms]');
DB = setcol(DB, 'peak1_acc',     [P.acc1]');
DB = setcol(DB, 'peak2_frames',  {P.frames2}');
DB = setcol(DB, 'peak2_time_ms', [P.t2_ms]');
DB = setcol(DB, 'peak2_acc',     [P.acc2]');
S.(db_var) = DB;

[d, nm] = fileparts(db_path);
backup = fullfile(d, sprintf('%s__backup_%s.mat', nm, datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
copyfile(db_path, backup);
tmp = fullfile(tempdir, [nm '.mat']);
save(tmp, '-struct', 'S');
movefile(tmp, db_path, 'f');
fprintf('\nDatabase updated: %s\n(backup: %s)\n', db_path, backup);
end


% =====================================================================================
function [i1, it, i2, note] = detect(acc, t, opt)
% indices into the FULL curve; neighbours outside the window are still used for the local-max test
i1 = []; it = []; i2 = []; note = '';
n = numel(acc);
win = find(t >= 0 & t <= opt.MaxMs);
if isempty(win); note = 'no windows in range'; return; end
prev = @(i) acc(max(i - 1, 1));
next = @(i) acc(min(i + 1, n));

% 1) phase 1: first local max reaching MinPeakAcc
for i = win
    if acc(i) >= prev(i) && acc(i) >= next(i) && acc(i) >= opt.MinPeakAcc
        i1 = plateau_middle(acc, i, win(end)); break
    end
end
if isempty(i1); note = sprintf('no peak >= %.2f', opt.MinPeakAcc); return; end

% 2) trough: first local min after phase 1, at least MinDip below phase 1
for i = win(win > i1)
    if acc(i) <= prev(i) && acc(i) <= next(i) && acc(i) <= acc(i1) - opt.MinDip
        it = i; break
    end
end
if isempty(it); note = 'no dip after phase 1'; return; end

% 3) phase 2: first local max after the trough, at least MinDip above the trough
for i = win(win > it)
    if acc(i) >= prev(i) && acc(i) >= next(i) && acc(i) >= acc(it) + opt.MinDip
        i2 = plateau_middle(acc, i, win(end)); break
    end
end
if isempty(i2); note = sprintf('no 2nd peak before %g ms', opt.MaxMs); end
end


function m = plateau_middle(acc, i, last)
% a peak that is a flat run (e.g. accuracy 1.0 for several windows) -> its middle window
j = i;
while j < last && acc(j + 1) == acc(i)
    j = j + 1;
end
m = floor((i + j) / 2);
end


function plot_all(P, opt)
n = numel(P);
nper = opt.PerFig;
ncol = ceil(sqrt(nper));  nrow = ceil(nper / ncol);
for f0 = 1:nper:n
    idx = f0:min(f0 + nper - 1, n);
    figure('Name', sprintf('Phase peaks: rows %d-%d', idx(1), idx(end)), 'Color', 'w');
    for s = 1:numel(idx)
        q = P(idx(s));
        subplot(nrow, ncol, s); hold on;
        if isempty(q.acc)
            title(sprintf('row %d %s: %s', q.row, q.tag, q.note), 'Interpreter', 'none'); axis off; continue
        end
        yl = [0.4 1.05];
        patch([0 opt.MaxMs opt.MaxMs 0], [yl(1) yl(1) yl(2) yl(2)], [0.92 0.92 1], 'EdgeColor', 'none');
        plot([q.time_ms(1) q.time_ms(end)], [0.5 0.5], ':', 'Color', [0.5 0.5 0.5]);
        plot([0 0], yl, '--k');
        % what the database had before (grey)
        o1 = norm_frames_num(q.old1);  o2 = norm_frames_num(q.old2);
        for o = {o1, o2}
            oo = o{1};
            if ~isempty(oo)
                k = ismember(q.centers, oo);
                plot(q.time_ms(k), q.acc(k), 's', 'Color', [0.55 0.55 0.55], 'MarkerSize', 7, 'LineWidth', 1.2);
            end
        end
        plot(q.time_ms, q.acc, 'k-', 'LineWidth', 1.2);
        % new phases: the 3 frames (red line) and the peak (red dot)
        for pk = [q.p1 q.p2]
            k = ismember(q.centers, pk-1:pk+1);
            plot(q.time_ms(k), q.acc(k), 'r-', 'LineWidth', 3);
            k = q.centers == pk;
            plot(q.time_ms(k), q.acc(k), 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'MarkerSize', 6);
        end
        if ~isempty(q.trough)
            k = q.centers == q.trough;
            plot(q.time_ms(k), q.acc(k), 'v', 'MarkerFaceColor', [0.2 0.4 1], 'MarkerEdgeColor', [0.2 0.4 1], 'MarkerSize', 6);
        end
        set(gca, 'XLim', [-100 opt.MaxMs + 150], 'YLim', yl, 'FontSize', 7);
        flag = '';
        if q.changed; flag = '  [CHANGED]'; end
        if ~isempty(q.note); flag = ['  [' q.note ']']; end
        ttl = sprintf('row %d %s (%s)%s\nP1 %s = %gms   P2 %s = %gms', q.row, q.tag, q.source, flag, ...
                      q.frames1, q.t1_ms, q.frames2, q.t2_ms);
        title(ttl, 'Interpreter', 'none', 'FontSize', 8);
        hold off;
    end
    % legend text in the last panel
    annotation('textbox', [0 0 1 0.04], 'String', ...
        ['red dot = peak, red line = the 3 frames saved | blue triangle = trough | grey squares = database before ' ...
         '| shaded = search window'], 'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontSize', 8);
end
end


% =====================================================================================
function q = empty_peaks(q)
q.time_ms = []; q.acc = []; q.centers = []; q.p1 = []; q.p2 = []; q.trough = [];
q.frames1 = ''; q.frames2 = ''; q.t1_ms = NaN; q.t2_ms = NaN; q.acc1 = NaN; q.acc2 = NaN;
q.changed = false; q.source = 'auto';
end

function v = num_col(col)
if iscell(col)
    v = cellfun(@(x) str2double(as_text(x)), col(:));
else
    v = double(col(:));
end
end

function i = center_index(c, v)
v = v(:)';
if numel(v) > 1; v = v(ceil(numel(v) / 2)); end     % 33:35 -> 34
i = find(c == v, 1);
if isempty(i); error('center %d is not a window center of this model', v); end
end

function v = pick(c, i)
if isempty(i); v = []; else; v = c(i); end
end

function s = yesno(b)
if b; s = 'YES'; else; s = ''; end
end

function s = norm_frames(v)
f = norm_frames_num(v);
if isempty(f); s = ''; else; s = mat2str(f); end
end

function f = norm_frames_num(v)
if isempty(v); f = []; return; end
if isnumeric(v); f = v(:)'; return; end
f = str2num(char(v)); %#ok<ST2NM>
f = f(:)';
end

function c = colnames(DB)
if isa(DB, 'table'); c = DB.Properties.VariableNames; else; c = fieldnames(DB)'; end
end

function v = getcol(DB, name)
v = DB.(name);
end

function DB = setcol(DB, name, v)
% write a whole column; a numeric column stays numeric, a text column becomes a cell of char
if isa(DB, 'table') && any(strcmp(DB.Properties.VariableNames, name))
    old = DB.(name);
    if iscell(v) && ~iscell(old) && isnumeric(old)
        v = cellfun(@str2num, v, 'UniformOutput', false); %#ok<ST2NM>
    end
end
DB.(name) = v;
end

function c = as_cell(col)
if iscell(col); c = col(:);
elseif ischar(col); c = cellstr(col);
elseif isa(col, 'string'); c = cellstr(col);
else; c = num2cell(col(:));
end
end

function s = as_text(v)
if isnumeric(v); s = num2str(v); else; s = strtrim(char(v)); end
end
