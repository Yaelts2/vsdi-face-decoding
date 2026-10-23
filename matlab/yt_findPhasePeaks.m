function P = yt_findPhasePeaks(db_path, model_dir, varargin)
%YT_FINDPHASEPEAKS  Go model by model through the database: find the phase-1 / phase-2 accuracy
%   peaks automatically, plot them, ask you to confirm or choose them yourself, and save each
%   confirmed model into the database right away.
%
%   P = yt_findPhasePeaks(db_path, model_dir)
%   P = yt_findPhasePeaks(db_path, model_dir, 'StartRow', 9)      % continue from row 9
%
%   db_path   : models_database .mat (the variable whose name starts with 'DB' is the table)
%   model_dir : folder of the model .mat files; the model is fullfile(model_dir, [model_id '.mat']),
%               falling back to the stored mat_path with a warning
%
%   For each model a figure shows the accuracy + the proposed peaks, and a dialog window asks:
%       Good - save   -> this model is saved to the database right away
%       Change peaks  -> a window to type the two peaks (center frame, e.g. 35, or 34:36);
%                        the figure is redrawn with your peaks and you are asked again
%       Skip / Quit   -> skip this model (row unchanged) or stop (everything saved so far is kept)
%
%   Automatic proposal (trial-level accuracy, windows with center time in [0, MaxMs] ms,
%   time = (centers - zero_frame) * frame_duration_ms, as in plotModelFigures):
%     1. PHASE 1 = the first local maximum (>= both neighbours) with accuracy >= MinPeakAcc
%     2. TROUGH  = the first local minimum after phase 1, at least MinDip below phase 1
%     3. PHASE 2 = the first local maximum after the trough, at least MinDip above the trough
%     a flat peak (e.g. 1.0 for several windows) -> its middle window
%   Saved per phase: the peak center +/- 1 (e.g. '34:36'), its time (ms) and accuracy, in
%   peak1_frames / peak1_time_ms / peak1_acc and peak2_frames / peak2_time_ms / peak2_acc.
%
%   Options (name/value):
%     'MaxMs'      250    end of the search window (ms after stimulus onset)
%     'MinPeakAcc' 0.70   phase 1 must reach this accuracy (skips noise bumps near onset)
%     'MinDip'     0.02   minimum dip / rise between a peak and the trough (0 = no tolerance)
%     'StartRow'   1      first database row to go through
%
%   The database file is backed up once (dated copy next to it) before the first save; every
%   save writes to tempdir and then moves onto the file. Other variables in the file are kept.

opt = struct('MaxMs', 250, 'MinPeakAcc', 0.70, 'MinDip', 0.02, 'StartRow', 1);
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
n_rows = numel(DB.(cols{1}));
ids = cellfun(@as_text, as_cell(DB.model_id), 'UniformOutput', false);
paths = {};
if any(strcmp(cols, 'mat_path')); paths = cellfun(@as_text, as_cell(DB.mat_path), 'UniformOutput', false); end
old1 = text_col(DB, cols, 'peak1_frames', n_rows);
old2 = text_col(DB, cols, 'peak2_frames', n_rows);

% the 6 peak columns, kept in memory and written as whole columns
f1c = old1;  f2c = old2;
t1c = num_col(DB, cols, 'peak1_time_ms', n_rows);  t2c = num_col(DB, cols, 'peak2_time_ms', n_rows);
a1c = num_col(DB, cols, 'peak1_acc', n_rows);      a2c = num_col(DB, cols, 'peak2_acc', n_rows);

fprintf('database %s: %d models (variable %s)\n', db_path, n_rows, db_var);
backed_up = false;
P = struct('row', {}, 'tag', {}, 'frames1', {}, 'frames2', {}, 'status', {});
fig = figure('Name', 'Phase peaks', 'Color', 'w');

for r = opt.StartRow:n_rows
    tag = regexp(ids{r}, '\d{6}[a-z]\d{2}', 'match', 'once');
    if isempty(tag); tag = ids{r}; end
    fprintf('\n===== row %d / %d : %s =====\n', r, n_rows, ids{r});

    % --- model file ---
    mp = fullfile(model_dir, [ids{r} '.mat']);
    if ~exist(mp, 'file') && ~isempty(paths) && exist(paths{r}, 'file')
        warning('row %d: %s not in model_dir, using stored mat_path', r, ids{r});
        mp = paths{r};
    end
    if ~exist(mp, 'file')
        fprintf('model .mat not found -> skipped (database row unchanged)\n');
        P(end+1) = struct('row', r, 'tag', tag, 'frames1', old1{r}, 'frames2', old2{r}, 'status', 'no model file'); %#ok<AGROW>
        continue
    end
    M = load(mp, 'centers', 'trial_acc_mean', 'zero_frame', 'frame_duration_ms');
    c   = M.centers(:)';
    acc = M.trial_acc_mean(:)';
    t   = (c - M.zero_frame) * M.frame_duration_ms;

    % --- automatic proposal ---
    [i1, it, i2, note] = detect(acc, t, opt);
    source = 'auto';
    if ~isempty(note); fprintf('automatic: %s\n', note); end

    % --- confirm / choose loop ---
    status = '';
    while isempty(status)
        draw(fig, r, tag, t, c, acc, i1, it, i2, old1{r}, old2{r}, source, note, opt);
        if isempty(i1) || isempty(i2)
            fprintf('no complete proposal - choose the peaks in the dialog\n');
        else
            fprintf('proposal: phase 1 = %d:%d (%g ms, acc %.3f) | phase 2 = %d:%d (%g ms, acc %.3f) | database now: %s / %s\n', ...
                c(i1)-1, c(i1)+1, t(i1), acc(i1), c(i2)-1, c(i2)+1, t(i2), acc(i2), old1{r}, old2{r});
        end
        % ---- dialog window ----
        if isempty(i1) || isempty(i2)
            msg = sprintf('Row %d  -  %s\n\nNo automatic proposal (%s).\nChoose the peaks yourself.', r, tag, note);
        else
            msg = sprintf(['Row %d  -  %s\n\nPhase 1:  %d:%d   (%g ms, acc %.3f)\nPhase 2:  %d:%d   (%g ms, acc %.3f)\n\n' ...
                           'Database now:  %s  /  %s\n\nAre these peaks good?'], r, tag, ...
                           c(i1)-1, c(i1)+1, t(i1), acc(i1), c(i2)-1, c(i2)+1, t(i2), acc(i2), old1{r}, old2{r});
        end
        btn = questdlg(msg, sprintf('Phase peaks - row %d of %d', r, n_rows), ...
                       'Good - save', 'Change peaks', 'Skip / Quit', 'Good - save');
        switch btn
            case 'Good - save'
                if isempty(i1) || isempty(i2)
                    uiwait(warndlg('There are no two peaks yet - use "Change peaks".', 'Phase peaks', 'modal'));
                else
                    status = 'saved';
                end
            case 'Change peaks'
                if isempty(i1); d1 = ''; else; d1 = sprintf('%d:%d', c(i1)-1, c(i1)+1); end
                if isempty(i2); d2 = ''; else; d2 = sprintf('%d:%d', c(i2)-1, c(i2)+1); end
                a = inputdlg({sprintf('Phase 1 - center frame or 3 frames (e.g. 35 or 34:36)\n[time (ms) = (center - %g) x %g]', ...
                                      M.zero_frame, M.frame_duration_ms), ...
                              'Phase 2 - center frame or 3 frames (e.g. 47 or 46:48)'}, ...
                             sprintf('Change peaks - row %d  %s', r, tag), [1 60], {d1, d2});
                if ~isempty(a)
                    j1 = center_index(c, str2num(a{1})); %#ok<ST2NM>
                    j2 = center_index(c, str2num(a{2})); %#ok<ST2NM>
                    if isempty(j1) || isempty(j2)
                        uiwait(warndlg('Not a window center of this model - try again.', 'Phase peaks', 'modal'));
                    elseif c(j2) <= c(j1)
                        uiwait(warndlg('Phase 2 must come after phase 1 - try again.', 'Phase peaks', 'modal'));
                    else
                        i1 = j1;  i2 = j2;  it = [];  source = 'manual';  note = '';
                    end
                end
            case 'Skip / Quit'
                b2 = questdlg(sprintf('Row %d  -  %s', r, tag), 'Skip or quit', ...
                              'Skip this model', 'Quit (saved so far is kept)', 'Back', 'Back');
                if strcmp(b2, 'Skip this model'); status = 'skipped'; end
                if strcmp(b2, 'Quit (saved so far is kept)'); status = 'quit'; end
            otherwise   % dialog closed with X -> ask again
        end
        if ~isempty(i1) && ~isempty(i2) && c(i2) <= c(i1) && strcmp(status, 'saved')
            fprintf('phase 2 must come after phase 1 - choose again\n');
            status = '';
        end
    end

    if strcmp(status, 'quit')
        fprintf('\nstopped at row %d. To continue later: yt_findPhasePeaks(db_path, model_dir, ''StartRow'', %d)\n', r, r);
        break
    end
    if strcmp(status, 'skipped')
        fprintf('row %d skipped (database unchanged)\n', r);
        P(end+1) = struct('row', r, 'tag', tag, 'frames1', old1{r}, 'frames2', old2{r}, 'status', 'skipped'); %#ok<AGROW>
        continue
    end

    % --- save this row now ---
    f1c{r} = sprintf('%d:%d', c(i1) - 1, c(i1) + 1);   f2c{r} = sprintf('%d:%d', c(i2) - 1, c(i2) + 1);
    t1c(r) = t(i1);  t2c(r) = t(i2);  a1c(r) = acc(i1);  a2c(r) = acc(i2);
    DB = setcol(DB, 'peak1_frames', f1c);  DB = setcol(DB, 'peak2_frames', f2c);
    DB = setcol(DB, 'peak1_time_ms', t1c); DB = setcol(DB, 'peak2_time_ms', t2c);
    DB = setcol(DB, 'peak1_acc', a1c);     DB = setcol(DB, 'peak2_acc', a2c);
    S.(db_var) = DB;
    if ~backed_up
        [d, nm] = fileparts(db_path);
        backup = fullfile(d, sprintf('%s__backup_%s.mat', nm, datestr(now, 'yyyy-mm-dd_HH-MM-SS')));
        copyfile(db_path, backup);
        fprintf('database backup: %s\n', backup);
        backed_up = true;
    end
    [~, nm] = fileparts(db_path);
    tmp = fullfile(tempdir, [nm '.mat']);
    save(tmp, '-struct', 'S');
    movefile(tmp, db_path, 'f');
    fprintf('SAVED row %d: phase 1 = %s (%g ms) | phase 2 = %s (%g ms)  [%s]\n', r, f1c{r}, t1c(r), f2c{r}, t2c(r), source);
    P(end+1) = struct('row', r, 'tag', tag, 'frames1', f1c{r}, 'frames2', f2c{r}, 'status', ['saved (' source ')']); %#ok<AGROW>
end

% ---------------- summary ----------------
fprintf('\n=== summary ===\n');
for q = P
    fprintf('row %-3d %-11s %-8s %-8s %s\n', q.row, q.tag, q.frames1, q.frames2, q.status);
end
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


function draw(fig, r, tag, t, c, acc, i1, it, i2, old1, old2, source, note, opt)
figure(fig); clf; hold on;
yl = [0.4 1.05];
patch([0 opt.MaxMs opt.MaxMs 0], [yl(1) yl(1) yl(2) yl(2)], [0.92 0.92 1], 'EdgeColor', 'none');
plot([t(1) t(end)], [0.5 0.5], ':', 'Color', [0.5 0.5 0.5]);
plot([0 0], yl, '--k');
% what the database has now (grey squares)
for o = {old1, old2}
    oo = frames_num(o{1});
    if ~isempty(oo)
        k = ismember(c, oo);
        plot(t(k), acc(k), 's', 'Color', [0.55 0.55 0.55], 'MarkerSize', 9, 'LineWidth', 1.2);
    end
end
plot(t, acc, 'k.-', 'LineWidth', 1.2, 'MarkerSize', 10);
% proposal: 3 frames (red line) + peak (red dot)
for i = [i1 i2]
    k = ismember(c, c(i)-1:c(i)+1);
    plot(t(k), acc(k), 'r-', 'LineWidth', 4);
    plot(t(i), acc(i), 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'MarkerSize', 9);
    text(t(i), acc(i) + 0.03, sprintf('%d:%d\n%g ms', c(i)-1, c(i)+1, t(i)), ...
         'HorizontalAlignment', 'center', 'Color', 'r', 'FontSize', 9);
end
if ~isempty(it)
    plot(t(it), acc(it), 'v', 'MarkerFaceColor', [0.2 0.4 1], 'MarkerEdgeColor', [0.2 0.4 1], 'MarkerSize', 9);
end
set(gca, 'XLim', [-100 opt.MaxMs + 200], 'YLim', yl);
xlabel('Time from stimulus onset (ms)'); ylabel('Trial accuracy');
if isempty(note); nt = ''; else; nt = ['  [' note ']']; end
title(sprintf('row %d  %s  (%s)%s\nred = proposed peaks (3 frames) | blue = trough | grey squares = database now', ...
      r, tag, source, nt), 'Interpreter', 'none');
box off; hold off; drawnow;
end


% =====================================================================================
function i = center_index(c, v)
i = [];
if isempty(v); return; end
v = v(:)';
if numel(v) > 1; v = v(ceil(numel(v) / 2)); end     % 34:36 -> 35
i = find(c == v, 1);
end

function f = frames_num(v)
if isempty(v); f = []; return; end
if isnumeric(v); f = v(:)'; return; end
f = str2num(char(v)); %#ok<ST2NM>
f = f(:)';
end

function c = colnames(DB)
if isa(DB, 'table'); c = DB.Properties.VariableNames; else; c = fieldnames(DB)'; end
end

function v = text_col(DB, cols, name, n)
if any(strcmp(cols, name))
    v = cellfun(@as_text, as_cell(DB.(name)), 'UniformOutput', false);
    v = v(:);
else
    v = repmat({''}, n, 1);
end
end

function v = num_col(DB, cols, name, n)
if ~any(strcmp(cols, name)); v = nan(n, 1); return; end
col = DB.(name);
if iscell(col)
    v = cellfun(@(x) str2double(as_text(x)), col(:));
else
    v = double(col(:));
end
end

function DB = setcol(DB, name, v)
% write a whole column; a text column existing as cells stays cells, numbers stay numbers
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
