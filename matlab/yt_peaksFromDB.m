function [f1, f2, row] = yt_peaksFromDB(DB, matname)
%YT_PEAKSFROMDB  Peak windows (CENTER frames) of a model, read from the models database table.
%   [f1, f2, row] = yt_peaksFromDB(DB, matname)
%
%   DB      : the database table (columns date, session, conditions, peak1_frames, peak2_frames)
%   matname : model .mat file name, e.g. 'sliding_window__110209a15_frame1-100__SVM_...mat'
%
%   The row is found by DATASET (date + session + conditions, e.g. 110209 / a / '1,5'), not by the
%   full model name, so a .mat from another training run of the same dataset still matches.
%   Peaks are stored as text ('35:37' or '[34 35 36]') or numbers; both work.
%   Returns f1 = f2 = [] and row = 0 if the dataset is not in the table or its peaks are empty.

f1 = []; f2 = []; row = 0;

tok = regexp(matname, '(\d{6})([a-z])(\d{2})', 'tokens', 'once');      % 110209 a 15
if isempty(tok)
    warning('cannot read date/session/conditions from %s', matname);
    return
end

if isa(DB, 'table')
    cols = DB.Properties.VariableNames;
else
    cols = fieldnames(DB);
end
c_p1 = cols{find(strncmp(cols, 'peak1_frames', 12), 1)};
c_p2 = cols{find(strncmp(cols, 'peak2_frames', 12), 1)};

dates = cellfun(@as_text, as_cell(DB.date), 'UniformOutput', false);
sess  = cellfun(@as_text, as_cell(DB.session), 'UniformOutput', false);
conds = cellfun(@(c) strrep(strrep(as_text(c), ',', ''), ' ', ''), as_cell(DB.conditions), 'UniformOutput', false);

hit = find(strcmp(dates, tok{1}) & strcmp(sess, tok{2}) & strcmp(conds, tok{3}));
if isempty(hit)
    return
end
if numel(hit) > 1
    warning('%s: %d database rows match %s%s%s - using the last one (row %d)', ...
            matname, numel(hit), tok{1}, tok{2}, tok{3}, hit(end));
end
row = hit(end);

p1 = as_cell(DB.(c_p1));  p2 = as_cell(DB.(c_p2));
f1 = as_frames(p1{row});
f2 = as_frames(p2{row});
end


function c = as_cell(col)
if iscell(col)
    c = col;
elseif isstring(col)
    c = cellstr(col);
elseif ischar(col)
    c = cellstr(col);
else
    c = num2cell(col, 2);
end
end

function s = as_text(v)
if isnumeric(v); s = num2str(v); else; s = strtrim(char(v)); end
end

function f = as_frames(v)
if isnumeric(v)
    f = v(:)';
else
    f = str2num(char(v)); %#ok<ST2NM>   % '35:37' -> [35 36 37], '[34 35 36]' -> [34 35 36]
    f = f(:)';
end
end
