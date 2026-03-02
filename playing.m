% -----------------------------
% Sliding Window Permutation Plot (Frames only)
% Uses:
%   null_mean_acc_acrossT
%   null_std_acc_acrossT
%   real_acc_acrossT
%   real_acc_acrossT_frame
%   signific_acrossT
% -----------------------------

figure('Color','w'); hold on;

N = numel(real_acc_acrossT);
frames = 17:122;   % simple frame index

% Force row vectors
null_mean  = double(null_mean_acc_acrossT(:))';
null_std   = double(null_std_acc_acrossT(:))';
real_trial = double(real_acc_acrossT(:))';
real_frame = double(real_acc_acrossT_frame(:))';
sig01      = logical(signific_acrossT(:))';

% ---- Null band ----
x = frames;
y1 = null_mean - null_std;
y2 = null_mean + null_std;

fill([x fliplr(x)], [y1 fliplr(y2)], ...
     [0.7 0.7 0.7], 'EdgeColor','none', 'FaceAlpha',0.3);

plot(frames, null_mean, 'k', 'LineWidth',2);

% ---- Real curves ----
plot(frames, real_trial, 'b', 'LineWidth',2.5);
plot(frames, real_frame, 'b--', 'LineWidth',2);

% ---- Chance line ----
plot([frames(1) frames(end)], [0.5 0.5], 'k--');

ylim([0.4 1.0]);

% ---- Significance bar ----
if any(sig01)
    yl = ylim;
    yr = yl(2) - yl(1);

    bar_bottom = yl(1) + 0.02*yr;
    bar_top    = bar_bottom + 0.02*yr;

    for k = find(sig01)
        fill([k-0.5 k+0.5 k+0.5 k-0.5], ...
             [bar_bottom bar_bottom bar_top bar_top], ...
             'k', 'EdgeColor','none');
    end
end

xlabel('Frame');
ylabel('Accuracy');
title('Sliding-window permutation');
legend({'Null ± Std','Null mean','Real (trial)','Real (frame)','Chance'}, ...
        'Location','best');

box on;
hold off;