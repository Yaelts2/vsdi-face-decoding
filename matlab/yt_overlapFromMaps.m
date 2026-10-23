function ov = yt_overlapFromMaps(firstpeak, secondpeak)
%YT_OVERLAPFROMMAPS  Phase-1 / phase-2 extreme-weight overlap, EXACTLY as in plotModelFigures (section 4).
%   ov = yt_overlapFromMaps(firstpeak, secondpeak)
%
%   firstpeak, secondpeak : 10000 x 1 weight maps (mean of Weights_acrossT over the peak windows),
%                           NaN / 0 outside the ROI.
%
%   Masks per phase: signed top 10% and bottom 10% of the valid (non-NaN, non-zero) pixels,
%   extreme mask = top | bottom.
%   overlap % = |extreme1 & extreme2| / ((|extreme1| + |extreme2|) / 2) * 100
%   The 4 sign combinations use the same formula on the single-sign masks.
%
%   ov.overall      : overall overlap (%)            -> same number plotModelFigures returns
%   ov.combo        : 1x4 overlap (%) of the sign pairs, order = ov.combo_names
%   ov.n_overlap, ov.n_mask1, ov.n_mask2, ov.combo_n
%   ov.top1, ov.bot1, ov.top2, ov.bot2 : the logical 10000x1 masks

firstpeak  = firstpeak(:);
secondpeak = secondpeak(:);
n_pix = numel(firstpeak);

valid_idx1 = find(firstpeak  ~= 0 & ~isnan(firstpeak));
valid_idx2 = find(secondpeak ~= 0 & ~isnan(secondpeak));

thresh1_top = quantile(firstpeak(valid_idx1), 0.90);
thresh1_bot = quantile(firstpeak(valid_idx1), 0.10);
top_mask1 = false(n_pix,1); bot_mask1 = false(n_pix,1);
top_mask1(valid_idx1(firstpeak(valid_idx1) >= thresh1_top)) = true;
bot_mask1(valid_idx1(firstpeak(valid_idx1) <= thresh1_bot)) = true;

thresh2_top = quantile(secondpeak(valid_idx2), 0.90);
thresh2_bot = quantile(secondpeak(valid_idx2), 0.10);
top_mask2 = false(n_pix,1); bot_mask2 = false(n_pix,1);
top_mask2(valid_idx2(secondpeak(valid_idx2) >= thresh2_top)) = true;
bot_mask2(valid_idx2(secondpeak(valid_idx2) <= thresh2_bot)) = true;

extreme_mask1 = top_mask1 | bot_mask1;
extreme_mask2 = top_mask2 | bot_mask2;
overlap_mask  = extreme_mask1 & extreme_mask2;

ov.n_overlap = sum(overlap_mask);
ov.n_mask1   = sum(extreme_mask1);
ov.n_mask2   = sum(extreme_mask2);
ov.overall   = ov.n_overlap / ((ov.n_mask1 + ov.n_mask2) / 2) * 100;

% 4 sign combinations (same order as the 2x2 figure in plotModelFigures)
pairs = {top_mask1, top_mask2; top_mask1, bot_mask2; bot_mask1, top_mask2; bot_mask1, bot_mask2};
ov.combo_names = {'P1 top & P2 top', 'P1 top & P2 bottom', 'P1 bottom & P2 top', 'P1 bottom & P2 bottom'};
ov.combo   = zeros(1,4);
ov.combo_n = zeros(1,4);
for i = 1:4
    a = pairs{i,1}; b = pairs{i,2};
    ov.combo_n(i) = sum(a & b);
    ov.combo(i)   = ov.combo_n(i) / ((sum(a) + sum(b)) / 2) * 100;
end

ov.top1 = top_mask1; ov.bot1 = bot_mask1;
ov.top2 = top_mask2; ov.bot2 = bot_mask2;
end
