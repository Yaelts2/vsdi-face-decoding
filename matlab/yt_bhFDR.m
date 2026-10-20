function q = yt_bhFDR(p)
% Benjamini-Hochberg adjusted p-values (NaN kept as NaN)
q = nan(size(p));
ok = ~isnan(p);
pv = p(ok);
n = numel(pv);
if n == 0; return; end
[ps, order] = sort(pv);
qs = ps .* n ./ (1:n);
qs = min(1, fliplr(cummin(fliplr(qs))));
qq = zeros(1, n);
qq(order) = qs;
q(ok) = qq;
end
