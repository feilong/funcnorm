addpath('../../matlab/')
n_nodes = 5000;
rng(0);
cart_coords = rand(3, n_nodes);
coord_maps = randi([1, 3], 1, n_nodes);
spher_coords = computeSphericalFromCartesian(cart_coords, coord_maps);
cart_coords = computeCartesianFromSpherical(spher_coords, coord_maps);
orig_nbrs = zeros(6, n_nodes);
for i = 1:n_nodes
    orig_nbrs(:, i) = datasample(0:(n_nodes-1), 6);
end
max_nbrs = size(orig_nbrs, 1);
orig_metric_distances = random('norm', 0, 1, max_nbrs, n_nodes);
res = 2;
[md_diff, dmd_diffs_dphi, dmd_diffs_dtheta] = ...
    computeMetricTerms(orig_nbrs, [], cart_coords, coord_maps, ...
                       orig_metric_distances, res);
fname = 'data_for_test_compute_metric_terms.mat';
save(fname, 'cart_coords', 'orig_nbrs', 'coord_maps', ...
     'orig_metric_distances', 'res', 'md_diff', 'dmd_diffs_dphi', ...
     'dmd_diffs_dtheta');
