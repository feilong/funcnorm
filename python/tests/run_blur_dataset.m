addpath('../../matlab/');
addpath('.');
rng(0);

n_nodes = 1000;
n_timepoints = 300;

cart_coords = rand(3, n_nodes);
coord_maps = randi([1, 3], 1, n_nodes);
spher_coords = computeSphericalFromCartesian(cart_coords, coord_maps);
cart_coords = computeCartesianFromSpherical(spher_coords, coord_maps) * 0.1;

dataset = rand(n_timepoints, n_nodes);
[U, S, V] = svd(dataset, 'econ');
s = diag(S);

total_nbrs = randi([1, 6], 1, n_nodes);
nbrs = -99 * ones(6, n_nodes);
for i = 1:n_nodes
    n_nbrs = total_nbrs(i);
    nbrs(1:n_nbrs, i) = datasample(0:(n_nodes-1), n_nbrs);
end

resolution = 2;

[V2, s2] = blurDataset(V, s, cart_coords, nbrs, total_nbrs, resolution);
[V3, s3, U3] = blurDataset(V, s, U, cart_coords, nbrs, total_nbrs, resolution);
Q = blurDatasetNoSVD(V, s, U, cart_coords, nbrs, total_nbrs, resolution);

fname = 'data_for_test_blur_dataset.mat';
% save(fname, 'U', 's', 'V', 'cart_coords', 'nbrs', 'total_nbrs', ...
%      'resolution', 'V2', 's2', 'V3', 's3', 'U3');
save(fname, 'U', 's', 'V', 'cart_coords', 'nbrs', 'total_nbrs', ...
     'resolution', 'Q', 'V2', 's2', 'V3', 's3', 'U3');
