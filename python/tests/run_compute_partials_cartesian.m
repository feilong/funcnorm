addpath('../../matlab/')
n_nodes = 5000;
rng(0);
cart_coords = rand(3, n_nodes);
coord_maps = randi([1, 3], 1, n_nodes);
dp_dphi = computePartialCartesian_dphi(cart_coords, coord_maps);
dp_dtheta = computePartialCartesian_dtheta(cart_coords, coord_maps);
fname = 'data_for_test_compute_partials_cartesian.mat';
save(fname, 'cart_coords', 'coord_maps', 'dp_dphi', 'dp_dtheta');
