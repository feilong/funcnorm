addpath('../../matlab/')
n_timepoints = 10;

% parseSurfaceFile would calculate neighbors infinitely for this file
% [n_nodes, coords] = parseSurfaceFile('../data/lh.sphere.reg.asc');
load('../data/lh.sphere.reg.mat');
cart_coords = coords.cart_coords;
n_nodes = size(cart_coords, 2);
neighbors = coords.neighbors;
rng(0);
T = random('norm', 0, 1, n_timepoints, n_nodes);
% If all gds are too large, f_interp would throw an error
factor = 0.01;
spher_coords = computeSphericalFromCartesian(cart_coords, 1);
warped_spher_coords = spher_coords + random('norm', 0, 1, 2, n_nodes);
warped_cart_coords = computeCartesianFromSpherical(warped_spher_coords, 1);
cart_coords = computeCartesianFromSpherical(spher_coords, 1) * factor;  % transformed onto a sphere
warp = warped_cart_coords - cart_coords * factor;
nn = 0;

TW = computeInterpOnSphere(T, cart_coords, neighbors, warp, nn);
fname = 'data_for_test_compute_interp_on_sphere_0.mat';
save(fname, 'T', 'cart_coords', 'neighbors', 'warp', 'nn', 'TW');

nn = 1;
TW = computeInterpOnSphere(T, cart_coords, neighbors, warp, nn);
fname = 'data_for_test_compute_interp_on_sphere_1.mat';
save(fname, 'T', 'cart_coords', 'neighbors', 'warp', 'nn', 'TW');
