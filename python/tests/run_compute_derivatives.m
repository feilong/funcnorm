addpath('../../matlab/')
n_nodes = 5000;
rng(0);
spher_coords_1 = rand(2, n_nodes);
spher_coords_1(1, :) = spher_coords_1(1, :) * pi;
spher_coords_1(2, :) = spher_coords_1(2, :) * 2 * pi;
spher_coords_2 = rand(2, n_nodes);
spher_coords_2(1, :) = spher_coords_2(1, :) * pi;
spher_coords_2(2, :) = spher_coords_2(2, :) * 2 * pi;
gds = computeGeodesicDistances(spher_coords_1, spher_coords_2);
resolution = 2;

dg_dphi = dgds_dphi(spher_coords_1, spher_coords_2, resolution, gds);
dg_dtheta = dgds_dtheta(spher_coords_1, spher_coords_2, resolution, gds);
df_dphi_vals = df_dphi(spher_coords_1, spher_coords_2, resolution, gds);
df_dtheta_vals = df_dtheta(spher_coords_1, spher_coords_2, resolution, gds);

fname = 'data_for_test_derivatives.mat';
save(fname, 'spher_coords_1', 'spher_coords_2', 'gds', 'resolution', ...
     'dg_dphi', 'dg_dtheta', 'df_dphi_vals', 'df_dtheta_vals');
