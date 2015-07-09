addpath('../../matlab/');
% The random data generated here would be far from real data.
% However, they should be good enough to test if the Python version works in the
% same way as the Matlab version.
n_triangles = 5000;
n_nodes = 5000;
rng(0);
cart_coords = rand(3, n_nodes);
coord_maps = randi([1, 3], 1, n_nodes);
triangles = zeros(3, n_triangles);
for i = 1:n_triangles
    triangles(:, i) = datasample(0:(n_nodes-1), 3);
end
orig_tri_areas = random('norm', 0, 1, 1, n_triangles);
tri_normals = random('norm', 0, 1, 3, n_triangles);

[tri_area, dareal_dphi, dareal_dtheta] = computeArealTerms(...
     triangles, cart_coords, coord_maps, orig_tri_areas, tri_normals);

fname = 'data_for_test_compute_areal_terms.mat';
save(fname, 'triangles', 'cart_coords', 'coord_maps', 'orig_tri_areas', ...
     'tri_normals', 'tri_area', 'dareal_dphi', 'dareal_dtheta');
