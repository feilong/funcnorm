addpath('../../matlab/')
coords = {'cartesian', 'spherical'};
submodes = {'single', 'multi'};
rng(0);
for coord_n = 1:2
    for submode_n = 1:2
        for i = 0:19
            coord = coords{coord_n};
            submode = submodes{submode_n};
            fname = sprintf('test_compute_geodesic_distances_data/%s-%s-%03d.mat',...
                            coord, submode, i);
            data = load(fname);
            zM = datasample(1:20, 1, 10, 'Replace', false);
            gds = computeGeodesicDistances(data.a, data.b);
            [interp_vals, non_zero_locs] = interp_f(data.a, data.b, 2, gds, 'double', zM);
            output = sprintf('test_compute_geodesic_distances_data/interp_f-%s-%s-%03d.mat',...
                             coord, submode, i);
            save(output, 'data', 'gds', 'zM', 'interp_vals', 'non_zero_locs');
        end
    end
end

rng(0);
n_nodes = 1000;

cart_coords = rand(3, n_nodes);
coord_maps = randi([1, 3], 1, n_nodes);
spher_coords = computeSphericalFromCartesian(cart_coords, coord_maps);
coords1 = computeCartesianFromSpherical(spher_coords, coord_maps) * 0.1;

%% total_nbrs = randi([1, 6], 1, n_nodes);
%% nbrs = -99 * ones(6, n_nodes);
%% for i = 1:n_nodes
%%     n_nbrs = total_nbrs(i);
%%     nbrs(1:n_nbrs, i) = datasample(0:(n_nodes-1), n_nbrs);
%% end

coords2 = coords1(:,randperm(size(coords1,2)));

resolution = 2;

gds = computeGeodesicDistances(coords1, coords2);
[interp_vals, non_zero_locs] = interp_f(coords1, coords2, resolution);

fname = 'data_for_test_interp_f.mat';
save(fname, 'coords1', 'coords2', 'interp_vals', 'non_zero_locs', 'resolution', 'gds');
