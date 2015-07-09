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
