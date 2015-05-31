addpath('../../matlab/')
coords = {'cartesian', 'spherical'};
submodes = {'single', 'multi'};
for coord_n = 1:2
    for submode_n = 1:2
        for i = 0:19
            coord = coords{coord_n};
            submode = submodes{submode_n};
            fname = sprintf('test_compute_geodesic_distances_data/%s-%s-%03d.mat', coord, submode, i);
            data = load(fname);
            res = computeGeodesicDistances(data.a, data.b);
            output = sprintf('test_compute_geodesic_distances_data/%s-%s-%03d-out.mat', coord, submode, i);
            save(output, 'res');
        end
    end
end