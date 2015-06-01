addpath('../../matlab/')
for i = 0:9
    N = 20;
    rng(0);
    cart_coords = rand(3, N) .* 0.5;
    warps = {};
    for j = 1:30
        warps{j} = rand(3, N) .* 0.5;
    end
    res = computeZeroCorrection(cart_coords, warps);
    fname = sprintf('test_compute_zero_correlation_data/case_%03d.mat', i);
    save(fname, 'cart_coords', 'warps', 'res');
end