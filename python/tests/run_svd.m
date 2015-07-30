rng(0);

n_nodes = 1000;
n_timepoints = 300;

dataset = rand(n_timepoints, n_nodes);
[U, S, V] = svd(dataset, 'econ');
s = diag(S);

fname = 'data_for_test_svd.mat';
save(fname, 'U', 's', 'V', 'dataset');
