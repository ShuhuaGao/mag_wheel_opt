clear;

% Dynamic sampling for wheel designs and incremental saving.

% Parameters
n_samples          = 6000;
d1                 = 1.0;    %#ok<NASGU>  % kept for compatibility if needed
range_w            = [10, 35];
range_n1           = [0.333, 3];
range_R            = [20, 30];
range_d_min        = 3;
range_d_max_global = 20;

% File names
csv_filename = 'lhs_dynamic_incremental.csv';
mat_filename = 'lhs_dynamic_incremental.mat';

% Prepare CSV file (write header if first run)
if ~isfile(csv_filename)
    header = {'d', 'w', 'n1', 'R', 'Fy_N', 'G_N', 'Fy_over_G'};
    writecell(header, csv_filename);
    existing_results = [];
else
    existing_results = readmatrix(csv_filename);
end

start_index = size(existing_results, 1);
samples = [];
results = [];

fprintf('Sampling %d new samples; each evaluation is saved immediately...\n', n_samples);
tic;

attempts     = 0;
max_attempts = n_samples * 20;

while size(samples, 1) < n_samples && attempts < max_attempts
    attempts = attempts + 1;

    % Sample R first, then derive an upper bound for d
    R = range_R(1) + rand() * (range_R(2) - range_R(1));

    d_max = min(range_d_max_global, R - 10);
    if d_max < range_d_min
        continue;
    end

    % Sample design variables
    d  = range_d_min + rand() * (d_max - range_d_min);
    w  = range_w(1)   + rand() * (range_w(2) - range_w(1));
    n1 = range_n1(1)  + rand() * (range_n1(2) - range_n1(1));

    x = [d, w, n1, R];

    % Run COMSOL evaluation
    [ratio, Fy, G] = evaluate_individual(x(1), x(2), x(3), x(4));

    % Append one result to CSV
    one_result = [d, w, n1, R, Fy, G, ratio];
    writematrix(one_result, csv_filename, 'WriteMode', 'append');

    % Store in memory (for MAT file)
    samples(end+1, :) = x;
    results(end+1, :) = one_result;

    % Display progress
    idx = start_index + size(samples, 1);
    fprintf('Sample %4d: d=%.2f, w=%.2f, n1=%.3f, R=%.2f -> Fy=%.2f N, G=%.2f N, Ratio=%.4f\n', ...
        idx, d, w, n1, R, Fy, G, ratio);

    if mod(size(samples, 1), 100) == 0
        fprintf('Generated %d samples, elapsed time: %.2f s\n', size(samples, 1), toc);
    end
end

% Append to MAT file
if isfile(mat_filename)
    load(mat_filename, 'samples_all', 'results_all');
    samples_all = [samples_all; samples];
    results_all = [results_all; results];
else
    samples_all = samples;
    results_all = results;
end
save(mat_filename, 'samples_all', 'results_all');

elapsed = toc;
fprintf('All new samples appended to MAT file: %s\n', mat_filename);
fprintf('Total runtime: %.2f s\n', elapsed);
