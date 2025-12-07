% Filter rows in a CSV file based on Fy_N and save to a new file.

% Input and output file names
file_in  = 'lhs_dynamic_incremental.csv';
file_out = 'lhs_dynamic_incrementalfiltered.csv';

% Read table (with header)
T = readtable(file_in);

% Filter rows with Fy_N > 1e-5
T_filtered = T(T.Fy_N > 1e-5, :);

% Save filtered table
writetable(T_filtered, file_out);

fprintf('Filtered CSV saved to %s: kept %d samples (original %d).\n', ...
    file_out, height(T_filtered), height(T));
