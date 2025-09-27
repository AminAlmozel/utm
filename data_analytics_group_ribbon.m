close all
clear
clc
format compact
format shortG

% Define the parent directory containing the folders
base_dir = pwd;
path = "/full_runs/mission3"; % Replace with your path
path = "/full_runs/profile4"; % Replace with your path
% Get a list of all subfolders
parentDir = fullfile(base_dir, path);
subfolders = dir(parentDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.')); % Exclude '.' and '..'

% Initialize a cell array to store data
sim_runs = {};

% Loop through each subfolder
for i = 1:length(subfolders)
    folderPath = fullfile(parentDir, subfolders(i).name);

    % Get all CSV files in the current subfolder
    csvFiles = dir(fullfile(folderPath, '*.csv'));

    % Loop through each CSV file
    for j = 1:length(csvFiles)
        filePath = fullfile(folderPath, csvFiles(j).name);

        % Read the CSV file
        data = importdata(filePath); % Use readtable for structured data

        % Store the data in the cell array
        sim_runs{end+1} = data; % Append the data
    end
end

% First, let's understand the data structure
fprintf('=== DATA STRUCTURE ANALYSIS ===\n');
fprintf('Total number of simulation runs (lambda values): %d\n', length(sim_runs));

for i = 1:length(sim_runs)
    m = sim_runs{i};
    fprintf('Run %d: %d rows x %d columns\n', i, size(m, 1), size(m, 2));
    if i == 1
        fprintf('  Sample of first few rows:\n');
        fprintf('  Row | Col2(dist) | Col3(time) | Col4(safe) | Col5(unsafe) | Col6(outside)\n');
        for j = 1:min(3, size(m, 1))
            fprintf('   %2d |    %6.2f |    %6.2f |    %6.2f |     %6.2f |     %6.2f\n', ...
                j, m(j, 2), m(j, 3), m(j, 4), m(j, 5), m(j, 6));
        end
    end
end

% Initialize arrays for means and confidence intervals
n_lambdas = length(sim_runs);
mean_distance = zeros(1, n_lambdas);
mean_safe = zeros(1, n_lambdas);
ci_distance = zeros(2, n_lambdas); % [lower; upper]
ci_safe = zeros(2, n_lambdas);

fprintf('\n=== CALCULATING METRICS FOR EACH LAMBDA ===\n');

% Calculate means and 95% confidence intervals for each lambda
for i = 1:n_lambdas
    % Process each simulation run (data table)
    m = sim_runs{i};
    
    % Extract data columns
    distance = m(:, 2);
    time = m(:, 3);
    safe = m(:, 4);
    unsafe = m(:, 5);
    outside = m(:, 6);
    
    n_samples = length(distance);
    fprintf('Lambda %d: %d data points\n', i, n_samples);
    
    % Distance metrics
    mean_distance(i) = mean(distance);
    std_distance = std(distance);
    
    fprintf('  Distance: mean=%.2f, std=%.2f\n', mean_distance(i), std_distance);
    
    % Handle case where we have only 1 sample or std = 0
    if n_samples <= 1 || std_distance == 0
        ci_distance(:, i) = [mean_distance(i); mean_distance(i)];
        fprintf('    Warning: Cannot calculate CI for distance (n=%d, std=%.2f)\n', n_samples, std_distance);
    else
        % 95% confidence interval (assuming normal distribution)
        t_critical = tinv(0.975, n_samples-1); % t-value for 95% CI
        margin_error_dist = t_critical * std_distance / sqrt(n_samples);
        ci_distance(:, i) = [mean_distance(i) - margin_error_dist; mean_distance(i) + margin_error_dist];
        fprintf('    Distance CI: [%.2f, %.2f]\n', ci_distance(1,i), ci_distance(2,i));
    end
    
    % Safe distance metrics
    mean_safe(i) = mean(safe);
    std_safe = std(safe);
    
    fprintf('  Safe: mean=%.2f, std=%.2f\n', mean_safe(i), std_safe);
    
    % Handle case where we have only 1 sample or std = 0
    if n_samples <= 1 || std_safe == 0
        ci_safe(:, i) = [mean_safe(i); mean_safe(i)];
        fprintf('    Warning: Cannot calculate CI for safe distance (n=%d, std=%.2f)\n', n_samples, std_safe);
    else
        t_critical = tinv(0.975, n_samples-1);
        margin_error_safe = t_critical * std_safe / sqrt(n_samples);
        ci_safe(:, i) = [mean_safe(i) - margin_error_safe; mean_safe(i) + margin_error_safe];
        fprintf('    Safe CI: [%.2f, %.2f]\n', ci_safe(1,i), ci_safe(2,i));
    end
end

% X-axis values (should match number of lambda values)
x = [0, 0.2, 0.4, 0.6, 0.8];
if length(x) ~= n_lambdas
    fprintf('Warning: x-axis values (%d) do not match number of lambda values (%d)\n', length(x), n_lambdas);
    x = 1:n_lambdas; % Use indices as fallback
end

%% FIRST PLOT: Distance Metrics with Confidence Ribbons
figure('Position', [100, 100, 800, 600]);

% Define colors
blue_color = [0 0.4470 0.7410];
darkerGreen = 0.8 * [0.4660 0.6740 0.1880];

% Create confidence ribbons using fill
x_fill = [x, fliplr(x)];
distance_fill = [ci_distance(1,:), fliplr(ci_distance(2,:))];
safe_fill = [ci_safe(1,:), fliplr(ci_safe(2,:))];

% Plot confidence ribbons first (behind lines)
h_ribbon1 = fill(x_fill, distance_fill, blue_color, 'FaceAlpha', 0.3, ...
                 'EdgeColor', 'none', 'DisplayName', '95% CI Total Distance');
hold on;
h_ribbon2 = fill(x_fill, safe_fill, darkerGreen, 'FaceAlpha', 0.3, ...
                 'EdgeColor', 'none', 'DisplayName', '95% CI Safe Distance');

% Plot main lines on top
h1 = plot(x, mean_distance, '-o', 'LineWidth', 3, 'MarkerSize', 8, ...
          'Color', blue_color, 'DisplayName', 'Mean Total Distance', ...
          'MarkerFaceColor', 'white', 'MarkerEdgeColor', blue_color);
h2 = plot(x, mean_safe, '-s', 'LineWidth', 3, 'MarkerSize', 8, ...
          'Color', darkerGreen, 'DisplayName', 'Mean Safe Distance', ...
          'MarkerFaceColor', 'white', 'MarkerEdgeColor', darkerGreen);

% Add horizontal dashed reference lines using first data points
yline(mean_distance(1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
yline(mean_safe(1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');

% Add vertical difference indicators and percentage annotations
y_range = max([max(ci_distance(2,:)), max(ci_safe(2,:))]) - min([min(ci_distance(1,:)), min(ci_safe(1,:))]);
text_offset = y_range * 0.05;

for i = 1:length(x)
    % Mean Distance differences
    if mean_distance(1) ~= 0
        plot([x(i) x(i)], [mean_distance(1) mean_distance(i)], ':', 'Color', blue_color, 'HandleVisibility', 'off');
        pct_change = ((mean_distance(i) - mean_distance(1)) / mean_distance(1)) * 100;
        text(x(i), ci_distance(2,i) + text_offset, sprintf('%.1f%%', pct_change), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', blue_color, ...
             'BackgroundColor', 'white', 'EdgeColor', 'none');
    end
    
    % Mean Safe differences  
    if mean_safe(1) ~= 0
        plot([x(i) x(i)], [mean_safe(1) mean_safe(i)], ':', 'Color', darkerGreen, 'HandleVisibility', 'off');
        pct_change = ((mean_safe(i) - mean_safe(1)) / mean_safe(1)) * 100;
        text(x(i), ci_safe(2,i) + text_offset, sprintf('%.1f%%', pct_change), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', darkerGreen, ...
             'BackgroundColor', 'white', 'EdgeColor', 'none');
    end
end

% Styling
xlabel('\lambda_{safe}', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Mean Distance (m)', 'FontSize', 14, 'FontWeight', 'bold');
title('Safety Performance Across \lambda_{safe} Values', 'FontSize', 16, 'FontWeight', 'bold');
legend([h1, h2, h_ribbon1, h_ribbon2], 'Location', 'northwest', 'FontSize', 12);
xlim([-0.05 0.95]);
grid on;
grid minor;
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1, 'FontSize', 12);
set(gca, 'Color', [0.98 0.98 0.98]);
hold off;

% Calculate percentage of safe distance from total distance (ratio of means approach)
safe_percentage = (mean_safe ./ mean_distance) * 100;

% Calculate confidence intervals for safe percentage using delta method
fprintf('\n=== CALCULATING SAFE PERCENTAGE CONFIDENCE INTERVALS ===\n');
ci_safe_percentage = zeros(2, n_lambdas);

for i = 1:n_lambdas
    % Get data for this lambda
    m = sim_runs{i};
    safe_data = m(:, 4);
    distance_data = m(:, 2);
    
    n_samples = length(safe_data);
    
    % Calculate means and standard errors
    mean_safe_i = mean(safe_data);
    mean_dist_i = mean(distance_data);
    se_safe = std(safe_data) / sqrt(n_samples);
    se_dist = std(distance_data) / sqrt(n_samples);
    
    % Calculate covariance between safe and distance
    cov_safe_dist = cov(safe_data, distance_data);
    if size(cov_safe_dist, 1) > 1
        cov_safe_dist = cov_safe_dist(1,2); % Extract the covariance
    else
        cov_safe_dist = 0; % If only one data point
    end
    se_cov = cov_safe_dist / n_samples;
    
    % Delta method for ratio of means: Var(X/Y) ≈ (μx/μy)² * [(σx/μx)² + (σy/μy)² - 2*(σxy)/(μx*μy)]
    if mean_dist_i ~= 0 && n_samples > 1
        ratio = mean_safe_i / mean_dist_i;
        var_ratio = (ratio^2) * ((se_safe/mean_safe_i)^2 + (se_dist/mean_dist_i)^2 - 2*se_cov/(mean_safe_i*mean_dist_i));
        
        if var_ratio >= 0
            se_ratio = sqrt(var_ratio);
            t_critical = tinv(0.975, n_samples-1);
            margin_error = t_critical * se_ratio * 100; % Convert to percentage
            
            ci_safe_percentage(:, i) = [safe_percentage(i) - margin_error; safe_percentage(i) + margin_error];
        else
            ci_safe_percentage(:, i) = [safe_percentage(i); safe_percentage(i)];
        end
    else
        ci_safe_percentage(:, i) = [safe_percentage(i); safe_percentage(i)];
    end
    
    fprintf('Lambda %d: Ratio of means = %.2f%%, CI = [%.2f%%, %.2f%%]\n', ...
        i, safe_percentage(i), ci_safe_percentage(1,i), ci_safe_percentage(2,i));
end

%% SECOND PLOT: Safe Distance Percentage with Confidence Ribbons
figure('Position', [950, 100, 800, 600]);

% Define color for percentage plot
percentage_color = [0.8500 0.3250 0.0980]; % Orange color

% Create confidence ribbon for percentage
percentage_fill = [ci_safe_percentage(1,:), fliplr(ci_safe_percentage(2,:))];

% Plot confidence ribbon first
h_ribbon_pct = fill(x_fill, percentage_fill, percentage_color, 'FaceAlpha', 0.3, ...
                    'EdgeColor', 'none', 'DisplayName', '95% CI Safe Percentage');
hold on;

% Plot main line on top
h_pct = plot(x, safe_percentage, '-o', 'LineWidth', 3, 'MarkerSize', 8, ...
             'Color', percentage_color, 'DisplayName', 'Safe Distance Percentage', ...
             'MarkerFaceColor', 'white', 'MarkerEdgeColor', percentage_color);

% Add grid
grid on;
grid minor;

% Add reference lines for key percentages
yline(50, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1, 'Alpha', 0.7, 'HandleVisibility', 'off');
yline(75, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1, 'Alpha', 0.7, 'HandleVisibility', 'off');

% Add percentage value annotations above each point (above confidence ribbon)
for i = 1:length(x)
    text(x(i), ci_safe_percentage(2,i) + 2, sprintf('%.1f%%', safe_percentage(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
         'Color', percentage_color);
end

% Styling
xlabel('\lambda_{safe}', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Safe Distance Percentage (%)', 'FontSize', 14, 'FontWeight', 'bold');
title('Safe Distance Percentage Across \lambda_{safe} Values', 'FontSize', 16, 'FontWeight', 'bold');
legend([h_pct, h_ribbon_pct], 'Location', 'northeast', 'FontSize', 12);

% Set axis limits
xlim([-0.05, 0.95]);
ylim([0, 100]);

% Enhance appearance
set(gca, 'FontSize', 12);
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
set(gca, 'Color', [0.98 0.98 0.98]);

hold off;

% Display results
lambda = x;
fprintf('\n=== FINAL RESULTS ===\n');
disp('Lambda values:');
disp(lambda);
disp('Safe percentage with 95% CI:');
for i = 1:length(lambda)
    if ci_safe_percentage(1,i) == ci_safe_percentage(2,i)
        fprintf('λ = %.1f: %.1f%% [No CI - insufficient variation]\n', lambda(i), safe_percentage(i));
    else
        fprintf('λ = %.1f: %.1f%% [%.1f%%, %.1f%%]\n', lambda(i), safe_percentage(i), ci_safe_percentage(1,i), ci_safe_percentage(2,i));
    end
end

% Additional diagnostic information
fprintf('\nDiagnostic Summary:\n');
fprintf('Number of lambda values: %d\n', n_lambdas);
for i = 1:n_lambdas
    m = sim_runs{i};
    fprintf('Lambda %d: %d data points per simulation\n', i, size(m, 1));
end