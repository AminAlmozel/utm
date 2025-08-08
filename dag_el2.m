close all
clear
clc
format compact
format shortG

% Define the parent directory containing the folders
base_dir = pwd;
path = "/full_runs/emergency_landing1"; % Replace with your path
% Get a list of all subfolders
parentDir = fullfile(base_dir, path);
subfolders = dir(parentDir);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.')); % Exclude '.' and '..'

dt = 0.1;
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

mean_distance = [];
mean_time = [];
mean_safe = [];
mean_unsafe = [];
mean_outside = [];
mean_response = []; % Fixed variable name consistency
all_drone_data = {}; % Store all drone data for each lambda

for i = 1:length(sim_runs)
    % Process each simulation run (data table)
    m = sim_runs{i};
    % Example processing: Calculate the mean of a specific column, e.g., 'distance'
    distance = m(:, 2);
    time = m(:, 3);
    safe = m(:, 4);
    unsafe = m(:, 5);
    outside = m(:, 6);
    response = m(:, 7);

    % Extract non-zero response times and corresponding drone numbers
    drone_numbers = m(:, 1); % Assuming first column contains drone numbers
    non_zero_indices = response ~= 0;
    
    % Store drone data for this lambda value
    drone_data = struct();
    drone_data.drone_numbers = drone_numbers(non_zero_indices);
    drone_data.response_times = response(non_zero_indices);
    drone_data.lambda_index = i;
    all_drone_data{i} = drone_data;

    % Calculate means
    mean_distance = [mean_distance, mean(distance)];
    mean_time = [mean_time, mean(time)];
    mean_safe = [mean_safe, mean(safe)];
    mean_unsafe = [mean_unsafe, mean(unsafe)];
    mean_outside = [mean_outside, mean(outside)];
    
    % Calculate mean response time only for non-zero values
    if sum(non_zero_indices) > 0
        mean_response = [mean_response, mean(response(non_zero_indices))];
    else
        mean_response = [mean_response, 0]; % No responses recorded
    end
end
x = [0, 0.3, 0.5, 0.7, 0.9];
% Find common drone IDs across all experiments
fprintf('\n=== Finding Common Drone IDs Across All Experiments ===\n');

% Get all unique drone IDs from each experiment
drone_sets = cell(length(all_drone_data), 1);
for i = 1:length(all_drone_data)
    drone_sets{i} = unique(all_drone_data{i}.drone_numbers);
    fprintf('Lambda %.1f: %d responding drones\n', x(i), length(drone_sets{i}));
end

% Find intersection of all sets (common drone IDs)
if ~isempty(drone_sets)
    common_drone_ids = drone_sets{1};
    for i = 2:length(drone_sets)
        common_drone_ids = intersect(common_drone_ids, drone_sets{i});
    end
else
    common_drone_ids = [];
end

fprintf('\nCommon drone IDs across all experiments: ');
if ~isempty(common_drone_ids)
    fprintf('%d ', common_drone_ids);
    fprintf('\nNumber of common responding drones: %d\n', length(common_drone_ids));
else
    fprintf('None found\n');
end

% Extract response times for common drones only
common_drone_response_data = {};
mean_response_common = [];

for i = 1:length(all_drone_data)
    drone_data = all_drone_data{i};
    
    % Find indices of common drones in this experiment
    [~, common_indices] = intersect(drone_data.drone_numbers, common_drone_ids);
    
    % Store data for common drones
    common_data = struct();
    common_data.drone_numbers = drone_data.drone_numbers(common_indices);
    common_data.response_times = drone_data.response_times(common_indices);
    common_data.lambda_index = i;
    common_drone_response_data{i} = common_data;
    
    % Calculate mean response time for common drones
    if ~isempty(common_data.response_times)
        mean_response_common = [mean_response_common, mean(common_data.response_times)];
    else
        mean_response_common = [mean_response_common, NaN];
    end
end

% Assuming you have your data arrays
x = [0, 0.3, 0.5, 0.7, 0.9];
figure;

% Plot the main lines
h1 = plot(x, mean_distance, '-o', 'LineWidth', 2, 'DisplayName', 'Mean Total Distance');
hold on;
h2 = plot(x, mean_safe, '-s', 'LineWidth', 2, 'DisplayName', 'Mean Safe Distance');
h3 = plot(x, mean_unsafe, '-^', 'LineWidth', 2, 'DisplayName', 'Mean Unsafe Distance');

% Get colors for consistency
colors = [h1.Color; h2.Color; h3.Color];

% Add horizontal dashed reference lines using first data points
yline(mean_distance(1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
yline(mean_safe(1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
yline(mean_unsafe(1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');

% Add vertical difference indicators (arrows/lines)
for i = 2:length(x)  % Start from 2 since first point is the reference
    % Mean Distance differences
    if mean_distance(i) ~= mean_distance(1)
        plot([x(i) x(i)], [mean_distance(1) mean_distance(i)], ':', 'Color', colors(1,:), 'HandleVisibility', 'off');
    end
    
    % Mean Safe differences  
    if mean_safe(i) ~= mean_safe(1)
        plot([x(i) x(i)], [mean_safe(1) mean_safe(i)], ':', 'Color', colors(2,:), 'HandleVisibility', 'off');
    end
    
    % Mean Unsafe differences
    if mean_unsafe(i) ~= mean_unsafe(1)
        plot([x(i) x(i)], [mean_unsafe(1) mean_unsafe(i)], ':', 'Color', colors(3,:), 'HandleVisibility', 'off');
    end
end

% Add vertical difference indicators (arrows/lines) and percentage annotations
y_range = max([max(mean_distance), max(mean_safe), max(mean_unsafe)]) - min([min(mean_distance), min(mean_safe), min(mean_unsafe)]);
text_offset = y_range * 0.1;  % Offset text by 10% of the y-range

for i = 1:length(x)  % Start from 1 since first point is the reference
    % Mean Distance differences
    if mean_distance(1) ~= 0
        plot([x(i) x(i)], [mean_distance(1) mean_distance(i)], ':', 'Color', colors(1,:), 'HandleVisibility', 'off');
        pct_change = ((mean_distance(i) - mean_distance(1)) / mean_distance(1)) * 100;
        text(x(i), mean_distance(i) + text_offset, sprintf('%.1f%%', pct_change), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', colors(1,:), ...
             'BackgroundColor', 'white', 'EdgeColor', 'none');
    end
    
    % Mean Safe differences  
    if mean_safe(1) ~= 0
        plot([x(i) x(i)], [mean_safe(1) mean_safe(i)], ':', 'Color', colors(2,:), 'HandleVisibility', 'off');
        pct_change = ((mean_safe(i) - mean_safe(1)) / mean_safe(1)) * 100;
        text(x(i), mean_safe(i) + text_offset, sprintf('%.1f%%', pct_change), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', colors(2,:), ...
             'BackgroundColor', 'white', 'EdgeColor', 'none');
    end
    
    % Mean Unsafe differences
    if mean_unsafe(1) ~= 0
        plot([x(i) x(i)], [mean_unsafe(1) mean_unsafe(i)], ':', 'Color', colors(3,:), 'HandleVisibility', 'off');
        pct_change = ((mean_unsafe(i) - mean_unsafe(1)) / mean_unsafe(1)) * 100;
        text(x(i), mean_unsafe(i) + text_offset, sprintf('%.1f%%', pct_change), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', colors(3,:), ...
             'BackgroundColor', 'white', 'EdgeColor', 'none');
    end
end
hold off;
xlabel('\lambda_{safe}');
ylabel('Mean Distance (m)');
% title('Comparison of Safety for Different   \lambdas');
legend('show');
xlim([-0.05 0.9]);
grid on;

% Calculate percentage of safe distance from total distance
safe_percentage = (mean_safe ./ mean_distance) * 100;

% Create figure for safe distance percentage
figure;

% Plot the safe distance percentage
h1 = plot(x, safe_percentage, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
          'DisplayName', 'Safe Distance Percentage');
hold on;

% Add grid
grid on;
grid minor;

% Add labels and formatting
xlabel('\lambda_{safe}');
ylabel('Safe Distance Percentage (%)');
% title('Percentage of Trajectory with Safe Distance vs \lambda_{safe}');

% Set axis limits
xlim([-0.05, 0.9]);
ylim([0, 100]);

% Add percentage value annotations above each point
for i = 1:length(x)
    text(x(i), safe_percentage(i) + 10, sprintf('%.1f%%', safe_percentage(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

% Optional: Add reference lines for key percentages
yline(50, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1, 'Alpha', 0.7);
yline(75, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1, 'Alpha', 0.7);

% Enhance appearance
set(gca, 'FontSize', 11);
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);

hold off;

% NEW PLOT: Lambda vs Response (All Drones and Common Drones)
figure;

% Plot lambda vs mean response for all drones
h1 = plot(x, mean_response, '-d', 'LineWidth', 2, 'MarkerSize', 8, ...
          'DisplayName', 'Mean Response (All Drones)');
hold on;

% Plot lambda vs mean response for common drones only
h2 = plot(x, mean_response_common, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
          'DisplayName', 'Mean Response (Common Drones Only)');

% Add grid
grid on;
grid minor;

% Add labels and formatting
xlabel('\lambda_{safe}');
ylabel('Mean Emergency Response Time');
% title('Emergency Response vs \lambda_{safe}');

% Set axis limits
xlim([-0.05, 0.9]);

% Add value annotations above each point for all drones
y_range_response = max([mean_response, mean_response_common]) - min([mean_response, mean_response_common]);
text_offset_response = y_range_response * 0.08;

for i = 1:length(x)
    % Annotations for all drones
    text(x(i), mean_response(i) + text_offset_response, sprintf('%.3f', mean_response(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
         'Color', h1.Color);
    
    % Annotations for common drones (if data exists)
    if ~isnan(mean_response_common(i))
        text(x(i), mean_response_common(i) - text_offset_response, sprintf('%.3f', mean_response_common(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', ...
             'Color', h2.Color);
    end
end

% Add horizontal reference lines
yline(mean_response(1), '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
if ~isnan(mean_response_common(1))
    yline(mean_response_common(1), ':', 'Color', [0.7 0.3 0.3], 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Enhance appearance
set(gca, 'FontSize', 11);
set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
legend('show', 'Location', 'best');

hold off;

% Store variables for reference
lambda = x;
safe_percentage

% Display extracted drone data for each lambda value
fprintf('\n=== All Drone Response Data (Non-zero values only) ===\n');
for i = 1:length(all_drone_data)
    drone_data = all_drone_data{i};
    lambda_val = lambda(i);
    
    fprintf('\nLambda = %.1f:\n', lambda_val);
    if ~isempty(drone_data.drone_numbers)
        fprintf('  All responding drone numbers: ');
        fprintf('%d ', drone_data.drone_numbers);
        fprintf('\n  All response times: ');
        fprintf('%.3f ', drone_data.response_times);
        fprintf('\n  Total responding drones: %d\n', length(drone_data.drone_numbers));
        fprintf('  Mean response time (all): %.3f\n', mean(drone_data.response_times));
    else
        fprintf('  No drones responded (all response times were zero)\n');
    end
end

fprintf('\n=== Common Drone Response Data ===\n');
for i = 1:length(common_drone_response_data)
    common_data = common_drone_response_data{i};
    lambda_val = lambda(i);
    
    fprintf('\nLambda = %.1f:\n', lambda_val);
    if ~isempty(common_data.drone_numbers)
        fprintf('  Common responding drone numbers: ');
        fprintf('%d ', common_data.drone_numbers);
        fprintf('\n  Common drone response times: ');
        fprintf('%.3f ', common_data.response_times);
        fprintf('\n  Mean response time (common drones): %.3f\n', mean(common_data.response_times));
    else
        fprintf('  No common drones found for this lambda\n');
    end
end

% Create comprehensive visualization
figure;

% Calculate number of responding drones for each lambda (all vs common)
num_responding_drones_all = zeros(size(lambda));
num_responding_drones_common = zeros(size(lambda));

for i = 1:length(all_drone_data)
    num_responding_drones_all(i) = length(all_drone_data{i}.drone_numbers);
    num_responding_drones_common(i) = length(common_drone_response_data{i}.drone_numbers);
end

% Plot 1: Number of responding drones comparison
% subplot(3,1,1);
% bar_data = [num_responding_drones_all; num_responding_drones_common]';
% h_bar = bar(lambda, bar_data, 'grouped');
% h_bar(1).FaceColor = [0.2 0.6 0.8];
% h_bar(2).FaceColor = [0.8 0.3 0.2];
% h_bar(1).DisplayName = 'All Responding Drones';
% h_bar(2).DisplayName = 'Common Responding Drones';
% xlabel('\lambda_{safe}');
% ylabel('Number of Drones');
% title('Number of Responding Drones: All vs Common');
% legend('show');
% grid on;
% 
% % Add value labels on bars
% for i = 1:length(lambda)
%     text(lambda(i)-0.1, num_responding_drones_all(i) + 0.1, sprintf('%d', num_responding_drones_all(i)), ...
%          'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
%     text(lambda(i)+0.1, num_responding_drones_common(i) + 0.1, sprintf('%d', num_responding_drones_common(i)), ...
%          'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8);
% end

% Plot 2: Box plot of response times for all drones
subplot(2,1,1);
response_data_all = [];
group_labels_all = [];

for i = 1:length(all_drone_data)
    if ~isempty(all_drone_data{i}.response_times * dt)
        response_data_all = [response_data_all; all_drone_data{i}.response_times * dt];
        group_labels_all = [group_labels_all; repmat(lambda(i), length(all_drone_data{i}.response_times), 1)];
    end
end

if ~isempty(response_data_all)
    boxplot(response_data_all, group_labels_all);
    xlabel('\lambda_{safe}');
    ylabel('Response Time (s)');
    title('Response Time Distribution - All Drones');
    grid on;
end

% Plot 3: Box plot of response times for common drones only
% subplot(2,1,2);
figure
response_data_common = [];
group_labels_common = [];

for i = 1:length(common_drone_response_data)
    if ~isempty(common_drone_response_data{i}.response_times)
        response_data_common = [response_data_common; common_drone_response_data{i}.response_times * dt];
        group_labels_common = [group_labels_common; repmat(lambda(i), length(common_drone_response_data{i}.response_times * dt), 1)];
    end
end

if ~isempty(response_data_common)
    boxplot(response_data_common, group_labels_common);
    xlabel('\lambda_{safe}');
    ylabel('Response Time (s)');
    % title('Response Time Distribution - Common Drones Only');
    grid on;
else
    text(0.5, 0.5, 'No common drone response data available', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'Units', 'normalized');
    title('Response Time Distribution - Common Drones Only');
end

% Display summary statistics
fprintf('\n=== Summary Statistics ===\n');
fprintf('Common drone IDs: ');
fprintf('%d ', common_drone_ids);
fprintf('\nTotal experiments: %d\n', length(lambda));
fprintf('Lambda values: ');
fprintf('%.1f ', lambda);
fprintf('\n\nMean response times (all drones): ');
fprintf('%.3f ', mean_response);
fprintf('\nMean response times (common drones): ');
fprintf('%.3f ', mean_response_common);
fprintf('\n');