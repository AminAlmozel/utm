close all
clear
clc
format compact
format shortG

% Define the parent directory containing the folders
base_dir = pwd;
path = "/full_runs/mission3"; % Replace with your path
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

mean_distance = [];
mean_time = [];
mean_safe = [];
mean_unsafe = [];
mean_outside = [];
for i = 1:length(sim_runs)
    % Process each simulation run (data table)
    m = sim_runs{i};
    % Example processing: Calculate the mean of a specific column, e.g., 'distance'
    distance = m(:, 2);
    time = m(:, 3);
    safe = m(:, 4);
    unsafe = m(:, 5);
    outside = m(:, 6);

    % Calculate means
    mean_distance = [mean_distance, mean(distance)];
    mean_time = [mean_time, mean(time)];
    mean_safe = [mean_safe, mean(safe)];
    mean_unsafe = [mean_unsafe, mean(unsafe)];
    mean_outside = [mean_outside, mean(outside)];
end
% Assuming you have your data arrays
x = 1:length(mean_distance); % x-axis values (indices)
x= [0, 0.3, 0.5, 0.7, 0.9]
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
text_offset = y_range * 0.1;  % Offset text by 5% of the y-range

for i = 1:length(x)  % Start from 2 since first point is the reference
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

% Create figure
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
lambda = x
safe_percentage