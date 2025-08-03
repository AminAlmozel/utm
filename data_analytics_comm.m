close all
clear
clc
format compact
format shortG
base_dir = pwd;
path = "/full_runs/comm1/9run_25-07-31-233141/stats.csv";
m = importdata(fullfile(base_dir, path));
distance = m(:, 2);
time = m(:, 3);
comm = m(:, 4);
outside_comm = m(:, 5);
% Stats
[mu, sigma, ~, ~] = normfit(distance);

fprintf("Number of drones in simulation: %d drones\n", size(distance, 1))
fprintf("Average distance traveled: \t%.1fm (%.1f standard deviation)\n", [mu, sigma])
% [mu2, sigma, ~, ~] = normfit(d_o);
% fprintf("Straight line distance: \t%.1fm (%.1f standard deviation)\n", [mu2, sigma])
% fprintf("Difference on average: \t\t%.1fm\n", mu - mu2)

nbins = 30;
%%
% Distance histogram
hold on
[bins] = histogram(distance, nbins);
title('Distance (m)')
xlabel('Distance (m)')
ylabel('Number of drones')

% Fit Gaussian mixture model
gmModel = fitgmdist(distance, 2);

% Extract parameters
mu1 = gmModel.mu(1); % Mean of first component
mu2 = gmModel.mu(2); % Mean of second component
sigma1 = sqrt(gmModel.Sigma(1)); % Std of first component
sigma2 = sqrt(gmModel.Sigma(2)); % Std of second component
w1 = gmModel.ComponentProportion(1); % Weight of first component
w2 = gmModel.ComponentProportion(2); % Weight of second component

% Calculate bin width from the histogram
bin_width = bins.BinWidth;  % This gets the bin width from the histogram object

% Plot results
x = linspace(min(distance), max(distance), 1000);
pdf_fitted = pdf(gmModel, x');
scaled_pdf = pdf_fitted * length(distance) * bin_width;
plot(x, scaled_pdf, 'r-', 'LineWidth', 2);
legend('Safe trajectory distance', 'Safe trajectory normal fit')

%%
% Time histogram
figure
histogram(time, nbins)
title('Time (s)')
xlabel('Time (s)')
ylabel('Number of drones')

%%
% Comparison between safe, unsafe, and outside
B = sortrows(m, 2);
y = B(:, 4:5);
figure
hold on
% bar(y(:, 3) + y(:, 2) + y(:, 1))
bar(y(:, 2) + y(:, 1))
bar(y(:, 2))


xlabel('Drone number')
ylabel('Distance traveled')
legend('In Proximity', 'Outside Range')