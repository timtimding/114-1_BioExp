present01=importdata('BIOPAC_2.txt');
%present02=importdata('EEG_raw.txt');

%% Basic Parameters
t = 0:0.005:60;     % 200 S/s
t_o = 0.005:0.005:20;
t_c = 0.005:0.005:20;
fs = 200;       % Sampling frequency
L = 12001;      % Data length
L_o = 4000;      % Data length (open)
L_c = 4000;      % Data length (closed)
% F0 = 3;         % Center frequency
Fnyq = fs / 2;  % Nyquist frequency


%% Initialize data
raw = zeros(size(t));
alpha = zeros(size(t));
beta = zeros(size(t));
delta = zeros(size(t));
theta = zeros(size(t));


for i = 1 : length(t)
    raw(i) = present01(i, 1);
end
%raw_ac = raw - mean(raw);

for i = 1 : length(t)
    alpha(i) = present01(i, 2);
end
for i = 1 : length(t)
    beta(i) = present01(i, 3);
end
for i = 1 : length(t)
    delta(i) = present01(i, 4);
end
for i = 1 : length(t)
    theta(i) = present01(i, 5);
end

%raw_closed = [raw(1 : 4000) raw(8001 : 12000)];
raw_open = raw(4001 : 8000);
raw_closed = raw(8001 : 12000);
%alpha_closed = [alpha(1 : 4000) alpha(8001 : 12000)];
alpha_open = alpha(4001 : 8000);
alpha_closed = alpha(8001 : 12000);
%beta_closed = [beta(1 : 4000) beta(8001 : 12000)];
beta_open = beta(4001 : 8000);
beta_closed = beta(8001 : 12000);
%delta_closed = [delta(1 : 4000) delta(8001 : 12000)];
delta_open = delta(4001 : 8000);
delta_closed = delta(8001 : 12000);
%theta_closed = [theta(1 : 4000) theta(8001 : 12000)];
theta_open = theta(4001 : 8000);
theta_closed = theta(8001 : 12000);
%% Filters
% Q = 5;          % Quality factor
% BW = 5;         % Bandwidth for the bandpass filter

% Butterworth(?) filter to filter out the waves
% Also use IIR filter (looks better)

% Alpha: Center frequency = 10
%[b, a] = designNotchPeakIIR(Response = "peak",...
%    CenterFrequency = 10 / Fnyq, ...
%    QualityFactor = 2, FilterOrder = 2);
[b, a] = butter(4, [8 13]/(fs/2), 'bandpass');
% eeg_alpha = filtfilt(b, a, raw);
eeg_alpha_open = filtfilt(b, a, raw_open);
eeg_alpha_closed = filtfilt(b, a, raw_closed);

% Beta: Center frequency = 15
% [d, c] = designNotchPeakIIR(Response = "peak",...
%    CenterFrequency = 15 / Fnyq, ...
%    QualityFactor = 2, FilterOrder = 2);

[d, c] = butter(4, [12 30]/(fs/2), 'bandpass');
%eeg_beta = filtfilt(d, c, raw);
eeg_beta_open = filtfilt(d, c, raw_open);
eeg_beta_closed = filtfilt(d, c, raw_closed);

% Delta: Center frequency = 1.3
%[f, e] = designNotchPeakIIR(Response = "peak",...
%    CenterFrequency = 1.3 / Fnyq, ...
%    QualityFactor = 1, FilterOrder = 2);
[f, e] = butter(4, [1 5]/(fs/2), 'bandpass');
%eeg_delta = filtfilt(f, e, raw);
eeg_delta_open = filtfilt(f, e, raw_open);
eeg_delta_closed = filtfilt(f, e, raw_closed);

% Theta: Center frequency = 4.2
%[h, g] = designNotchPeakIIR(Response = "peak",...
%    CenterFrequency = 4.2 / Fnyq, ...
%    QualityFactor = 2, FilterOrder = 2);
[h, g] = butter(4, [4 8]/(fs/2), 'bandpass');
%eeg_theta = filtfilt(h, g, raw);
eeg_theta_open = filtfilt(h, g, raw_open);
eeg_theta_closed = filtfilt(h, g, raw_closed);

%% fft to compare effects of BIOPAC and our filters
% BIOPAC

%Y_alpha = fftshift(fft(alpha));
%Y_beta = fftshift(fft(beta));
%Y_delta = fftshift(fft(delta));
%Y_theta = fftshift(fft(theta));

Y_alpha_open = fftshift(fft(alpha_open));
Y_alpha_closed = fftshift(fft(alpha_closed));
Y_beta_open = fftshift(fft(beta_open));
Y_beta_closed = fftshift(fft(beta_closed));
Y_delta_open = fftshift(fft(delta_open));
Y_delta_closed = fftshift(fft(delta_closed));
Y_theta_open = fftshift(fft(theta_open));
Y_theta_closed = fftshift(fft(theta_closed));

Y_raw = fftshift(fft(alpha));
[w, v] = butter(6, 1/(fs/2), 'high');
raw_filtered = filtfilt(w, v, raw);
Y_raw_ac = fftshift(fft(raw_filtered));

Y_raw_open = fftshift(fft(raw_open));
Y_raw_closed = fftshift(fft(raw_closed));

% Our filters
Y_alpha_filtered_open = fftshift(fft(eeg_alpha_open));
Y_alpha_filtered_closed = fftshift(fft(eeg_alpha_closed));
Y_beta_filtered_open = fftshift(fft(eeg_beta_open));
Y_beta_filtered_closed = fftshift(fft(eeg_beta_closed));
Y_delta_filtered_open = fftshift(fft(eeg_delta_open));
Y_delta_filtered_closed = fftshift(fft(eeg_delta_closed));
Y_theta_filtered_open = fftshift(fft(eeg_theta_open));
Y_theta_filtered_closed = fftshift(fft(eeg_theta_closed));

%% Prints figures out
close all;
% Raw data
figure;
subplot(3, 1, 1);
plot(t, raw, "LineWidth", 1);
title('Raw data');
ylim([-100 100]);

subplot(3, 1, 2);
plot(fs/L*(-L/2 : L/2 - 1), abs(Y_raw), "LineWidth", 1);
title('Raw - frequency');
subplot(3, 1, 3);
plot(fs/L*(-L/2 : L/2 - 1), abs(Y_raw_ac), "LineWidth", 1);
title('Raw - frequency (DC removed)');

% Time domain wave
figure;
subplot(4, 1, 1);
plot(t_o, alpha_open, "LineWidth", 1);
title('Alpha open - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 2);
plot(t_o, eeg_alpha_open, "LineWidth", 1);
title('Alpha open - Our filters');
ylim([-25 25]);
subplot(4, 1, 3);
plot(t_c, alpha_closed, "LineWidth", 1);
title('Alpha closed - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 4);
plot(t_c, eeg_alpha_closed, "LineWidth", 1);
title('Alpha closed - Our filters');
ylim([-25 25]);

figure;
subplot(4, 1, 1);
plot(t_o, eeg_beta_open, "LineWidth", 1);
title('Beta open - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 2);
plot(t_o, beta_open, "LineWidth", 1);
title('Beta open - Our filters');
ylim([-25 25]);
subplot(4, 1, 3);
plot(t_c, eeg_beta_closed, "LineWidth", 1);
title('Beta closed - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 4);
plot(t_c, beta_closed, "LineWidth", 1);
title('Beta closed - Our filters');
ylim([-25 25]);

figure;
subplot(4, 1, 1);
plot(t_o, delta_open, "LineWidth", 1);
title('Delta open - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 2);
plot(t_o, eeg_delta_open, "LineWidth", 1);
title('Delta open - Our filters');
ylim([-25 25]);
subplot(4, 1, 3);
plot(t_c, delta_closed, "LineWidth", 1);
title('Delta closed - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 4);
plot(t_c, eeg_delta_closed, "LineWidth", 1);
title('Delta closed - Our filters');
ylim([-25 25]);

figure;
subplot(4, 1, 1);
plot(t_o, eeg_theta_open, "LineWidth", 1);
title('Theta open - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 2);
plot(t_o, theta_open, "LineWidth", 1);
title('Theta open - Our filters');
ylim([-25 25]);
subplot(4, 1, 3);
plot(t_c, eeg_theta_closed, "LineWidth", 1);
title('Theta closed - BIOPAC');
ylim([-25 25]);
subplot(4, 1, 4);
plot(t_c, theta_closed, "LineWidth", 1);
title('Theta closed - Our filters');
ylim([-25 25]);

% Frequency domain
figure;
subplot(2, 2, 1);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_alpha_open), "LineWidth", 1);
title('Alpha open - BIOPAC');
subplot(2, 2, 3);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_alpha_filtered_open), "LineWidth", 1);
title('Alpha open - Our filters');
subplot(2, 2, 2);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_alpha_closed), "LineWidth", 1);
title('Alpha closed - BIOPAC');
subplot(2, 2, 4);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_alpha_filtered_closed), "LineWidth", 1);
title('Alpha closed - Our filters');

figure;
subplot(2, 2, 1);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_beta_open), "LineWidth", 1);
title('Beta open - BIOPAC');
subplot(2, 2, 3);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_beta_filtered_open), "LineWidth", 1);
title('Beta open - Our filters');
subplot(2, 2, 2);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_beta_closed), "LineWidth", 1);
title('Beta closed - BIOPAC');
subplot(2, 2, 4);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_beta_filtered_closed), "LineWidth", 1);
title('Beta closed - Our filters');

figure;
subplot(2, 2, 1);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_delta_open), "LineWidth", 1);
title('Detla open - BIOPAC');
subplot(2, 2, 3);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_delta_filtered_open), "LineWidth", 1);
title('Delta open - Our filters');
subplot(2, 2, 2);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_delta_closed), "LineWidth", 1);
title('Detla closed - BIOPAC');
subplot(2, 2, 4);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_delta_filtered_closed), "LineWidth", 1);
title('Delta closed - Our filters');

figure
subplot(2, 2, 1);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_theta_open), "LineWidth", 1);
title('Theta open - BIOPAC');
subplot(2, 2, 3);
plot(fs/L_o*(-L_o/2 : L_o/2 - 1), abs(Y_theta_filtered_open), "LineWidth", 1);
title('Theta open - Our filters');
subplot(2, 2, 2);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_theta_closed), "LineWidth", 1);
title('Theta closed - BIOPAC');
subplot(2, 2, 4);
plot(fs/L_c*(-L_c/2 : L_c/2 - 1), abs(Y_theta_filtered_closed), "LineWidth", 1);
title('Theta closed - Our filters');
