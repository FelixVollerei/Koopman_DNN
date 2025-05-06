% generate_data.m
% 功能: 生成非线性系统的仿真数据并保存

clearvars; close all; clc;

fprintf('开始生成数据...\n');

% --- 参数设置 ---
dt = 0.01; % 时间步长

% --- 1. Van der Pol 振子 ---
fprintf('生成 Van der Pol 数据...\n');
mu = 1.5; % vdp 参数
vdp_ode = @(t, x) [x(2); mu * (1 - x(1)^2) * x(2) - x(1)];
tspan_vdp = 0:dt:30;
x0_vdp = [1.0; 0.0];
[t_vdp, x_vdp] = ode45(vdp_ode, tspan_vdp, x0_vdp);
save('vanderpol_data.mat', 't_vdp', 'x_vdp', 'mu', 'dt');
fprintf('Van der Pol 数据已保存到 vanderpol_data.mat\n');

% --- 2. Duffing 振子 ---
fprintf('生成 Duffing 数据...\n');
alpha = -1.0;
beta = 1.0;
delta = 0.3;
gamma = 0.5; % 驱动力幅值
omega = 1.2; % 驱动力频率
duffing_ode = @(t, x) [x(2); -delta * x(2) - alpha * x(1) - beta * x(1)^3 + gamma * cos(omega * t)];
tspan_duffing = 0:dt:100;
x0_duffing = [1.0; 0.0];
[t_duffing, x_duffing] = ode45(duffing_ode, tspan_duffing, x0_duffing);
save('duffing_data.mat', 't_duffing', 'x_duffing', 'alpha', 'beta', 'delta', 'gamma', 'omega', 'dt');
fprintf('Duffing 数据已保存到 duffing_data.mat\n');

% --- 3. Lorenz 系统 ---
fprintf('生成 Lorenz 数据...\n');
sigma = 10.0;
rho = 28.0;
beta_lorenz = 8/3;
lorenz_ode = @(t, x) [sigma * (x(2) - x(1)); x(1) * (rho - x(3)) - x(2); x(1) * x(2) - beta_lorenz * x(3)];
tspan_lorenz = 0:dt:50;
x0_lorenz = [-8; 8; 27];
[t_lorenz, x_lorenz] = ode45(lorenz_ode, tspan_lorenz, x0_lorenz);
save('lorenz_data.mat', 't_lorenz', 'x_lorenz', 'sigma', 'rho', 'beta_lorenz', 'dt');
fprintf('Lorenz 数据已保存到 lorenz_data.mat\n');

% --- 4. Rossler 系统 ---
fprintf('生成 Rossler 数据...\n');
a = 0.2;
b = 0.2;
c = 5.7;
rossler_ode = @(t, x) [-x(2) - x(3); x(1) + a * x(2); b + x(3) * (x(1) - c)];
tspan_rossler = 0:dt:200;
x0_rossler = [0; -5; 0];
[t_rossler, x_rossler] = ode45(rossler_ode, tspan_rossler, x0_rossler);
save('rossler_data.mat', 't_rossler', 'x_rossler', 'a', 'b', 'c', 'dt');
fprintf('Rossler 数据已保存到 rossler_data.mat\n');

% --- 5. Lotka-Volterra 系统 ---
fprintf('生成 Lotka-Volterra 数据...\n');
alpha_lv = 1.1;
beta_lv = 0.4;
delta_lv = 0.1;
gamma_lv = 0.4;
lv_ode = @(t, x) [alpha_lv * x(1) - beta_lv * x(1) * x(2); delta_lv * x(1) * x(2) - gamma_lv * x(2)];
tspan_lv = 0:dt:50;
x0_lv = [20; 5]; % 初始种群数量
[t_lv, x_lv] = ode45(lv_ode, tspan_lv, x0_lv);
save('lotka_volterra_data.mat', 't_lv', 'x_lv', 'alpha_lv', 'beta_lv', 'delta_lv', 'gamma_lv', 'dt');
fprintf('Lotka-Volterra 数据已保存到 lotka_volterra_data.mat\n');

fprintf('数据生成完成。\n');