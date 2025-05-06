% utils/denormalize_data.m
function data = denormalize_data(normalized_data, norm_params)
    % 使用提供的参数对数据进行反归一化
    % 输入:
    %   normalized_data (N x D) - 归一化数据
    %   norm_params - 包含均值(mu)和标准差(sigma)的结构体
    % 输出:
    %   data (N x D) - 反归一化后的数据

    mu = norm_params.mu;
    sigma = norm_params.sigma;

    data = normalized_data .* sigma + mu;
end