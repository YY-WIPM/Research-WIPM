%压缩感知重构算法OMP测试
%以一维信号为例
clear all;close all;clc;
M = 64;%观测值个数
N = 256;%信号x的长度
K = 5;%信号x的稀疏度
Index_K = randperm(N);
x = zeros(N,1);
%x(Index_K(1:K)) = 5*randn(K,1);%x为K稀疏的，且位置是随机的
x([50 100 150 200 250])=[5 1 3 4 2 ]
Psi = eye(N);%x本身是稀疏的，定义稀疏矩阵为单位阵，x=Psi*theta
Phi = randn(M,N);%测量矩阵为高斯矩阵
A = Phi * Psi;%传感矩阵
y = Phi * x;%得到观测向量y
iter =2
%% 恢复重构信号x
    [m,n] = size(y);
    if m<n
        y = y'; %y should be a column vector
    end
    [M,N] = size(A); %传感矩阵A为M*N矩阵
    theta = zeros(N,1); %用来存储恢复的theta(列向量)
    At = zeros(M,iter); %用来迭代过程中存储A被选择的列
    pos_num = zeros(1,iter); %用来迭代过程中存储A被选择的列序号
    res = y; %初始化残差(residual)为y
    for ii=1:iter %迭代t次，t为输入参数
        product = A'*res; %传感矩阵A各列与残差的内积
        [val,pos] = max(abs(product)); %找到最大内积绝对值，即与残差最相关的列
        At(:,ii) = A(:,pos); %存储这一列
        val_num(ii) = val;
        pos_num(ii) = pos; %存储这一列的序号
        A(:,pos) = zeros(M,1); %清零A的这一列，其实此行可以不要，因为它与残差正交
        %y=At(:,1:ii)*theta，以下求theta的最小二乘解(Least Square)
        theta_ls = (At(:,1:ii)'*At(:,1:ii))^(-1)*At(:,1:ii)'*y;%最小二乘解
        % At(:,1:ii)*theta_ls是y在At(:,1:ii)列空间上的正交投影
        res = y - At(:,1:ii)*theta_ls; %更新残差        
    end
    theta(pos_num)=theta_ls;% 恢复出的theta
%% 恢复重构信号x
x_r = Psi * theta;% x=Psi * theta
%% 绘图
figure;
plot(x_r,'k.-');%绘出x的恢复信号
hold on;
plot(x,'r');%绘出原信号x
hold off;
legend('Recovery','Original')
fprintf('\n恢复残差：');
norm(x_r-x)%恢复残差