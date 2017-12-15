%ѹ����֪�ع��㷨OMP����
%��һά�ź�Ϊ��
clear all;close all;clc;
M = 64;%�۲�ֵ����
N = 256;%�ź�x�ĳ���
K = 5;%�ź�x��ϡ���
Index_K = randperm(N);
x = zeros(N,1);
%x(Index_K(1:K)) = 5*randn(K,1);%xΪKϡ��ģ���λ���������
x([50 100 150 200 250])=[5 1 3 4 2 ]
Psi = eye(N);%x������ϡ��ģ�����ϡ�����Ϊ��λ��x=Psi*theta
Phi = randn(M,N);%��������Ϊ��˹����
A = Phi * Psi;%���о���
y = Phi * x;%�õ��۲�����y
iter =2
%% �ָ��ع��ź�x
    [m,n] = size(y);
    if m<n
        y = y'; %y should be a column vector
    end
    [M,N] = size(A); %���о���AΪM*N����
    theta = zeros(N,1); %�����洢�ָ���theta(������)
    At = zeros(M,iter); %�������������д洢A��ѡ�����
    pos_num = zeros(1,iter); %�������������д洢A��ѡ��������
    res = y; %��ʼ���в�(residual)Ϊy
    for ii=1:iter %����t�Σ�tΪ�������
        product = A'*res; %���о���A������в���ڻ�
        [val,pos] = max(abs(product)); %�ҵ�����ڻ�����ֵ������в�����ص���
        At(:,ii) = A(:,pos); %�洢��һ��
        val_num(ii) = val;
        pos_num(ii) = pos; %�洢��һ�е����
        A(:,pos) = zeros(M,1); %����A����һ�У���ʵ���п��Բ�Ҫ����Ϊ����в�����
        %y=At(:,1:ii)*theta��������theta����С���˽�(Least Square)
        theta_ls = (At(:,1:ii)'*At(:,1:ii))^(-1)*At(:,1:ii)'*y;%��С���˽�
        % At(:,1:ii)*theta_ls��y��At(:,1:ii)�пռ��ϵ�����ͶӰ
        res = y - At(:,1:ii)*theta_ls; %���²в�        
    end
    theta(pos_num)=theta_ls;% �ָ�����theta
%% �ָ��ع��ź�x
x_r = Psi * theta;% x=Psi * theta
%% ��ͼ
figure;
plot(x_r,'k.-');%���x�Ļָ��ź�
hold on;
plot(x,'r');%���ԭ�ź�x
hold off;
legend('Recovery','Original')
fprintf('\n�ָ��в');
norm(x_r-x)%�ָ��в�