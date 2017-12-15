function [ theta ] = CS_OMP( y,A,iter )
%   CS_OMP
%   y = Phi * x
%   x = Psi * theta
%    y = Phi * Psi * theta
%   �� A = Phi*Psi, ��y=A*theta
%   ������֪y��A����theta
%   iter = �������� 
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
        pos_num(ii) = pos; %�洢��һ�е����
        A(:,pos) = zeros(M,1); %����A����һ�У���ʵ���п��Բ�Ҫ����Ϊ����в�����
        % y=At(:,1:ii)*theta��������theta����С���˽�(Least Square)
        theta_ls = (At(:,1:ii)'*At(:,1:ii))^(-1)*At(:,1:ii)'*y;%��С���˽�
        % At(:,1:ii)*theta_ls��y��At(:,1:ii)�пռ��ϵ�����ͶӰ
        res = y - At(:,1:ii)*theta_ls; %���²в�        
    end
    theta(pos_num)=theta_ls;% �ָ�����theta
end