function param = Set_Var(param)

%------�����㷨����------------

param.Size = 100;   % ��Ⱥ��ģ
param.Maxgen = 100;   % ��������
param.Mp = 0.5;   % (����)��������   (rangle:0~2)
param.Cp = 0.2;   % �������� �����ʣ�   ��rangle:0~1)

param.fun = str2func(param.Problem);    % �Ż�Ŀ�� ��� or ��С���ַ�ת���ţ�
% ------------��չ���߱���ȡֵ��Χ----------------
param.Var_max = repmat(param.Var_max,param.Size,1);
param.Var_min = repmat(param.Var_min,param.Size,1);
%% ----------------���칫ʽ������------------------
%
% # param.Mu_model = 'one';  % x1+F*(x2-x3)
% # param.Mu_model = 'two';  % gbest+F*(��x1-x2��+��x3-x4��)
%
param.Mu_model = 'two';
switch param.Mu_model
    case 'one'
        if param.Size < 4
            error('��Ⱥ��ģӦ�ô��ڵ���4')
        end
    case 'two'
        if param.Size < 5
            error('��Ⱥ��ģӦ�ô��ڵ���5')
        end
end