function param = Set_function_Var(param)
%  ��������
switch param.Test_function
    case'Schaffer'
        param.Dim = 2;
        param.Var_max = 10*ones(1,param.Dim); % ���߱�������
        param.Var_min = -10*ones(1,param.Dim); % ���߱�������
        param.Problem = 'min'; % ��� or ��С
    case'Ackley'
        param.Dim = 2;
        param.Var_max = 30*ones(1,param.Dim); % ���߱�������
        param.Var_min = -30*ones(1,param.Dim); % ���߱�������
        param.Problem = 'min'; % ��� or ��С
end
