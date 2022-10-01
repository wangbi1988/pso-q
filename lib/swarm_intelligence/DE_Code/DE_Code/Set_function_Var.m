function param = Set_function_Var(param)
%  建立函数
switch param.Test_function
    case'Schaffer'
        param.Dim = 2;
        param.Var_max = 10*ones(1,param.Dim); % 决策变量上限
        param.Var_min = -10*ones(1,param.Dim); % 决策变量下限
        param.Problem = 'min'; % 最大 or 最小
    case'Ackley'
        param.Dim = 2;
        param.Var_max = 30*ones(1,param.Dim); % 决策变量上限
        param.Var_min = -30*ones(1,param.Dim); % 决策变量下限
        param.Problem = 'min'; % 最大 or 最小
end
