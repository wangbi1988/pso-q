function param = Set_Var(param)

%------定义算法参数------------

param.Size = 100;   % 种群规模
param.Maxgen = 100;   % 进化代数
param.Mp = 0.5;   % (步长)变异算子   (rangle:0~2)
param.Cp = 0.2;   % 交叉算子 （概率）   （rangle:0~1)

param.fun = str2func(param.Problem);    % 优化目标 最大 or 最小（字符转符号）
% ------------扩展决策变量取值范围----------------
param.Var_max = repmat(param.Var_max,param.Size,1);
param.Var_min = repmat(param.Var_min,param.Size,1);
%% ----------------变异公式的类型------------------
%
% # param.Mu_model = 'one';  % x1+F*(x2-x3)
% # param.Mu_model = 'two';  % gbest+F*(（x1-x2）+（x3-x4）)
%
param.Mu_model = 'two';
switch param.Mu_model
    case 'one'
        if param.Size < 4
            error('种群规模应该大于等于4')
        end
    case 'two'
        if param.Size < 5
            error('种群规模应该大于等于5')
        end
end