function pop = Code_Population(param,CostFunction)

%%   初始化种群
pop.Var = param.Var_min + (param.Var_max-param.Var_min).*rand(param.Size,param.Dim);

%%  计算适应度函数值
for u = 1:param.Size
    pop.Value(u,:) = CostFunction(pop.Var(u,:),param.Test_function);
end

pop.Gbest_Var = zeros(param.Maxgen+1,param.Dim); % 
pop.Gbest_Value = zeros(param.Maxgen+1,1); % 
% --------------比较最优--------------------
[pop.Gbest_Value(1),Gbest_Index] = param.fun(pop.Value);  % 当前最优适应度函数值
pop.Gbest_Var(1,:) = pop.Var(Gbest_Index,:);  %  当前最优适应度函数值对应的个体 or 决策变量