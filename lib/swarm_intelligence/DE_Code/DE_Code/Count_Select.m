function pop = Count_Select(pop,param,var_Cr,CostFunction,t_num)

%% ----------------计算交叉后的适应度函数值----------------
Value_Cr = pop.Value;
for u = 1:param.Size
    Value_Cr(u,:) = CostFunction(var_Cr(u,:),param.Test_function);
end
%% --------------------------选择------------------------

[temp_Value,temp_logi] = param.fun([pop.Value,Value_Cr],[],2);

pop.Value = temp_Value;  % 新适应度函数赋值

temp_logi = logical(temp_logi-1);  % 转化为逻辑变量

pop.Var(temp_logi,:) = var_Cr(temp_logi,:); %  个体决策变量和替换
%% ----------------------更新全局最优--------------------
[pop.Gbest_Value(t_num+1),temp_Index] = param.fun([pop.Value;pop.Gbest_Value(t_num)]);
temp_Var = [pop.Var;pop.Gbest_Var(t_num,:)];
pop.Gbest_Var(t_num+1,:) = temp_Var(temp_Index,:);