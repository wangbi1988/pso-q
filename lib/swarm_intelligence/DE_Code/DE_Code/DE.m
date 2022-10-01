%% DE Algorithm

%%   清除工作区
clear
clc
%% ----------------构造优化目标-------------
param.Test_function = 'Schaffer'; % 待优化函数类型 Schaffer Ackley
param = Set_function_Var(param); % 待优化函数参数
CostFunction = @(x,y_type)Optimal_function(x,y_type); % 待优化函数类型
%% ---------------定义算法参数--------------
param = Set_Var(param);
%% ------------------初始化-----------------
pop = Code_Population(param,CostFunction);

%% ---------------------迭代寻优-------------
for t_num = 1:param.Maxgen
    % ------------------变异---------------------
    var_Mu = Count_Mutate(pop,param,t_num);
    % ------------------交叉---------------------
    var_Cr = Count_Cross(pop,param,var_Mu);
    % ------------------选择---------------------
    pop = Count_Select(pop,param,var_Cr,CostFunction,t_num);
end
%% -----------出图---------------
figure(1)
plot(1:param.Maxgen,pop.Gbest_Value(2:end),'r-o','LineWidth',2)
xlabel('进化代数')
ylabel('适应度函数值')
axis tight
grid on