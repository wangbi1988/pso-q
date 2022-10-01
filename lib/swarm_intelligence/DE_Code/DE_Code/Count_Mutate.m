function var_Mu = Count_Mutate(pop,param,t_num)

var_Mu = pop.Var;
switch param.Mu_model
    case 'one'
        for u = 1:param.Size
            temp = randperm(param.Size,4);   % 随机选出4个个体
            temp(temp == u) = [];  % 将等于当前个体的索引删除
            %      变异产生新个体
            var_Mu(u,:) = pop.Var(temp(1),:)+param.Mp*(pop.Var(temp(2),:)-pop.Var(temp(3),:));          
        end
    case 'two'
        for u = 1:param.Size
            temp = randperm(param.Size,5);   % 随机选出4个个体
            temp(temp == u) = [];  % 将等于当前个体的索引删除
            %      变异产生新个体
            var_Mu(u,:) = pop.Gbest_Var(t_num,:)+param.Mp*(pop.Var(temp(1),:)-...
                pop.Var(temp(2),:)+pop.Var(temp(3),:)-pop.Var(temp(4),:));
        end
end
% ----------------------限制取值范围------------------------------
temp_logical = var_Mu>param.Var_max;
var_Mu(temp_logical) = param.Var_max(temp_logical);

temp_logical = var_Mu<param.Var_min;
var_Mu(temp_logical) = param.Var_min(temp_logical);