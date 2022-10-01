function pop = Code_Population(param,CostFunction)

%%   ��ʼ����Ⱥ
pop.Var = param.Var_min + (param.Var_max-param.Var_min).*rand(param.Size,param.Dim);

%%  ������Ӧ�Ⱥ���ֵ
for u = 1:param.Size
    pop.Value(u,:) = CostFunction(pop.Var(u,:),param.Test_function);
end

pop.Gbest_Var = zeros(param.Maxgen+1,param.Dim); % 
pop.Gbest_Value = zeros(param.Maxgen+1,1); % 
% --------------�Ƚ�����--------------------
[pop.Gbest_Value(1),Gbest_Index] = param.fun(pop.Value);  % ��ǰ������Ӧ�Ⱥ���ֵ
pop.Gbest_Var(1,:) = pop.Var(Gbest_Index,:);  %  ��ǰ������Ӧ�Ⱥ���ֵ��Ӧ�ĸ��� or ���߱���