function pop = Count_Select(pop,param,var_Cr,CostFunction,t_num)

%% ----------------���㽻������Ӧ�Ⱥ���ֵ----------------
Value_Cr = pop.Value;
for u = 1:param.Size
    Value_Cr(u,:) = CostFunction(var_Cr(u,:),param.Test_function);
end
%% --------------------------ѡ��------------------------

[temp_Value,temp_logi] = param.fun([pop.Value,Value_Cr],[],2);

pop.Value = temp_Value;  % ����Ӧ�Ⱥ�����ֵ

temp_logi = logical(temp_logi-1);  % ת��Ϊ�߼�����

pop.Var(temp_logi,:) = var_Cr(temp_logi,:); %  ������߱������滻
%% ----------------------����ȫ������--------------------
[pop.Gbest_Value(t_num+1),temp_Index] = param.fun([pop.Value;pop.Gbest_Value(t_num)]);
temp_Var = [pop.Var;pop.Gbest_Var(t_num,:)];
pop.Gbest_Var(t_num+1,:) = temp_Var(temp_Index,:);