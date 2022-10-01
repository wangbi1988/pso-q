%% DE Algorithm

%%   ���������
clear
clc
%% ----------------�����Ż�Ŀ��-------------
param.Test_function = 'Schaffer'; % ���Ż��������� Schaffer Ackley
param = Set_function_Var(param); % ���Ż���������
CostFunction = @(x,y_type)Optimal_function(x,y_type); % ���Ż���������
%% ---------------�����㷨����--------------
param = Set_Var(param);
%% ------------------��ʼ��-----------------
pop = Code_Population(param,CostFunction);

%% ---------------------����Ѱ��-------------
for t_num = 1:param.Maxgen
    % ------------------����---------------------
    var_Mu = Count_Mutate(pop,param,t_num);
    % ------------------����---------------------
    var_Cr = Count_Cross(pop,param,var_Mu);
    % ------------------ѡ��---------------------
    pop = Count_Select(pop,param,var_Cr,CostFunction,t_num);
end
%% -----------��ͼ---------------
figure(1)
plot(1:param.Maxgen,pop.Gbest_Value(2:end),'r-o','LineWidth',2)
xlabel('��������')
ylabel('��Ӧ�Ⱥ���ֵ')
axis tight
grid on