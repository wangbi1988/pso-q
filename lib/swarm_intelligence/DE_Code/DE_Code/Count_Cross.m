function var_Cr = Count_Cross(pop,param,var_Mu)

var_Cr = pop.Var;
p_flag = rand(param.Size,param.Dim)<param.Cp; % �Ƿ񽻲� ��1:yes 0:no)
var_Cr(p_flag) = var_Mu(p_flag);  % ��������¸���