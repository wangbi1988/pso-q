function y = Optimal_function(x,Test_function)

switch Test_function
    case 'Schaffer'
        temp = sum(x.^2);
        y = (sin(sqrt(temp)))^2-0.5;
        y = 0.5+y/((1+0.001*temp)^2);
    case 'Ackley'
        x_num = 1/numel(x);
        y = -20*exp(-0.2*sqrt(x_num*sum(x.^2)));
        y = y-exp(x_num*sum(2*pi*x))+20+exp(1);
        y = -y;
end