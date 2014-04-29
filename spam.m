function [out] = spam(in)

%p4_1

data1 = preprocessing(load('spam.mat'), 1);
data2 = preprocessing(load('spam.mat'), 2);
data3 = preprocessing(load('spam.mat'), 3);

error = x_val(data2, 10, 10^-4, .07)

[new_beta,NLL_GD_1] = grad_descent(double(data1.ytrain),double(data1.Xtrain), 10^-5 , .65);
[new_beta,NLL_GD_2] = grad_descent(double(data2.ytrain),double(data2.Xtrain), 10^-5 , .65);
[new_beta,NLL_GD_3] = grad_descent(double(data3.ytrain),double(data3.Xtrain), 10^-5 , .65);

figure(1);
hold on
plot(NLL_GD_1,'r');
plot(NLL_GD_2,'b');
plot(NLL_GD_3,'g');
legend('Standardized', 'log(x_ij + .1)', 'Binarized');
hold off

[new_beta,NLL_SGD_1] = sgd(double(data1.ytrain),double(data1.Xtrain), 10^-3, 1);
[new_beta,NLL_SGD_2] = sgd(double(data2.ytrain),double(data2.Xtrain), 10^-3, 1);
[new_beta,NLL_SGD_3] = sgd(double(data3.ytrain),double(data3.Xtrain), 10^-3, 1);

figure(2);
hold on
plot(NLL_SGD_1,'r');
plot(NLL_SGD_2,'b');
plot(NLL_SGD_3,'g');
axis([0,3500,2300,2550])
legend('Standardized', 'log(x_ij + .1)', 'Binarized');
hold off



%%%%%%Experimenting with the test set%%%%%%%%%
% lambda = [.65, 5];
% data = preprocessing(load('spam.mat'), 2);
% y = zeros(size(data.Xtest,1));
% for i = lambda
% [new_beta,NLL_GD] = grad_descent(double(data.ytrain),double(data.Xtrain), 10^-6 , i);
% y = [y, round((1 ./ (1 + exp(-new_beta * data.Xtest'))))'];
% end
% y(:,1) = [];
% sum(y(:,1) == y(:,2))

%%%%%%%CLASSIFYING THE TEST SET, TO BE SUBMITTED ON KAGGLE%%%%%%%%

data = preprocessing(load('spam.mat'), 2);
[new_beta,NLL_GD] = grad_descent(double(data.ytrain),double(data.Xtrain), 10^-4, .07);
y = round((1 ./ (1 + exp(-new_beta * data.Xtest'))))';
sum(y)
id = [1:size(data.Xtest,1)]';
y = [id, y];
csvwrite('thomas_labels.csv', y);

end

function [beta_est, NLL] = sgd(y,x,eta,lambda)
beta = zeros(1,57);
mu = 0;
grad = x' * (mu - y);
NLL = zeros(1,1);
beta_new = beta;
orig_eta = eta;
for i = 1: size(x,1)
    eta = orig_eta / i;
    mu = 1.0 / (1 + exp(-beta_new * x(i,:)'));
    grad = 2 * lambda * beta_new' - x(i,:)' * (y(i) - mu);
    beta_new = beta_new - eta * grad';
    mu_2 = (1 ./ (1 + exp(-beta_new * x')))';
    NLL(end+1) = lambda * norm(beta_new)^2 - sum(y' * log(mu_2) + (1 - y') * log(1 - mu_2));
end

grad;
beta_est = beta_new;
end
function [avg_error] = x_val(data,k, eta, lambda)

errors = zeros(1,1);
data.Xtrain = double(data.Xtrain);
data.ytrain = double(data.ytrain);

%permute the data, so chunking of data is random
a = randperm(size(data.Xtrain,1));

x_train = double(data.Xtrain);
x_valid = double(data.Xtrain);
y_train = double(data.ytrain);
y_valid = double(data.ytrain);
chunk_size = size(data.Xtrain,1) / k;

%k-fold cross validation
for i = 1:k
    
    x_train = double(data.Xtrain);
    x_train(a(i * chunk_size + 1 - chunk_size:i * chunk_size),:) = [];
    y_train = double(data.ytrain);
    y_train(a(i * chunk_size + 1 - chunk_size:i * chunk_size)) = [];
    
    x_valid = double(data.Xtrain(a(i * chunk_size + 1 - chunk_size:i * chunk_size),:));
    y_valid = double(data.ytrain(a(i * chunk_size + 1 - chunk_size:i * chunk_size),:));
    
    [new_beta,NLL] = grad_descent(y_train,x_train, eta , lambda);
    y = round((1 ./ (1 + exp(-new_beta * x_valid'))))';
    errors(end+1) = 1 - sum((y == y_valid)) / size(y_valid,1);
end

avg_error = mean(errors);

end
function [beta_est, NLL] = grad_descent(y, x, eta, lambda)

beta = zeros(1,57);
mu = (1 ./ (1 + exp(-beta * x')))';
grad = x' * (mu - y);
NLL = zeros(1,1);


beta_new = beta;
for i = 1:30000
    mu = (1 ./ (1 + exp(-beta_new * x')))';
    grad = 2 * lambda * beta_new'  - x' * (y - mu);
    beta_new = beta_new - eta * grad';
    NLL(end+1) = lambda * norm(beta_new)^2 - sum(y' * log(mu) + (1 - y') * log(1 - mu));
end

beta_est = beta_new;
end
function [data_out] = preprocessing(data, method)
data_out = data;
switch method
    case 1
        means = repmat(mean(data.Xtrain),size(data.Xtrain,1),1);
        std_dev = repmat(std(data.Xtrain), size(data.Xtrain,1),1);
        data.Xtrain = data.Xtrain - means;
        data.Xtrain = data.Xtrain ./ std_dev;
        
        means = repmat(mean(data.Xtest),size(data.Xtest,1),1);
        std_dev = repmat(std(data.Xtest), size(data.Xtest,1),1);
        data.Xtest = data.Xtest - means;
        data.Xtest = data.Xtest ./ std_dev;
    case 2
        data.Xtrain = log(data.Xtrain + .1);
        data.Xtest = log(data.Xtest + .1);
    case 3
        data.Xtrain = data.Xtrain > 0;
        data.Xtest = data.Xtest > 0;
end
data_out = data;
end
function [out] = p4_1(in)

X =[0     3     1;1     3     1;0     1     1;1     1     1];
Y = [1;1;0;0];
B_0 = [-2;1;0];
lambda = .07;

mu_0 = 1 ./ ( 1 + exp(- X * B_0))
H_0 = 2 * lambda * eye(3) + X' * (diag((1-mu_0) .* mu_0) * X)
B_1 = B_0 - inv(H_0) * (2 * lambda * B_0 - X' * (Y - mu_0))

mu_1 = 1 ./ ( 1 + exp(- X * B_1))
H_1 = 2 * lambda * eye(3) + X' * (diag((1-mu_1) .* mu_1) * X) 
B_2 = B_1 - inv(H_1) * (2 * lambda * B_1 - X' * (Y - mu_1))


x = linspace(-10,10);

y_0 = polyval(B_0, x);
y_1 = polyval(B_1, x);
y_2 = polyval(B_2, x);

hold on
axis([-10,10,-10,10])
plot(x,y_0)
plot(x,y_1,'r')
plot(x,y_2,'g')
scatter(X(:,1), X(:,2));
legend('B_0', 'B_1', 'B_2')
hold off


end