data = load('spam.mat');
xtrain = data.Xtrain;
ytrain = double(data.ytrain);
xmean = repmat(mean(xtrain), 3450, 1);
xstd = repmat(std(xtrain), 3450, 1);

x_norm1 = (xtrain - xmean)./(xstd);
x_norm2 = log(xtrain + 0.1);
x_norm3 = xtrain > 0;
lambda = 1;


%%
%batch
alpha = .000001;
beta = zeros(57,1);
mu = zeros(3450, 1);
nlls1 = zeros(1,10000);
nlls2 = zeros(1,10000);
nlls3 = zeros(1,10000);
for iter = 1:10000,
  mu = 1 ./ (1 + exp(-(x_norm1 * beta)));
  gradient = (2 * lambda * beta) - transpose(x_norm1) * (ytrain - mu);
  beta = beta - alpha * gradient;
  summation = sum(ytrain .* log(mu) + (1 - ytrain) .* log(1-mu));
  nlls1(iter) = lambda * norm(beta)^2 - summation;
end
beta = zeros(57,1);
mu = zeros(3450, 1);
for iter = 1:10000,
  mu = 1 ./ (1 + exp(-(x_norm2 * beta)));
  gradient = (2 * lambda * beta) - transpose(x_norm2) * (ytrain - mu);
  beta = beta - alpha * gradient;
  summation = sum(ytrain .* log(mu) + (1 - ytrain) .* log(1-mu));
  nlls2(iter) = lambda * norm(beta)^2 - summation;
end
beta = zeros(57,1);
mu = zeros(3450, 1);
for iter = 1:10000,
  mu = 1 ./ (1 + exp(-(x_norm3 * beta)));
  gradient = (2 * lambda * beta) - transpose(x_norm3) * (ytrain - mu);
  beta = beta - alpha * gradient;
  summation = sum(ytrain .* log(mu) + (1 - ytrain) .* log(1-mu));
  nlls3(iter) = lambda * norm(beta)^2 - summation;
end
hold on
plot(stoc_nlls1,'r')
plot(stoc_nlls2,'g')
plot(stoc_nlls3,'b')
legend('norm 1','norm 2','norm 3')
xlabel('Iterations')
ylabel('Negative Log Likelihood')
%%
% stochastic
alpha = .001;
beta1 = zeros(57,1);
beta2 = zeros(57,1);
beta3 = zeros(57,1);
stoc_nlls1 = zeros(1,100);
stoc_nlls2 = zeros(1,100);
stoc_nlls3 = zeros(1,100);
for loops = 1:5,
  for email = 1:3450,
    mu1 = 1 / (1 + exp(-(x_norm1(email,:) * beta1)));
    mu2 = 1 / (1 + exp(-(x_norm2(email,:) * beta2)));
    mu3 = 1 / (1 + exp(-(x_norm3(email,:) * beta3)));
    
    gradient1 = (2 * lambda * beta1) - transpose(x_norm1(email,:)) * (ytrain(email) - mu1);
    gradient2 = (2 * lambda * beta2) - transpose(x_norm2(email,:)) * (ytrain(email) - mu2);
    gradient3 = (2 * lambda * beta3) - transpose(x_norm3(email,:)) * (ytrain(email) - mu3);
    
    beta1 = beta1 - alpha * gradient1;
    beta2 = beta2 - alpha * gradient2;
    beta3 = beta3 - alpha * gradient3;
    
    new_mu1 = 1 ./ (1 + exp(-(x_norm1 * beta1)));
    new_mu2 = 1 ./ (1 + exp(-(x_norm2 * beta2)));
    new_mu3 = 1 ./ (1 + exp(-(x_norm3 * beta3)));
    
    summation1 = sum(ytrain .* log(new_mu1) + (1 - ytrain) .* log(1-new_mu1));
    summation2 = sum(ytrain .* log(new_mu2) + (1 - ytrain) .* log(1-new_mu2));
    summation3 = sum(ytrain .* log(new_mu3) + (1 - ytrain) .* log(1-new_mu3));
    
    stoc_nlls1(email + 3450*(loops-1)) = lambda * norm(beta1)^2 - summation1;
    stoc_nlls2(email + 3450*(loops-1)) = lambda * norm(beta2)^2 - summation2;
    stoc_nlls3(email + 3450*(loops-1)) = lambda * norm(beta3)^2 - summation3;
  end
end
%%
% stochastic alpha
alpha = .0001;
beta1 = zeros(57,1);
beta2 = zeros(57,1);
beta3 = zeros(57,1);
astoc_nlls1 = zeros(1,100);
astoc_nlls2 = zeros(1,100);
astoc_nlls3 = zeros(1,100);
for loops = 1:25,
  for email = 1:3450,
    mu1 = 1 / (1 + exp(-(x_norm1(email,:) * beta1)));
    mu2 = 1 / (1 + exp(-(x_norm2(email,:) * beta2)));
    mu3 = 1 / (1 + exp(-(x_norm3(email,:) * beta3)));
    
    gradient1 = (2 * lambda * beta1) - transpose(x_norm1(email,:)) * (ytrain(email) - mu1);
    gradient2 = (2 * lambda * beta2) - transpose(x_norm2(email,:)) * (ytrain(email) - mu2);
    gradient3 = (2 * lambda * beta3) - transpose(x_norm3(email,:)) * (ytrain(email) - mu3);
    
    beta1 = beta1 - (alpha / (email + 3450*(loops-1)) ) * gradient1;
    beta2 = beta2 - (alpha / (email + 3450*(loops-1)) ) * gradient2;
    beta3 = beta3 - (alpha / (email + 3450*(loops-1)) ) * gradient3;
    
    new_mu1 = 1 ./ (1 + exp(-(x_norm1 * beta1)));
    new_mu2 = 1 ./ (1 + exp(-(x_norm2 * beta2)));
    new_mu3 = 1 ./ (1 + exp(-(x_norm3 * beta3)));
    
    summation1 = sum(ytrain .* log(new_mu1) + (1 - ytrain) .* log(1-new_mu1));
    summation2 = sum(ytrain .* log(new_mu2) + (1 - ytrain) .* log(1-new_mu2));
    summation3 = sum(ytrain .* log(new_mu3) + (1 - ytrain) .* log(1-new_mu3));
    
    astoc_nlls1(email + 3450*(loops-1)) = lambda * norm(beta1)^2 - summation1;
    astoc_nlls2(email + 3450*(loops-1)) = lambda * norm(beta2)^2 - summation2;
    astoc_nlls3(email + 3450*(loops-1)) = lambda * norm(beta3)^2 - summation3;
  end
end
%%

x1 = [1:690];
x2 = [691:1380];
x3 = [1381:2070];
x4 = [2071:2760];
x5 = [2761:3450];

held1 = cat(2,x2,x3,x4,x5);
held2 = cat(2,x1,x3,x4,x5);
held3 = cat(2,x1,x2,x4,x5);
held4 = cat(2,x1,x2,x3,x5);
held5 = cat(2,x1,x2,x3,x4);

subset = cat(1,held1,held2,held3,held4,held5);
validation = cat(1,x1,x2,x3,x4,x5);

nlls = zeros(1,5)


lambda = 1;
alpha = .0000001;

for i = 1:5,
  iter_training = subset(i,:);
  iter_validation = validation(i,:);
  training_labels = ytrain(iter_training);
  training_emails = xtrain(iter_training,:);
  validation_labels = ytrain(iter_validation);
  validation_emails = xtrain(iter_validation,:);
  
  beta = zeros(57,1);
  mu = zeros(2760, 1);
   
  for iter = 1:1000,
    for num = 1:2760,
      mu(num) = 1 / (1 + exp(-( training_emails(num,:) * beta)));
    end

    gradient = (2 * lambda * beta) - transpose(training_emails) * (training_labels - mu);
    beta = beta - alpha * gradient;
  end
  
  mu = zeros(690, 1);
    for num = 1:690,
      mu(num) = 1 / (1 + exp(-( validation_emails(num,:) * beta)));
    end
  
  summation = sum(validation_labels .* log(mu) + (1 - validation_labels) .* log(1-mu));
  nlls(i) = lambda * norm(beta)^2 - summation;
end

mean(nlls)



















