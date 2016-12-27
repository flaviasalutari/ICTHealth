close all
clear all
clc

load('updrs.mat')
totalpatients = parkinsonsupdrs(size(parkinsonsupdrs,1),1);
matricepazienti = zeros(1,22);
for k = 1:totalpatients
    patient_matrix = parkinsonsupdrs(find(parkinsonsupdrs(:,1)==k),:);
    patient_matrix(:,4) = abs(fix(patient_matrix(:,4)));
    days_patient = unique(patient_matrix(:,4));
    for i = 1:length(days_patient)
        day = days_patient(i);
        indexes_day = find(patient_matrix(:,4)==day);
        new_matrix = mean(patient_matrix(indexes_day,:),1);
        matricepazienti = [matricepazienti;new_matrix];
    end
end 

matricepazienti = matricepazienti(2:end,:);
matricepazienti_norm = (matricepazienti - (ones(size(matricepazienti,1),1)*mean(matricepazienti))) ./ sqrt(ones(size(matricepazienti,1),1)*var(matricepazienti));

%%% Perform regression [1] 

data_train = matricepazienti(matricepazienti(:,1)<37,:);
data_test = matricepazienti(matricepazienti(:,1)>36,:);

m_data_train=mean(data_train);
v_data_train=var(data_train);
o = ones(size(data_train,1),1);
data_train_norm = data_train;
data_train_norm(:,5:end) = (data_train(:,5:end) - o*m_data_train(:,5:end)) ./ sqrt(o*v_data_train(:,5:end));

mean_train_norm = mean(data_train_norm,1); % verify it is zero mean and va = 1?? 
var_train_norm = var(data_train_norm,1); 

o = ones(size(data_test,1),1);
data_test_norm = data_test;
data_test_norm(:,5:end) = (data_test(:,5:end) - o*m_data_train(:,5:end)) ./ sqrt(o*v_data_train(:,5:end));

mean_test_norm = mean(data_train_norm,1); % verify it is zero mean and va = 1?? 
var_test_norm = var(data_train_norm,1);

% Perform regression [2]
F0 = 5;
y_train=data_train_norm(:,F0); %% feature che elimino e poi vorr? stimare
X_train=data_train_norm;
X_train(:,F0)=[];  %% tutte le feature senza F0 

y_test=data_test_norm(:,F0); 
X_test=data_test_norm;
X_test(:,F0)=[];
%%
%%% PCR
N = size(X_train,1);
R = (1/N) * X_train(:,5:end).' * X_train(:,5:end);
[U, A] = eig(R);
Z = X_train(:, 5:end) * U;

Z_norm = (1 / sqrt(N)) * Z * A^(-1/2);
Z_y = Z_norm.' * y_train;
y_hat = Z_norm * Z_y;
% a = inv(X_train*X_train.') * X_train * y_train;
a = (1/N) * U * inv(A) * U.' * X_train(:,5:end).' * y_train;
y_hat_2_train = X_train(:,5:end) * a; %%% ?????????????????????

figure
plot(y_hat_2_train, '-y')
hold on
plot(y_train)
grid on
title('y train')

% test

y_hat_test = X_test(:,5:end) * a;

figure
plot(y_hat_test, '-y')
hold on
plot(y_test)
grid on
title('y test')


%%%%%%%%%%%% L %%%%%%%%%%

K = 1;
total_eig = sum(diag(A));
percentage_thresh = 0.98 * total_eig;

sum_diag = 0;
L = 1;
while sum_diag < percentage_thresh
    sum_diag = A(L,L) + sum_diag; 
    L = L+1;
end
%precedente U = 5

U_L = U(:,K:L);
A_L = A(K:L,K:L);

Z_norm_L = 1/sqrt(N) * X_train(:,5:end) * U_L * A_L ^(-1/2);

Z_y_L = Z_norm_L.' * y_train;

y_hat_L = Z_norm_L * Z_y_L;

a_hat_L = 1/N * U_L * inv(A_L) * U_L.' * X_train(:,5:end).' *y_hat_L;

stima_L = X_train(:, 5:end) * a_hat_L;
stima_L_2 = X_test(:, 5:end) * a_hat_L;

errore = norm(stima_L_2-y_test);

figure
plot(stima_L)
hold on
plot(y_train)
grid on
title('With L, train')

figure
plot(stima_L_2)
hold on
plot(y_test)
grid on
title('With L, test')