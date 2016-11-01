close all
clear all
clc

load('arrhythmia.mat');
rows_arrythmic = find(arrhythmia(:,end)>1);
arrhythmia(rows_arrythmic, end) = 2; 
j=1;
for i = 1:size(arrhythmia,2)
    if (unique(arrhythmia(:,i))==0)
       col_del(j) = i;
       j = j+1;
    end
end
col_del = sort(col_del, 'descend');
for i = col_del
    arrhythmia(:,i) = [];
end

class_id = arrhythmia(:,end);
y = arrhythmia(:,1: end-1);
ind_sani = find(class_id<2)';
ind_malati = find(class_id>1)';
y1= y(ind_sani, :);
y2= y(ind_malati, :);

x1 = mean(y1);
x2 = mean(y2);

xmeans=[x1;x2]; % matrix with x1 and x2
eny=diag(y*y'); % |y(n)|^2
enx=diag(xmeans*xmeans'); % |x1|^2 and |x2|^2
dotprod = y*xmeans'; % matrix with the dot product between
% each y(n) and each x
[U,V]=meshgrid(enx,eny); 
dist2=U+V-2*dotprod; %|y(n)|^2+|x(k)|^2-2y(n)x(k)= 
% =|y(n)-x(k)|^2

[dummy, previsione] = min(dist2.');
previsione = previsione.';

verinegativi = 0;
falsipositivi = 0;
veripositivi = 0;
falsinegativi = 0;
arrhythmia_lastcol = arrhythmia(:,end);
for i = 1 : length(arrhythmia) 
    if (arrhythmia_lastcol(i) == 1)
        if (arrhythmia_lastcol(i) - previsione(i) == 0)
            verinegativi = verinegativi + 1;
        else 
            falsipositivi = falsipositivi + 1;
        end
    else
        if (arrhythmia_lastcol(i) - previsione(i) == 0) 
            veripositivi = veripositivi + 1;
        else 
            falsinegativi = falsinegativi + 1;
        end
    end
end

specificity = verinegativi / (verinegativi + falsipositivi); %true negative
sensitivity = veripositivi / (veripositivi + falsinegativi); % true positive
falsealarm = falsipositivi / (verinegativi + falsipositivi);
misseddetection = falsinegativi / (veripositivi + falsinegativi);


% before apply Bayes whiten the data, applying PCA

pi1 = size(y1,1)/ size(y,1);
pi2 = size(y2,1)/ size(y,1);

% perform PCR
N = size(y,1);
R = (1/N) * y.' * y;
[U, A] = eig(R);
% DINAMICA L
total_eig = sum(diag(A));
percentage_thresh = 0.97 * total_eig;

sum_diag = 0;
L = 1;
while sum_diag < percentage_thresh
    sum_diag = A(L,L) + sum_diag; 
    L = L+1;
end

U_L = U(:, 1:L);
Z = y * U_L;
z1 = Z(ind_sani,:);
z2 = Z(ind_malati,:);
w1 = mean(z1,1);
w2 = mean(z2,1);

