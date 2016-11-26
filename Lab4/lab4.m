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

meany = mean(y);
std_y = std(y);
for i= 1:size(y)
    y_norm(i,:) = y(i,:) - meany;
    y_norm(i,:) = y_norm(i,:)./std_y;
end
y =y_norm;

% Class 1 e 2

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
for i = 1 : length(class_id) 
    if (class_id(i) == 1)
        if (previsione(i) == 1)
            verinegativi = verinegativi + 1;
        else 
            falsipositivi = falsipositivi + 1;
        end
    else
        if (previsione(i) == 2)     
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

% perform PCA
meany = mean(y);
std_y = std(y);
for i= 1:size(y)
    y_norm(i,:) = y(i,:) - meany;
    y_norm(i,:) = y_norm(i,:)./std_y;
end

y =y_norm;
N = size(y,1);
R = (1/N) * y.' * y;
[U, A] = eig(R);
% DINAMICA L
total_eig = sum(diag(A));
percentage_thresh = 0.999999 * total_eig;

sum_diag = 0;
L = 1;
while sum_diag < percentage_thresh
    sum_diag = A(L,L) + sum_diag; 
    L = L+1;
end

U_L = U(:, 1:L);
A_L = A(1:L,1:L);

Z = y * U_L;

%Z_norm = (1 / sqrt(N)) * Z * A_L^(-1/2); % NORMALIZZAZIONE CHE DA PROBLEMI
for i= 1:size(Z)
Z_norm(i,:) = (Z(i,:) - mean(Z))./ std(Z);
end
Z = [];
Z = Z_norm;
Rz = (1/N) * Z.' * Z; 
z1 = Z(ind_sani,:);
z2 = Z(ind_malati,:);
w1 = mean(z1,1);
w2 = mean(z2,1);

wmeans=[w1;w2]; % matrix with x1 and x2
eny2=diag(Z*Z'); % |y(n)|^2
enx2=diag(wmeans*wmeans'); % |x1|^2 and |x2|^2
dotprod2 = Z*wmeans'; % matrix with the dot product between
% each y(n) and each x
[U2,V2]=meshgrid(enx2,eny2); 
dist2_b=U2+V2-2*dotprod2; %|y(n)|^2+|x(k)|^2-2y(n)x(k)= 
% =|y(n)-x(k)|^2
var_patients = var(Z);
dist2_b(:,1) = dist2_b(:,1) - (2 * log(pi1));
dist2_b(:,2) = dist2_b(:,2) - (2 * log(pi2));

[dummy, previsione_bayes] = min(dist2_b.');
previsione_bayes = previsione_bayes.';

verinegativi_b = 0;
falsipositivi_b = 0;
veripositivi_b = 0;
falsinegativi_b = 0;
for i = 1 : length(class_id) 
    if (class_id(i) == 1)
        if (previsione_bayes(i) == 1)
            verinegativi_b = verinegativi_b + 1;
        else 
            falsipositivi_b = falsipositivi_b + 1;
        end
    else
        if (previsione_bayes(i) == 2) 
            veripositivi_b = veripositivi_b + 1;
        else 
            falsinegativi_b = falsinegativi_b + 1;
        end
    end
end

specificity_b = verinegativi_b / (verinegativi_b + falsipositivi_b); % true negative
sensitivity_b = veripositivi_b / (veripositivi_b + falsinegativi_b); % true positive
falsealarm_b = falsipositivi_b / (verinegativi_b + falsipositivi_b); 
misseddetection_b = falsinegativi_b / (veripositivi_b + falsinegativi_b);

%%%%%% risolvi con formula di normalizzazione 
%%%%% prova anche togliendo le feature/pazienti con diversi criteri (es.
%%%%% una colonna che ? tutta nulla tranne in un punto) => var = 0

