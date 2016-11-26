close all
clear all
clc

load('arrhythmia.mat');

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
classes = unique(class_id); 
y = arrhythmia(:,1: end-1);

meany = mean(y);
std_y = std(y);
for i= 1:size(y)
    y_norm(i,:) = y(i,:) - meany;
    y_norm(i,:) = y_norm(i,:)./std_y;
end

for ii=1:16
    indexes = find(class_id == ii);
    x16(ii,:) = mean(y_norm(indexes,:),1);
    if length(indexes) ~= 0
        pi16(i) = size(indexes,1)/ size(y,1);
    else
        pi16(i) = 0;
    end
end
eny=diag(y_norm*y_norm'); 
enx=diag(x16*x16'); 
dotprod = y_norm*x16'; 
[U,V]=meshgrid(enx,eny); 
dist16 =U+V-2*dotprod;

truedetected = 0;
falsedetected = 0;
for i=1:length(dist16)
[dummy,pos] = min(dist16(i,:));
previsione(i,1) = pos;
    if pos == class_id(i)
        truedetected = truedetected + 1;
    else
        falsedetected = falsedetected + 1;
    end
end

truedetection = truedetected/length(dist16);
falsedetection = falsedetected/length(dist16);

%%% Bayes


y =y_norm;
N = size(y,1);
R = (1/N) * y.' * y;
[U, A] = eig(R);
% DINAMICA L
total_eig = sum(diag(A));
percentage_thresh = 0.99 * total_eig;

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
for i=1:length(classes)
    ii = classes(i);
    indexes = find(class_id == ii);
    w16(ii,:) = mean(Z(indexes,:),1);
end

eny2=diag(Z*Z'); 
enx2=diag(w16*w16');
dotprod2 = Z*w16';
[U2,V2]=meshgrid(enx2,eny2); 
dist16_b=U2+V2-2*dotprod2;

var_patients = var(Z);


truedetectedB = 0;
falsedetectedB = 0;
for k = classes
    if pi16(k) ~= 0
    dist16_b(:,k) = dist16_b(:,k) - (2 * log(pi16(k)));
    end
end

for i=1:length(dist16_b)
[dummy,pos] = min(dist16_b(i,:));
previsione(i,1) = pos;
    if pos == class_id(i)
        truedetectedB = truedetectedB + 1;
    else
        falsedetectedB = falsedetectedB + 1;
    end
end
truedetectionB = truedetectedB/length(dist16_b);
falsedetectionB = falsedetectedB/length(dist16_b);
                                                                            