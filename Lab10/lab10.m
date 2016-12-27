close all
clear all
clc

load('arrhythmia.mat');
rows_arrythmic = find(arrhythmia(:,end)>2);
arrhythmia(rows_arrythmic, end) = 2; 

arrhythmia(:, find(sum(abs(arrhythmia)) == 0)) = []; %remove null columns

class_id = arrhythmia(:,end);
class_id_((class_id==2),:) = 1;
class_id_((class_id==1),:) = -1;

y = arrhythmia(:,1: end-1);
y_withclass = [y class_id_];

[N,F]=size(y);
mean_y = mean(y);
std_y = std(y);
o=ones(N,1);% o is a column vector 
y=(y-o*mean_y)./(o*std_y);% y is normalized


Mdl=fitcsvm(y,class_id_,'BoxConstraint',5.0,'KernelFunction','linear'); 
classhat=sign(y*Mdl.Beta+Mdl.Bias); % segno del valore della funzione: se 0 siamo sull'iperpiano, >0 a dx, <0 a sx...


CVMdl = crossval(Mdl);
classLoss = kfoldLoss(CVMdl);