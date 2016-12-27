close all
clear all
clc

load('arrhythmia.mat');
iii=find(arrhythmia(:,end)>2); 
arrhythmia(iii,end)=2;

y=arrhythmia(:,1:end-1); 
class_id=arrhythmia(:,end); 
[N,F]=size(y);
ymean=mean(y);% ymean is a row vector 
yvar=var(y);% yvar is a row vector 
o=ones(N,1);% o is a column vector 
y=(y-o*ymean)./sqrt(o*yvar);% y is normalized

%% Class 1 e 2

ind_sani = find(class_id==1);
ind_malati = find(class_id==2);

y1= y(ind_sani, :);
y2= y(ind_malati, :);
x1 = mean(y1);
x2 = mean(y2);

xmeans=[x1;x2];
rhoy=y*xmeans'; 
en1=diag(y*y');
en2=diag(xmeans*xmeans');

[Uy,Vy]=meshgrid(en2,en1); 
disty=Uy+Vy-2*rhoy;




