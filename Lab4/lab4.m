close all
clear all
clc

load('arrhythmia.mat');
rows_arrythmic = find(arrhythmia(:,end)>2);
arrhythmia(rows_arrythmic, end) = 2; 

arrhythmia(:, find(sum(abs(arrhythmia)) == 0)) = []; %remove null columns

class_id = arrhythmia(:,end);
y = arrhythmia(:,1: end-1);
[N,F]=size(y);
mean_y = mean(y);
std_y = std(y);
o=ones(N,1);% o is a column vector 
y=(y-o*mean_y)./(o*std_y);% y is normalized

%% Apply Classification

ind_sani = find(class_id==1)';
ind_malati = find(class_id==2)';
y1= y(ind_sani, :);
y2= y(ind_malati, :);
x1 = mean(y1);
x2 = mean(y2);

xmeans = [x1;x2]; % matrix with x1 and x2
eny = diag(y*y'); % |y(n)|^2
enx = diag(xmeans*xmeans'); % |x1|^2 and |x2|^2
rhoy = y*xmeans'; % matrix with the dot product between% each y(n) and each x
[U,V] = meshgrid(enx,eny); 
dist2 = U+V-2*rhoy; %|y(n)|^2+|x(k)|^2-2y(n)x(k) =|y(n)-x(k)|^2
% [dummy, xk] = min(dist2');

%% Hard K-Means Algorithm

K = 2;
dist_2 = dist2;
var_k = ones(1,K);
pi_k =  1/K*ones(1,K);
% xk_norm = (xk - mean(xk))./std(xk)
patient_cluster = zeros(N,1);

for i= 1:9
    R_k = (pi_k(1)/((2*pi*var_k(1))^(N/2)))*exp(-dist_2(:,1)/(2*var_k(1)));
    R_j = (pi_k(2)/((2*pi*var_k(2))^(N/2)))*exp(-dist_2(:,2)/(2*var_k(2)));

    patient_cluster(find(R_k<R_j),1)=1;
    patient_cluster(find(R_k>=R_j),1)=2;
    
    w_1 = y(patient_cluster==1,:);
    w_2 = y(patient_cluster==2,:);
    pi_k(:,1) = length(w_1) / N;
    pi_k(:,2) = length(w_2) / N;
    x_k(1,:) = mean(w_1,1);
    x_k(2,:) = mean(w_2,1);

    var_k(:,1) = sum(norm(bsxfun(@minus,w_1,x_k(1,:))).^2) /((length(w_1) - 1)*F);
    var_k(:,2) = sum(norm(bsxfun(@minus,w_2,x_k(2,:))).^2) /((length(w_2) - 1)*F);
    
    dist_2(:,1) = norm(bsxfun(@minus,y,x_k(1,:))).^2;
    dist_2(:,2) = norm(bsxfun(@minus,y,x_k(2,:))).^2;
 
end


[dummy,previsione]=min(dist2,[],2);
N1=sum(class_id==1);
N2=sum(class_id==2);
false_positive=sum((previsione==2)&(class_id==1))/N1;
true_positive=sum((previsione==2)&(class_id==2))/N2; 
false_negative=sum((previsione==1)&(class_id==2))/N2;
true_negative=sum((previsione==1)&(class_id==1))/N1;


false_positive_c=sum((patient_cluster==2)&(class_id==1))/N1; 
true_positive_c=sum((patient_cluster==2)&(class_id==2))/N2; 
false_negative_c=sum((patient_cluster==1)&(class_id==2))/N2; 
true_negative_c=sum((patient_cluster==1)&(class_id==1))/N1;



%% Start from random initial vectors x_k 


K = 2;
var_k = ones(1,K);
pi_k =  1/K*ones(1,K);
% xk_norm = (xk - mean(xk))./std(xk)
patient_cluster = zeros(N,1);

x_k = rand(K,F);

dist__2(:,1) = norm(bsxfun(@minus,y,x_k(1,:))).^2;
dist__2(:,2) = norm(bsxfun(@minus,y,x_k(2,:))).^2;
 
for i= 1:990
    R_k = (pi_k(1)/((2*pi*var_k(1))^(N/2)))*exp(-dist__2(:,1)/(2*var_k(1)));
    R_j = (pi_k(2)/((2*pi*var_k(2))^(N/2)))*exp(-dist__2(:,2)/(2*var_k(2)));

    patient_cluster(find(R_k<R_j),1)=1;
    patient_cluster(find(R_k>=R_j),1)=2;
    
    w_1 = y(patient_cluster==1,:);
    w_2 = y(patient_cluster==2,:);
    pi_k(:,1) = length(w_1) / N;
    pi_k(:,2) = length(w_2) / N;
    x_k(1,:) = mean(w_1,1);
    x_k(2,:) = mean(w_2,1);

    var_k(:,1) = sum(norm(bsxfun(@minus,w_1,x_k(1,:))).^2) /((length(w_1) - 1)*F);
    var_k(:,2) = sum(norm(bsxfun(@minus,w_2,x_k(2,:))).^2) /((length(w_2) - 1)*F);

    dist__2(:,1) = norm(bsxfun(@minus,y,x_k(1,:))).^2;
    dist__2(:,2) = norm(bsxfun(@minus,y,x_k(2,:))).^2;

end




false_positive_c_INIT=sum((patient_cluster==2)&(class_id==1))/N1; 
true_positive_c_INIT=sum((patient_cluster==2)&(class_id==2))/N2; 
false_negative_c_INIT=sum((patient_cluster==1)&(class_id==2))/N2; 
true_negative_c_INIT=sum((patient_cluster==1)&(class_id==1))/N1;