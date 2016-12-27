close all
clear all
clc

load('a.mat')
keylist={'normal','abnormal','present','notpresent','yes','no', 'good','poor','ckd','notckd','?',''};
keymap=[0,1,0,1,0,1,0,1,2,1,NaN,NaN];
a = chronickidneydisease;
%% 

for kr = 1:size(a,1)
    for kc = 1:size(a,2)
        c=strtrim(a(kr,kc));
        check=strcmp(c,keylist);% check(i)=1 if c==keylist(i)
        if sum(check)==0
            b(kr,kc)=str2num(a{kr,kc});% from text to numeric
        else
            ii=find(check==1);
            b(kr,kc)=keymap(ii);% use the lists
        end;
    end
end

%% Perform clustering
b = b(:,1:end-1);% nell'import era stata selezionata una colonna in pi?
vera = b(:,end);
X_withlast=b;
x = b(:,1:end-1); 
[N,F]=size(x);
d = pdist(x);
tree = linkage(d);
c = cluster(tree,'maxclust',2);

figure
p=0;
dendrogram(tree,p);
figure
nn=[1:N];
plot(nn,c,'o'),grid on
xlabel('i')
ylabel('cluster for the i-th row of X')


%error probability
err_prob =sum(c ~= b(:,F))/N; 


%% Classification
tc = fitctree(x,vera); 
view(tc,'Mode','graph')
view(tc)

for i = 1:N
if x(i,15) < 13.05 
    if x(i,16)<44.5 
        previsione(i) = 2;
    else
        previsione(i) =1;
    end
else
    if x(i,3) < 1.0175
        previsione(i) = 2;
    else
        if x(i,4)<0.5
            previsione(i) = 1;
        else
            previsione(i) = 2;
        end
    end
    end
end
previsione = previsione';

err =(previsione ~= vera);
ber =sum(err)/length(err);



 