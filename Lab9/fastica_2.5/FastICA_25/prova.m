clear all
close all
%%
T=1;% duration of the simulation (s)
Ns=1000;%number of samples in the simulation
dt=T/Ns;% sampling interval
time=[0:dt:T];% time axis
% sinusoidal signal
f0=2;% frequency of the sinusoid
s1=sqrt(2)*cos(2*pi*f0*time);
%s1=sign(sin(2*pi*f0*time+pi/10));
% sawtooth signal
s2=mod(5*time,T)-T/2;
% Gaussian noise
s3=randn(1,Ns+1);
% weights in a 2 rows by 3 columns matrix A1, which must be found by ICA
A1=[0.5,0.1,0;-0.3,0.1,-0.01];
% x is a mtrix: 1st row is the 1st signal, 2nd row is the 2nd signal
x=A1*[s1;s2;s3];
figure
plot(time,x),grid on
title('signals x(1,:) and x(2,:)')
xlabel('t')
%% run Fastica (note that we have 2 signals x(1,:) and x(2,:), so we can at most find 2 indep. comp.
[Out,A,W]=fastica(x,'numOfIC',2);
figure
subplot(2,2,1),plot(time,s1),grid on,title('True components')
subplot(2,2,2),plot(time,s2),grid on
subplot(2,2,3),plot(time,Out(1,:)),grid on, title('ICA components')
subplot(2,2,4),plot(time,Out(2,:)),grid on
figure
subplot(2,2,1), hist(s1,40),title('Histograms true components')
subplot(2,2,2), hist(s2,40)
subplot(2,2,3), hist(x(1,:),40),title('Histograms of x')
subplot(2,2,4), hist(x(2,:),40)
%% compare A and A1
display('Original matrix A')
A1
display('ICA matrix A')
A


