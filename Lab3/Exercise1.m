clear all
close all
clc

% LEGEND:
% arr               --> original data
% classIdOriginal   --> Original column with patients classes
% y                 --> Feature matrix WITHOUT classes
% yNorm             --> Feature matrix WITHOUT classes
% pi16              --> Vector containing the prob related to each class
% xxxMDC            --> Variable related to MINIMUM DISTANCE CRITERION
% xxxBay            --> Variable related to BAYES CRITERION
% xxxMeans          --> Matrix ORDERED with the means of the features per each class

% ===================== MATRIX LOADING ======================
load('arrhythmia.mat')

% ===================== DATA PREPARING ====================
arr = arrhythmia;
nOfPatients = length(arr(:, 1));
classIdOriginal = arr(:, end);
for ii = 1:nOfPatients
    if arr(ii, end) > 1
        arr(ii, end) = 2;
    end
end
% Comando che toglie tutte le colonne che non contengono informazione:
arr(:, ~any(arr,1)) = [];
class_id = arr(:, end);     % Classe della malattia
y = arr(:, 1:(end - 1));    % Matrice di features SENZA la class_id
meanY = mean(y, 1);
varY = var(y, 1);
% NORMALIZATION of the features matrix
for aa = 1:nOfPatients
    yNorm(aa, :) = (y(aa, :) - meanY) ./ sqrt(varY); 
end

index1 = find(class_id == 1);   % Indici relativi alla class_id 1
index2 = find(class_id == 2);   % Indici relativi alla class_id 2
y1 = yNorm(index1, 1:end);    % Class 1 matrix, WITHOUT CLASSID
y2 = yNorm(index2, 1:end);    % Class 2 matrix, WITHOUT CLASSID
x1 = mean(y1, 1);   % Vettore riga che contiene la media di ogni colonna per classe 1
x2 = mean(y2, 1);   % Vettore riga che contiene la media di ogni colonna per classe 2

% Matrix with x1 and x2: first row contains the mean of the features of the first
% class, second row contains the mean of the features of the second class:
xMeans = [x1; x2];  

% ===================== MINIMUM DISTANCE CRITERION ===================
eny = diag(yNorm * yNorm');
enx = diag(xMeans * xMeans');
dotProd = yNorm * xMeans';
[U, V] = meshgrid(enx, eny);
distance = U + V - 2*dotProd;   % Matrice che contiene la distanza di ogni paziente da ogni classe
est_class_id = [];

for ii = 1:nOfPatients
    if distance(ii, 1) < distance(ii, 2)
        est_class_id(ii) = 1;
    else 
        est_class_id(ii) = 2;
    end 
end

% --------------------- Sensitivity & Specificity --------------------
truePositiveMDC = 0;
trueNegativeMDC = 0;
falsePositiveMDC = 0;
falseNegativeMDC = 0;
for ii = 1:nOfPatients
    if est_class_id(ii) == 1 && class_id(ii) == 1
        trueNegativeMDC = trueNegativeMDC + 1;
    elseif est_class_id(ii) == 2 && class_id(ii) == 2
        truePositiveMDC = truePositiveMDC + 1;
    elseif est_class_id(ii) == 2 && class_id(ii) == 1
        falsePositiveMDC = falsePositiveMDC + 1;
    elseif est_class_id(ii) == 1 && class_id(ii) == 2
        falseNegativeMDC = falseNegativeMDC + 1;
    end    
end
sensitivityMDC = truePositiveMDC / (truePositiveMDC + falseNegativeMDC);
falseNegativeProbMDC = 1 - sensitivityMDC;
specificityMDC = trueNegativeMDC / (trueNegativeMDC + falsePositiveMDC);
falsePositiveProbMDC = 1 - specificityMDC;

% ===================== BAYESIAN CRITERION =========================
pi1 = length(index1) / nOfPatients; % Prob that hypotesis 1 is correct
pi2 = length(index2) / nOfPatients; % Prob that hypotesis 2 is correct
R = (1/nOfPatients) * (yNorm') * (yNorm);   % Covariance matrix of the ORIGINAL FEATURES
[U, Lambda] = eig(R);               
Lambdas = diag(Lambda);
sommaLambdas = sum(Lambdas);
P = 0.99;
somm = 0;
ii = 0;
while somm < P * sommaLambdas
    ii = ii + 1;
    somm = somm + Lambdas(ii); 
end
UL = U(:, 1:ii);
z = yNorm * UL;     % z è la proiezione delle features originali ORTONORMALIZZATE
% Rz = (1/nOfPatients) * (z') * (z);    % The covariance matrix has to be
% DIAGONAL 
% Z è IL NUOVO VETTORE DI FEATURES ORTOGONALI TRA DI LORO, ma non
% ortonormali. Per renderli ortonormali bisogna RI-NORMALIZZARLE.
zMean = mean(z, 1);
zVar = var(z, 1);
for aa = 1:nOfPatients
    zNorm(aa, :) = (z(aa, :) - zMean) ./ sqrt(zVar);
end
w1 = mean(zNorm(index1, :), 1);
w2 = mean(zNorm(index2, :), 1);
% Probability COMPARISON
wMeans = [w1; w2];
% zVar = var(z, 1);   % Varianza di ogni feature nuova
eny = diag(zNorm * zNorm');
enx = diag(wMeans * wMeans');
dotProd = zNorm * wMeans';
[U, V] = meshgrid(enx, eny);
distanceBay = U + V - 2*dotProd;   % Matrice che contiene la distanza di ogni paziente da ogni classe
est_class_id_bay = [];

for ii = 1:nOfPatients
    if distanceBay(ii, 1) - (2 * log(pi1)) < distanceBay(ii, 2) - (2 * log(pi2))
        est_class_id_bay(ii) = 1;
    else 
        est_class_id_bay(ii) = 2;
    end 
end
% Ogni feature è caratterizzata da una varianza che è comune a tutte le
% classi.

% --------------------- Sensitivity & Specificity --------------------
truePositiveBay = 0;
trueNegativeBay = 0;
falsePositiveBay = 0;
falseNegativeBay = 0;
for ii = 1:nOfPatients
    if est_class_id_bay(ii) == 1 && class_id(ii) == 1
        trueNegativeBay = trueNegativeBay + 1;
    elseif est_class_id_bay(ii) == 2 && class_id(ii) == 2
        truePositiveBay = truePositiveBay + 1;
    elseif est_class_id_bay(ii) == 2 && class_id(ii) == 1
        falsePositiveBay = falsePositiveBay + 1;
    elseif est_class_id_bay(ii) == 1 && class_id(ii) == 2
        falseNegativeBay = falseNegativeBay + 1;
    end    
end
sensitivityBay = truePositiveBay / (truePositiveBay + falseNegativeBay);
falseNegativeProbBay = 1 - sensitivityBay;
specificityBay = trueNegativeBay / (trueNegativeBay + falsePositiveBay);
falsePositiveProbBay = 1 - specificityBay;

% ========================== 16 CLASSES =========================
% ===================== MINIMUM DISTANCE CRITERION =====================
for aa = 1:max(classIdOriginal)
    indici = find(classIdOriginal == aa);
    pi16(aa) = length(indici) / nOfPatients;
    xMean16(aa, :) = mean(yNorm(indici, :), 1);
end
eny16 = diag(yNorm * yNorm');
enx16 = diag(xMean16 * xMean16');
dotProd16 = yNorm * xMean16';
[U16, V16] = meshgrid(enx16, eny16);
distance16 = U16 + V16 - 2*dotProd16;

% Valutazione del minimum distance criterion: con true detection individuo
% tutte le occorrenze in cui la classe stimata è uguale a quella originale.
trueDetection = 0;
falseDetection = 0;
for j = 1:nOfPatients
    % pos è un vettore che contiene la posizione della minima distanza,
    % quindi il relativo indice è uguale alla classe stimata.
    [mini(j), pos(j)] = min(distance16(j, :));
    if pos(j) == classIdOriginal(j)
        trueDetection = trueDetection + 1;
    else
        falseDetection = falseDetection + 1;
    end
end
percTrueDetection = trueDetection / nOfPatients;
percFalseDetection = falseDetection / nOfPatients;
figure, plot(pos, '*'), hold on, grid on, plot(classIdOriginal, 'o'), legend('Prova', 'Class ID')
title(['Class detection MDC plot: true detection = ', num2str(percTrueDetection * 100), ' %'])

% ========================== 16 CLASSES =========================
% ========================= BAYESIAN CRITERION =====================
R16 = (1/nOfPatients) * (yNorm') * (yNorm);   % Covariance matrix of the ORIGINAL FEATURES
[U16, Lambda16] = eig(R16);               
Lambdas16 = diag(Lambda16);
sommaLambdas16 = sum(Lambdas16);
P16 = 0.99;
somm16 = 0;
ii16 = 0;
while somm16 < P16 * sommaLambdas16
    ii16 = ii16 + 1;
    somm16 = somm16 + Lambdas16(ii16); 
end
UL16 = U16(:, 1:ii16);
z16 = yNorm * UL16;
zMean16 = mean(z16, 1);
zVar16 = var(z16, 1);
for aa = 1:nOfPatients
    zNorm16(aa, :) = (z16(aa, :) - zMean16) ./ sqrt(zVar16);
end
% MATRIX SORTING
for aa = 1:max(classIdOriginal)
    indici16 = find(classIdOriginal == aa);
    xMean16Bay(aa, :) = mean(zNorm16(indici16, :));
end
eny16 = diag(zNorm16 * zNorm16');
enx16 = diag(xMean16Bay * xMean16Bay');
dotProd16 = zNorm16 * xMean16Bay';
[U16, V16] = meshgrid(enx16, eny16);
distanceBay16 = U16 + V16 - 2*dotProd16;
trueDetectionBay = 0;
falseDetectionBay = 0;
for aaa = 1:nOfPatients
    [mini, pos(aaa)] = min(distanceBay16(aaa, :));
    if pos(aaa) == classIdOriginal(aaa)
        trueDetectionBay = trueDetectionBay + 1;
    else
        falseDetectionBay = falseDetectionBay + 1;
    end
end
percTrueDetectionBay = trueDetectionBay / nOfPatients;
percFalseDetectionBay = falseDetectionBay / nOfPatients;
figure, plot(pos, '*'), hold on, grid on, plot(classIdOriginal, 'o'), legend('Prova', 'Class ID')
title(['Class detection BAYES plot: true detection = ', num2str(percTrueDetectionBay * 100), ' %'])


% ============================ HARD K-MEANS =========================
% Clustering alg starts with random  vectors x1 and x2 that we have found
% in classification.