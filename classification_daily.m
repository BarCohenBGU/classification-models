clear all;
close all;
rng('default');
filename='C:\Users\Bar\Documents\תואר\תואר שני\תזה\זמן אופטימלי\Data_27_10_2020_daily.xlsx';
opts = detectImportOptions(filename, 'PreserveVariableNames',true);
opts.VariableNamesRange = 'A1';
opts.DataRange = 'A2';
data = readtable(filename, opts,'ReadVariableNames',true);

group = data{:,3} == 3;
bygroup = data(group,:);

selectedData=bygroup(:,[1 6 7 18 20 21]);


% %no test
[response, Accuracy, Precision, Recall, F1, classificationM,validationPredictions,validationAccuracy, validationAccuracy2, partitionedModel,validationScores,trainedClassifier]=SVM(selectedData);

[X,Y,T,AUC,OPTROCPT] = perfcurve(response,validationScores(:,2),1);
figure(2)
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')