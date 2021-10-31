clear all;
close all;
rng('default');
filename='C:\Users\Bar\Documents\תואר\תואר שני\תזה\זמן אופטימלי\round3 hours.xlsx';
opts = detectImportOptions(filename, 'PreserveVariableNames',true);
opts.VariableNamesRange = 'A1';
opts.DataRange = 'A2';
data = readtable(filename, opts,'ReadVariableNames',true);

selectedData=data(:,[1 8 12 13 17 20 24]);
%selectedData=data(:,[1 8 9 20 22 24]);
%selectedData=data(:,[1 6 7 18 20 21]);
% 
% rng(123);
% PD = 1-0.2355;
% %PD = 1-0.086;
% cv = cvpartition(size(selectedData,1),'HoldOut',PD);
% Ptrain = selectedData(cv.training,:);


%no test
[response, Accuracy, Precision, Recall, F1, classificationM,validationPredictions,validationAccuracy, validationAccuracy2, partitionedModel,validationScores,trainedClassifier]=SVM_hours(selectedData);
% set1=training(partitionedModel.Partition,1);
% set2=training(partitionedModel.Partition,2);
% set3=training(partitionedModel.Partition,3);
% set4=training(partitionedModel.Partition,4);
% set5=training(partitionedModel.Partition,5);
% data=[set1,set2,set3,set4,set5];
[X,Y,T,AUC,OPTROCPT] = perfcurve(response,validationScores(:,2),1);
figure(2)
 plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')
%table= table(validationPredictions);
%results= [trainingData  table];



% rescale data
% features=data(:,[8 9 20 22 24]);
% respons=data(:,1);
% x=table2array(features);
% colmin = min(x);
% colmax = max(x);
% rescale_Data = rescale(x,'InputMin',colmin,'InputMax',colmax);
% rescaled_Data=array2table(rescale_Data,'VariableNames',{'MTD', 'STD', 'perc90-Tair', 'CWSI2', 'Cv'});
% selectedData = [respons rescaled_Data];

%Nr = normalize(selectedData,'range');





% % ordinal
% [classificationSVM, validationPredictions, trainingData]=SVM_ordinal(selectedData);
% table= table(validationPredictions);
% results= [trainingData  table];




% realY=data(:,1);
% realY = table2array(realY);
%[trainedClassifier, validationAccuracy]=trainClassifierLOGISTIC(cutdata1);
%imp = predictorImportance(classificationTree);
%imp
%[imp,ind]=sort(imp,"descend");
%ylabel('Estimates');
%xlabel('Predictors');
%h2 = gca;
%h2.XTickLabel = classificationTree.PredictorNames(ind);
%[GeneralizedLinearModel, trainedClassifier, validationAccuracy]=DecisionTree(selectedData);

% [Accuracy, Precision, Recall, F1, classificationSVM, validationPredictions, trainingData]=SVM(selectedData);
% 
% table= table(validationPredictions);
% results= [trainingData  table];

%ScoreSVMModel = fitPosterior(classificationSVM);
%[label,scores] = resubPredict(classificationSVM);
%[~,postProbs] = resubPredict(ScoreSVMModel);
% table(realY(1:1012),label(1:1012),scores(1:1012,2),postProbs(1:1012,2),'VariableNames',...
%     {'TrueLabel','PredictedLabel','Score','PosteriorProbability'})
%GeneralizedLinearModel
%imp






% %test
% filename='C:\Users\Bar\Documents\תואר\תואר שני\תזה\ניתוחים\ניתוח רלוונטי\test2.xlsx';
% opts = detectImportOptions(filename, 'PreserveVariableNames',true);
% opts.VariableNamesRange = 'A1';
% opts.DataRange = 'A2';
% Tdata = readtable(filename, opts,'ReadVariableNames',true);
% 
% testData=Tdata(:,[8 9 20 22 24]);
% 
% testLabels=Tdata(:,1);
% 
% yfit = trainedClassifier.predictFcn(testData);
% x=table2array(testLabels);
% 
% SVMM=trainedClassifier.ClassificationSVM;
% 
% [labels,score] = predict(SVMM,testData);
% 
% %[C,order]= confusionmat(x,yfit);
% %confusionchart(C)
% cc = confusionchart(x,yfit);
% correctPredictions = (x == yfit);
% Accuracyt = sum(correctPredictions)/length(correctPredictions);
% Precisiont = cc.NormalizedValues(2,2)/(cc.NormalizedValues(2,2)+cc.NormalizedValues(1,2));
% Recallt = cc.NormalizedValues(2,2)/(cc.NormalizedValues(2,2)+cc.NormalizedValues(2,1));
% F1t= (2*Precisiont*Recallt)/(Precisiont+Recallt);
% 
% 
% 
% [Xt,Yt,Tt,AUCt,OPTROCPTt] = perfcurve(x,score(:,2),1);
% figure(2)
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by Logistic Regression')