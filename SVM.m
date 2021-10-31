function [response, Accuracy, Precision, Recall, F1,classificationSVM, validationPredictions, validationAccuracy, validationAccuracy2, partitionedModel, validationScores, trainedClassifier] = SVM (trainingData)

% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a table containing the same predictor and response
%       columns as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 04-Mar-2021 11:43:40


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'MTD', 'STD', 'perc90-Tair', 'CWSI2', 'Cv'};
predictors = inputTable(:, predictorNames);
response = inputTable.Y;
isCategoricalPredictor = [false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [false; true]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'CWSI2', 'Cv', 'MTD', 'STD', 'perc90-Tair'};
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2019b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'MTD', 'STD', 'perc90-Tair', 'CWSI2', 'Cv'};
predictors = inputTable(:, predictorNames);
response = inputTable.Y;
%response=logical(response);
isCategoricalPredictor = [false, false, false, false, false];

rng('default');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy

validationAccuracy = 1 - kfoldLoss(partitionedModel,'LossFun', 'ClassifError');

validationAccuracy2 = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError','Mode', 'individual');

%results= [trainingData(:)  table(validationPredictions)];

cc = confusionchart(response,validationPredictions);
correctPredictions = (response == validationPredictions);
Accuracy = sum(correctPredictions)/length(correctPredictions);
Precision = cc.NormalizedValues(2,2)/(cc.NormalizedValues(2,2)+cc.NormalizedValues(1,2));
Recall = cc.NormalizedValues(2,2)/(cc.NormalizedValues(2,2)+cc.NormalizedValues(2,1));
F1= (2*Precision*Recall)/(Precision+Recall);

