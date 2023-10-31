clear;
close all;
clc

data = readtable('Yagis.csv');

data.removevars(data,'Var16');
data.removevars(data,'Var17');
data.removevars(data,'Var18');
data.removevars(data,'Var19');


data_matrix = data{:,:};

norm_data = (data_matrix - min(data_matrix)) ./ (max(data_matrix)-min(data_matrix));


cv = cvpartition(size(norm_data,1),'HoldOut',0.2);
idx = cv.test;
dataTrain = norm_data(~idx,:);
dataTest = norm_data(idx,:);

y_train = dataTrain(:,15);
x_train = dataTrain(:,1:14);
x_test = dataTest(:,1:14);
y_test = dataTest(:,15);

svmModel = fitcsvm(x_train,y_train);
[predictedLabels,scores] = predict(svmModel,x_test);

conf_mat = confusionmat(y_test,predictedLabels);
figure
confusionchart(conf_mat);

svmModel.ClassNames;
rocObj = rocmetrics(y_test,scores,svmModel.ClassNames);
rocObj.AUC;
idx = strcmp(rocObj.Metrics.ClassName,svmModel.ClassNames(1));
head(rocObj.Metrics(idx,:));
plot(rocObj);

Accuracymodel = 100*sum(diag(conf_mat))./sum(conf_mat(:));

Recall = conf_mat(1,1)/(conf_mat(1,1)+conf_mat(1,2));

Precision = conf_mat(1,1)/(conf_mat(1,1)+conf_mat(2,1));

