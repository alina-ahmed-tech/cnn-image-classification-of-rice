% Load image data
imds = imageDatastore('Rice Image Dataset','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%%%%%%%%%%%%%%%%%%%%%%%%%%% Initial ALEXNET %%%%%%%%%%%%%%%%%%%
% Load image data
imds = imageDatastore('Rice Image Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the data into training, validation, and test sets
[imdsTrain, imdsRemaining] = splitEachLabel(imds, 0.70, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 0.15, 'randomized');

% Load pre-trained AlexNet
net = alexnet;

% Define transfer layers
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Define new layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Create imageDataAugmenter object with random reflection and translation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
% Apply color jittering 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
% No data augmentation for validation data
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('adam', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false);

% Train the network
netTransfer = trainNetwork(augimdsTrain,layers,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy: ', num2str(accuracy)]);

% Calculate confusion matrix
C = confusionmat(YValidation, YPred);
disp('Confusion Matrix:');
disp(C);

% Plot confusion matrix
figure;
confusionchart(C, categories(imdsValidation.Labels));
title('Confusion Matrix');

%%%%%%%%%%%%%%%%%%%%%%%%%%% Initial GOOGLENET %%%%%%%%%%%%%%%%%%%
% Load image data
imds = imageDatastore('Rice Image Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the data into training, validation, and test sets
[imdsTrain, imdsRemaining] = splitEachLabel(imds, 0.70, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 0.15, 'randomized');

% Load pre-trained Googlenet
net = googlenet;

inputSize = net.Layers(1).InputSize;
% Create layer graph from googlenet
lgraph = layerGraph(net);
% Find layers to remove
layersToRemove = {'loss3-classifier', 'prob', 'output'};
% Remove specified layers
lgraph = removeLayers(lgraph, layersToRemove);
% Define new fully connected layer
numClasses = numel(categories(imdsTrain.Labels));
newFullyConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
% Define softmax layer
softmaxLayer = softmaxLayer();
% Define classification layer
classificationLayer = classificationLayer();
classificationLayer.Name = 'classOutput';
% Add new layers
lgraph = addLayers(lgraph, newFullyConnectedLayer);
lgraph = addLayers(lgraph, softmaxLayer);
lgraph = addLayers(lgraph, classificationLayer);
% Connect new layers
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
lgraph = connectLayers(lgraph, 'new_fc', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classOutput');

% Create imageDataAugmenter object with random reflection and translation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
% Apply color jittering 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
% No data augmentation for validation data
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('adam', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false);

% Train the network
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy: ', num2str(accuracy)]);

% Calculate confusion matrix
C = confusionmat(YValidation, YPred);
disp('Confusion Matrix:');
disp(C);

% Plot confusion matrix
figure;
confusionchart(C, categories(imdsValidation.Labels));
title('Confusion Matrix');

%----------------------------------------------------- Data Splicing - Alexnet --------------------------------------------------
% Total number of images in the dataset
total_images = numel(imds.Files);

% Data splitting proportions
train_validate_splice = {0.80, 0.70, 0.60};
test_splice = {0.1, 0.15, 0.20};

% Preallocate array to store accuracies
num_iterations = 3;
average_accuracies = zeros(numel(train_validate_splice), 1);

for i = 1:numel(train_validate_splice)
    accuracies = zeros(num_iterations, 1); % Store accuracies for each iteration

    % Calculate the number of images for each split
    train_size = total_images * train_validate_splice{i};
    validation_size = total_images * (1 - train_validate_splice{i} - test_splice{i});
    test_size = total_images * test_splice{i};

    % Split the data into training, validation, and test sets
    [imdsTrain, imdsRemaining] = splitEachLabel(imds, train_validate_splice{i}, 'randomized');
    [imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 1 - train_validate_splice{i} - test_splice{i}, 'randomized');

    % Calculate the number of classes
    num_classes = numel(categories(imdsTrain.Labels));

    for iter = 1:num_iterations
        % Load pre-trained AlexNet
        net = alexnet;

        % Define transfer layers
        inputSize = net.Layers(1).InputSize;
        layersTransfer = net.Layers(1:end-3);

        % Define new layers
        layers = [
            layersTransfer
            fullyConnectedLayer(num_classes,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
            softmaxLayer
            classificationLayer];

        % Data augmentation
        pixelRange = [-30 30];
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange);
        augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
            'DataAugmentation',imageAugmenter);
        augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

        % Define training options
        options = trainingOptions('sgdm', ...
            'MiniBatchSize',10, ...
            'MaxEpochs',6, ...
            'InitialLearnRate',1e-4, ...
            'Shuffle','every-epoch', ...
            'ValidationData',augimdsValidation, ...
            'ValidationFrequency',3, ...
            'Verbose',false);

        % Train the network
        netTransfer = trainNetwork(augimdsTrain,layers,options);

        % Data augmentation for test images (including resizing)
        augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, 'DataAugmentation', imageAugmenter);

        % Classify test data
        YPred = classify(netTransfer, augimdsTest);
        YTest = imdsTest.Labels;

        % Calculate accuracy
        accuracies(iter) = mean(YPred == YTest);
    end

    % Calculate average accuracy for the data split
    average_accuracies(i) = mean(accuracies);

    % Display average accuracy for this data split
    disp(['Average accuracy for Data split ', num2str(i), ': ', num2str(average_accuracies(i))]);
end

% Display average accuracies for all data splits
disp('Average accuracies for all Data Splits in Alexnet:');
disp(average_accuracies);
%----------------------------------------------------- Data Splicing - Googlenet --------------------------------------------------
% Total number of images in the dataset
total_images = numel(imds.Files);

% Data splitting proportions
train_validate_splice = {0.80, 0.70, 0.60};
test_splice = {0.1, 0.15, 0.20};

% Preallocate array to store accuracies
num_iterations = 3;
average_accuracies = zeros(numel(train_validate_splice), 1);

for i = 1:numel(train_validate_splice)
    accuracies = zeros(num_iterations, 1); % Store accuracies for each iteration

    % Calculate the number of images for each split
    train_size = total_images * train_validate_splice{i};
    validation_size = total_images * (1 - train_validate_splice{i} - test_splice{i});
    test_size = total_images * test_splice{i};

    % Split the data into training, validation, and test sets
    [imdsTrain, imdsRemaining] = splitEachLabel(imds, train_validate_splice{i}, 'randomized');
    [imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 1 - train_validate_splice{i} - test_splice{i}, 'randomized');

    % Calculate the number of classes
    num_classes = numel(categories(imdsTrain.Labels));

    for iter = 1:num_iterations
        % Load pre-trained Googlenet
        net = googlenet;

        inputSize = net.Layers(1).InputSize;
        % Create layer graph from googlenet
        lgraph = layerGraph(net);
        % Find layers to remove
        layersToRemove = {'loss3-classifier', 'prob', 'output'};
        % Remove specified layers
        lgraph = removeLayers(lgraph, layersToRemove);
        % Define new fully connected layer
        numClasses = numel(categories(imdsTrain.Labels));
        newFullyConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
        % Define softmax layer
        softmaxLayer = softmaxLayer();
        % Define classification layer
        classificationLayer = classificationLayer();
        classificationLayer.Name = 'classOutput';
        % Add new layers
        lgraph = addLayers(lgraph, newFullyConnectedLayer);
        lgraph = addLayers(lgraph, softmaxLayer);
        lgraph = addLayers(lgraph, classificationLayer);
        % Connect new layers
        lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
        lgraph = connectLayers(lgraph, 'new_fc', 'softmax');
        lgraph = connectLayers(lgraph, 'softmax', 'classOutput');

        % Data augmentation
        pixelRange = [-30 30];
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange);
        augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
            'DataAugmentation',imageAugmenter);
        augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

        % Define training options
        options = trainingOptions('sgdm', ...
            'MiniBatchSize',10, ...
            'MaxEpochs',6, ...
            'InitialLearnRate',1e-4, ...
            'Shuffle','every-epoch', ...
            'ValidationData',augimdsValidation, ...
            'ValidationFrequency',3, ...
            'Verbose',false);

        % Train the network
        netTransfer = trainNetwork(augimdsTrain,lgraph,options);

        % Data augmentation for test images (including resizing)
        augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, 'DataAugmentation', imageAugmenter);

        % Classify test data
        YPred = classify(netTransfer, augimdsTest);
        YTest = imdsTest.Labels;

        % Calculate accuracy
        accuracies(iter) = mean(YPred == YTest);
    end

    % Calculate average accuracy for the data split
    average_accuracies(i) = mean(accuracies);

    % Display average accuracy for this data split
    disp(['Average accuracy for Data split ', num2str(i), ': ', num2str(average_accuracies(i))]);
end

% Display average accuracies for all data splits
disp('Average accuracies for all Data Splits in Googlenet:');
disp(average_accuracies);
%-------------------------------------------- Training Functions/Optimizers - Alexnet -----------------------------------------------------------------
% Load pre-trained AlexNet
net = alexnet;

% Define transfer layers
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Define new layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options for different optimizers
optimizers = {'sgdm', 'adam', 'rmsprop'};
for opt = optimizers
optimizer = opt{1};
% Define training options
options = trainingOptions(optimizer, ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false);

% Train the network
disp(['Training with optimizer in Alexnet: ', optimizer]);
netTransfer = trainNetwork(augimdsTrain,layers,options);
% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
% Display results
idx = randperm(numel(imdsValidation.Files),4);
% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy with ', optimizer, ' optimizer: ', num2str(accuracy)]);
end
%-------------------------------------------- Training Functions/Optimizers - Googlenet -----------------------------------------------------------------
% Load pre-trained Googlenet
net = googlenet;

inputSize = net.Layers(1).InputSize;
% Create layer graph from googlenet
lgraph = layerGraph(net);
% Find layers to remove
layersToRemove = {'loss3-classifier', 'prob', 'output'};
% Remove specified layers
lgraph = removeLayers(lgraph, layersToRemove);
% Define new fully connected layer
numClasses = numel(categories(imdsTrain.Labels));
newFullyConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
% Define softmax layer
softmaxLayer = softmaxLayer();
% Define classification layer
classificationLayer = classificationLayer();
classificationLayer.Name = 'classOutput';
% Add new layers
lgraph = addLayers(lgraph, newFullyConnectedLayer);
lgraph = addLayers(lgraph, softmaxLayer);
lgraph = addLayers(lgraph, classificationLayer);
% Connect new layers
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
lgraph = connectLayers(lgraph, 'new_fc', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classOutput');

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options for different optimizers
optimizers = {'sgdm', 'adam', 'rmsprop'};
for opt = optimizers
optimizer = opt{1};
% Define training options
options = trainingOptions(optimizer, ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false);

% Train the network
disp(['Training with optimizer in Googlenet: ', optimizer]);
netTransfer = trainNetwork(augimdsTrain,lgraph,options);
% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
% Display results
idx = randperm(numel(imdsValidation.Files),4);
% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy with ', optimizer, ' optimizer: ', num2str(accuracy)]);
end

%-------------------------------------------- HP Tuning - Alexnet -----------------------------------------------------------------

% Load pre-trained AlexNet
net = alexnet;

% Define transfer layers
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Define new layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define arrays for hyperparameters
initialLearnRates = [1e-4, 5e-5, 1e-5];
miniBatchSizes = [5, 10, 20];

% Initialize variables to store results
accuracyResults = zeros(length(initialLearnRates), length(miniBatchSizes));

% Tune initial learning rate
for i = 1:length(initialLearnRates)
    % Define training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', miniBatchSizes(1), ... % Defaulting to the first batch size
        'MaxEpochs', 6, ... % Keeping epochs fixed
        'InitialLearnRate', initialLearnRates(i), ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'Verbose',false);
    
    % Train the network
    netTransfer = trainNetwork(augimdsTrain, layers, options);
    
    % Classify validation data
    YPred = classify(netTransfer, augimdsValidation);
    YValidation = imdsValidation.Labels;
    
    % Calculate accuracy
    accuracyResults(i, 1) = mean(YPred == YValidation);
end

% Find best initial learning rate
[bestInitialLearnRateAccuracy, bestInitialLearnRateIndex] = max(accuracyResults(:, 1));
bestInitialLearnRate = initialLearnRates(bestInitialLearnRateIndex);

disp(['Best initial learning rate: ', num2str(bestInitialLearnRate), ', Accuracy: ', num2str(bestInitialLearnRateAccuracy)]);

% Tune mini-batch size
for j = 1:length(miniBatchSizes)
    % Define training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', miniBatchSizes(j), ...
        'MaxEpochs', 6, ... % Keeping epochs fixed
        'InitialLearnRate', bestInitialLearnRate, ... % Using the best initial learning rate found
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'Verbose',false);
    
    % Train the network
    netTransfer = trainNetwork(augimdsTrain, layers, options);
    
    % Classify validation data
    YPred = classify(netTransfer, augimdsValidation);
    YValidation = imdsValidation.Labels;
    
    % Calculate accuracy
    accuracyResults(bestInitialLearnRateIndex, j) = mean(YPred == YValidation);
end

% Find best mini-batch size
[bestMiniBatchAccuracy, bestMiniBatchIndex] = max(accuracyResults(bestInitialLearnRateIndex, :));
bestMiniBatchSize = miniBatchSizes(bestMiniBatchIndex);

disp(['Best mini-batch size: ', num2str(bestMiniBatchSize), ', Accuracy: ', num2str(bestMiniBatchAccuracy)]);

%---------- show graph for HP tuned and WITHOUT learning rate schedule - Alexnet
% Load pre-trained AlexNet
net = alexnet;

% Define transfer layers
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Define new layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',bestMiniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',bestInitialLearnRate, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augimdsTrain,layers,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy WITHOUT piecwise learning rate schedule: ', num2str(accuracy)]);

%---------- Try changing learning rate schedule from constant to piecwise
% Load pre-trained AlexNet
net = alexnet;

% Define transfer layers
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Define new layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',bestMiniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',bestInitialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augimdsTrain,layers,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy with piecwise learning rate schedule: ', num2str(accuracy)]);

%-------------------------------------------- HP Tuning - Googlenet -----------------------------------------------------------------

% Load pre-trained Googlenet
net = googlenet;

inputSize = net.Layers(1).InputSize;
% Create layer graph from googlenet
lgraph = layerGraph(net);
% Find layers to remove
layersToRemove = {'loss3-classifier', 'prob', 'output'};
% Remove specified layers
lgraph = removeLayers(lgraph, layersToRemove);
% Define new fully connected layer
numClasses = numel(categories(imdsTrain.Labels));
newFullyConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
% Define softmax layer
softmaxLayer = softmaxLayer();
% Define classification layer
classificationLayer = classificationLayer();
classificationLayer.Name = 'classOutput';
% Add new layers
lgraph = addLayers(lgraph, newFullyConnectedLayer);
lgraph = addLayers(lgraph, softmaxLayer);
lgraph = addLayers(lgraph, classificationLayer);
% Connect new layers
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
lgraph = connectLayers(lgraph, 'new_fc', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classOutput');

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define arrays for hyperparameters
initialLearnRates = [1e-4, 5e-5, 1e-5];
miniBatchSizes = [5, 10, 20];

% Initialize variables to store results
accuracyResults = zeros(length(initialLearnRates), length(miniBatchSizes));

% Tune initial learning rate
for i = 1:length(initialLearnRates)
    % Define training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', miniBatchSizes(1), ... % Defaulting to the first batch size
        'MaxEpochs', 6, ... % Keeping epochs fixed
        'InitialLearnRate', initialLearnRates(i), ...
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'Verbose',false);
    
    % Train the network
    netTransfer = trainNetwork(augimdsTrain, lgraph, options);
    
    % Classify validation data
    YPred = classify(netTransfer, augimdsValidation);
    YValidation = imdsValidation.Labels;
    
    % Calculate accuracy
    accuracyResults(i, 1) = mean(YPred == YValidation);
end

% Find best initial learning rate
[bestInitialLearnRateAccuracy, bestInitialLearnRateIndex] = max(accuracyResults(:, 1));
bestInitialLearnRate = initialLearnRates(bestInitialLearnRateIndex);

disp(['Best initial learning rate: ', num2str(bestInitialLearnRate), ', Accuracy: ', num2str(bestInitialLearnRateAccuracy)]);

% Tune mini-batch size
for j = 1:length(miniBatchSizes)
    % Define training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', miniBatchSizes(j), ...
        'MaxEpochs', 6, ... % Keeping epochs fixed
        'InitialLearnRate', bestInitialLearnRate, ... % Using the best initial learning rate found
        'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'ValidationFrequency',3, ...
        'Verbose',false);
    
    % Train the network
    netTransfer = trainNetwork(augimdsTrain, lgraph, options);
    
    % Classify validation data
    YPred = classify(netTransfer, augimdsValidation);
    YValidation = imdsValidation.Labels;
    
    % Calculate accuracy
    accuracyResults(bestInitialLearnRateIndex, j) = mean(YPred == YValidation);
end

% Find best mini-batch size
[bestMiniBatchAccuracy, bestMiniBatchIndex] = max(accuracyResults(bestInitialLearnRateIndex, :));
bestMiniBatchSize = miniBatchSizes(bestMiniBatchIndex);

disp(['Best mini-batch size: ', num2str(bestMiniBatchSize), ', Accuracy: ', num2str(bestMiniBatchAccuracy)]);

%---------- show graph for HP tuned and WITHOUT learning rate schedule - Googlenet
% Load pre-trained Googlenet
net = googlenet;

inputSize = net.Layers(1).InputSize;
% Create layer graph from googlenet
lgraph = layerGraph(net);
% Find layers to remove
layersToRemove = {'loss3-classifier', 'prob', 'output'};
% Remove specified layers
lgraph = removeLayers(lgraph, layersToRemove);
% Define new fully connected layer
numClasses = numel(categories(imdsTrain.Labels));
newFullyConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
% Define softmax layer
softmaxLayer = softmaxLayer();
% Define classification layer
classificationLayer = classificationLayer();
classificationLayer.Name = 'classOutput';
% Add new layers
lgraph = addLayers(lgraph, newFullyConnectedLayer);
lgraph = addLayers(lgraph, softmaxLayer);
lgraph = addLayers(lgraph, classificationLayer);
% Connect new layers
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
lgraph = connectLayers(lgraph, 'new_fc', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classOutput');

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',bestMiniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',bestInitialLearnRate, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy WITHOUT piecewise learning rate schedule: ', num2str(accuracy)]);
%---------- Try changing learning rate schedule from constant to piecwise
% Load pre-trained Googlenet
net = googlenet;

inputSize = net.Layers(1).InputSize;
% Create layer graph from googlenet
lgraph = layerGraph(net);
% Find layers to remove
layersToRemove = {'loss3-classifier', 'prob', 'output'};
% Remove specified layers
lgraph = removeLayers(lgraph, layersToRemove);
% Define new fully connected layer
numClasses = numel(categories(imdsTrain.Labels));
newFullyConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
% Define softmax layer
softmaxLayer = softmaxLayer();
% Define classification layer
classificationLayer = classificationLayer();
classificationLayer.Name = 'classOutput';
% Add new layers
lgraph = addLayers(lgraph, newFullyConnectedLayer);
lgraph = addLayers(lgraph, softmaxLayer);
lgraph = addLayers(lgraph, classificationLayer);
% Connect new layers
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
lgraph = connectLayers(lgraph, 'new_fc', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classOutput');

% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',bestMiniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',bestInitialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy WITH piecewise learning rate schedule: ', num2str(accuracy)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Final Optimised ALEXNET %%%%%%%%%%%%%%%%%%%
% Load image data
imds = imageDatastore('Rice Image Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the data into training, validation, and test sets
[imdsTrain, imdsRemaining] = splitEachLabel(imds, 0.80, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 0.10, 'randomized');

% Load pre-trained AlexNet
net = alexnet;

% Define transfer layers
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

% Define new layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% Create imageDataAugmenter object with random reflection and translation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
% Apply color jittering 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
% No data augmentation for validation data
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('rmsprop', ...
    'MiniBatchSize',5, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',5e-05, ...
    'LearnRateSchedule','piecewise', ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false);

% Train the network
netTransfer = trainNetwork(augimdsTrain,layers,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy: ', num2str(accuracy)]);

% Calculate confusion matrix
C = confusionmat(YValidation, YPred);
disp('Confusion Matrix:');
disp(C);

% Plot confusion matrix
figure;
confusionchart(C, categories(imdsValidation.Labels));
title('Confusion Matrix');

%%%%%%%%%%%%%%%%%%%%%%%%%%% Final Optimised GOOGLENET %%%%%%%%%%%%%%%%%%%
% Load image data
imds = imageDatastore('Rice Image Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split the data into training, validation, and test sets
[imdsTrain, imdsRemaining] = splitEachLabel(imds, 0.70, 'randomized');
[imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 0.15, 'randomized');

% Load pre-trained Googlenet
net = googlenet;

inputSize = net.Layers(1).InputSize;
% Create layer graph from googlenet
lgraph = layerGraph(net);
% Find layers to remove
layersToRemove = {'loss3-classifier', 'prob', 'output'};
% Remove specified layers
lgraph = removeLayers(lgraph, layersToRemove);
% Define new fully connected layer
numClasses = numel(categories(imdsTrain.Labels));
newFullyConnectedLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc');
% Define softmax layer
softmaxLayer = softmaxLayer();
% Define classification layer
classificationLayer = classificationLayer();
classificationLayer.Name = 'classOutput';
% Add new layers
lgraph = addLayers(lgraph, newFullyConnectedLayer);
lgraph = addLayers(lgraph, softmaxLayer);
lgraph = addLayers(lgraph, classificationLayer);
% Connect new layers
lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'new_fc');
lgraph = connectLayers(lgraph, 'new_fc', 'softmax');
lgraph = connectLayers(lgraph, 'softmax', 'classOutput');

% Create imageDataAugmenter object with random reflection and translation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
% Apply color jittering 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
% No data augmentation for validation data
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',5, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-05, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false);

% Train the network
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

% Classify validation data
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;

% Calculate accuracy
accuracy = mean(YPred == YValidation);
disp(['Accuracy: ', num2str(accuracy)]);

% Calculate confusion matrix
C = confusionmat(YValidation, YPred);
disp('Confusion Matrix:');
disp(C);

% Plot confusion matrix
figure;
confusionchart(C, categories(imdsValidation.Labels));
title('Confusion Matrix');

% Function to perform color jittering
function imgOut = augmentColorHSV(img)
    % Convert RGB image to HSV
    imgHSV = rgb2hsv(img);
    % Randomly adjust hue, saturation, and value
    imgHSV(:, :, 1) = imgHSV(:, :, 1) + (rand - 0.5) * 0.1; % Adjust hue
    imgHSV(:, :, 2) = imgHSV(:, :, 2) + (rand - 0.5) * 0.1; % Adjust saturation
    imgHSV(:, :, 3) = imgHSV(:, :, 3) + (rand - 0.5) * 0.1; % Adjust value
    % Convert back to RGB
    imgOut = hsv2rgb(imgHSV);
end
