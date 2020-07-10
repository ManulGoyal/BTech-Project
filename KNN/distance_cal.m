% features = {'DenseHue.hvecs', 'DenseHueV3H1.hvecs', 'DenseSift.hvecs', 'DenseSiftV3H1.hvecs', 'Gist.fvec', ...
%     'HarrisHue.hvecs', 'HarrisHueV3H1.hvecs', 'HarrisSift.hvecs', 'HarrisSiftV3H1.hvecs', 'Hsv.hvecs32', ...
% 'HsvV3H1.hvecs32', 'Lab.hvecs32', 'LabV3H1.hvecs32', 'Rgb.hvecs32', 'RgbV3H1.hvecs32'};
% dist_metrics = {'chi_square', 'chi_square', 'chi_square', 'chi_square', 'l2', 'chi_square', ...
%     'chi_square', 'chi_square', 'chi_square', 'l1', 'l1', 'l1', 'l1', 'l1', 'l1'};
% sets = {'train', 'test'};
% datasetsCap = {'Corel5k', 'ESPGame', 'IAPRTC12'};
datasets = ['iaprtc12'];
test_image_count = [1957];
train_image_count = [17495];
dict_size = [291];
%17495
%1957

features = {'dia'};
dist_metrics = {'l2'};
ids = 1;  

labels_per_image = 5;           %labels to be allotted per test image
nearest_neighbours = 5;         %number of nearest neighbours considered per test image

% data_features = cell(numel(datasets)*numel(sets), numel(features));

data_features = cell(2,train_image_count(ids));

fileId = fopen('train_features.txt','r');
% sizeA = [1957 536] 
% A = fscanf(fileId,'%f,');
% B = fscanf(fileId,'%f,');
% c = fscanf(fileId,'%f,');

for i = 1:train_image_count(ids)
    % for j = 1:537
        data_features{1,i} = fscanf(fileId,'%f,');
        % fscanf(fileId,',');
    % end
    % fscanf(fileId,'/n');
end
fclose(fileId);


fileId = fopen('test_features.txt','r');
% sizeA = [1957 536] 
% A = fscanf(fileId,'%f,');
% B = fscanf(fileId,'%f,');
% c = fscanf(fileId,'%f,');

for i = 1:test_image_count(ids)
    % for j = 1:537
        data_features{2,i} = fscanf(fileId,'%f,');
        % fscanf(fileId,',');
    % end
    % fscanf(fileId,'/n');
end
fclose(fileId);


% for i = 1:numel(features)
%     data_features{2*ids-1,i} = double(vec_read(['datasets/' datasetsCap{ids} '/' datasets{ids} '_train_' features{i}]));
%     data_features{2*ids,i} = double(vec_read(['datasets/' datasetsCap{ids} '/' datasets{ids} '_test_' features{i}]));
% end


% minf = load('max_iaprtc12.mat');
% maxf = load('min_iaprtc12.mat');
% mini = minf.mini;
% maxi = maxf.maxi;

distances = zeros(test_image_count(ids), train_image_count(ids));
for i = 1:test_image_count(ids)

    for j = 1:train_image_count(ids)
        dist = zeros(1, numel(features));
        for k = 1:numel(features)
            test_ft = data_features{2, i};
            train_ft = data_features{1, j};
            switch dist_metrics{k}
                case 'chi_square'
                    train_ft = train_ft / (0.0000000001+norm(train_ft));
                    test_ft = test_ft / (0.0000000001+norm(test_ft));
                    dist(k) = (0.5*sum(((train_ft-test_ft).^2)./(train_ft+test_ft+0.0000000001)));
                case 'l1'
                    train_ft = train_ft / (0.0000000001+sum(abs(train_ft)));
                    test_ft = test_ft / (0.0000000001+sum(abs(test_ft)));
                    dist(k) = sum(abs(train_ft-test_ft));
                case 'l2'
                    train_ft = train_ft / (0.0000000001+norm(train_ft));
                    test_ft = test_ft / (0.0000000001+norm(test_ft));
                    dist(k) = sqrt(sum((train_ft-test_ft).^2));
            end
            % dist(k) = (dist(k) - mini(ids, k))/(maxi(ids, k)-mini(ids, k)+0.0000000001);
        end
        distances(i, j) = sum(dist)/numel(features);
    end
end

csvwrite("iaprtc12_distances.csv",distances);
save('iaprtc12_distances.mat', 'distances', '-v7.3');

% save([datasets{ids} '_distances.mat'], 'distances', '-v7.3');

