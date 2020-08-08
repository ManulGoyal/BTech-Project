% features = {'DenseHue.hvecs', 'DenseHueV3H1.hvecs', 'DenseSift.hvecs', 'DenseSiftV3H1.hvecs', 'Gist.fvec', ...
%     'HarrisHue.hvecs', 'HarrisHueV3H1.hvecs', 'HarrisSift.hvecs', 'HarrisSiftV3H1.hvecs', 'Hsv.hvecs32', ...
% 'HsvV3H1.hvecs32', 'Lab.hvecs32', 'LabV3H1.hvecs32', 'Rgb.hvecs32', 'RgbV3H1.hvecs32'};
% dist_metrics = {'chi_square', 'chi_square', 'chi_square', 'chi_square', 'l2', 'chi_square', ...
%     'chi_square', 'chi_square', 'chi_square', 'l1', 'l1', 'l1', 'l1', 'l1', 'l1'};
% sets = {'train', 'test'};
% datasetsCap = {'iaprtc12', 'ESPGame', 'IAPRTC12'};
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

[test_annot] = get_test_annot();
[train_annot] = get_train_annot();


% iaprtc12_test_annot = double(vec_read(['datasets/' datasetsCap{ids} '/' datasets{ids} '_test_annot.hvecs']));
% iaprtc12_train_annot = double(vec_read(['datasets/' datasetsCap{ids} '/' datasets{ids} '_train_annot.hvecs']));

iaprtc12_test_annot = zeros(test_image_count(ids),dict_size(ids));
iaprtc12_train_annot = zeros(train_image_count(ids),dict_size(ids));

for i = 1:dict_size(ids)
    for j = 1:train_image_count(ids)
        iaprtc12_train_annot(j,i) = train_annot{1,j}(i,1);
    end
end

for i = 1:dict_size(ids)
    for j = 1:test_image_count(ids)
        iaprtc12_test_annot(j,i) = test_annot{1,j}(i,1);
    end
end

% [results] = parameters_cal(iaprtc12_test_annot,iaprtc12_test_annot);

iaprtc12_label_train_freq = sum(iaprtc12_test_annot);     

distf = load('iaprtc12_distances.mat');
iaprtc12_distances = distf.distances;

cooccur = (iaprtc12_train_annot.')*iaprtc12_train_annot;
test_labels = zeros(test_image_count(ids), dict_size(ids));
for i = 1:test_image_count(ids)
    distances = iaprtc12_distances(i, :);
    
    [~, neighbours] = sort(distances);

    % Perform label transfer here
    labels = zeros(1, dict_size(ids));                % labels to be assigned to test image
    
    % Sorting labels for nearest neighbour wrt their frequencies in
    % training dataset
    nearest_nbr_labels = find(iaprtc12_train_annot(neighbours(1), :));
    [~, label_freq_sort] = sort(iaprtc12_label_train_freq(nearest_nbr_labels), 'descend');
    nearest_nbr_labels = nearest_nbr_labels(label_freq_sort);
        
    sz = numel(nearest_nbr_labels);
    if sz >= labels_per_image
        % assign first n labels to test image if nearest nbr labels are
        % more than n (n = labels_per_image)
        labels(nearest_nbr_labels(1:labels_per_image)) = 1;    
    else
        % if nearest nbr has less than n labels, assign all of them to
        % test image
        labels(nearest_nbr_labels(1:sz)) = 1;
        other_nbrs_annot = iaprtc12_train_annot(neighbours(2:nearest_neighbours), :);
        local_labels_freq = sum(other_nbrs_annot);
        other_nbrs_labels = find(local_labels_freq);
        local_labels_cooccurrence = zeros(1, dict_size(ids));
        for lbl = 1:numel(other_nbrs_labels)
            if ismember(other_nbrs_labels(lbl), nearest_nbr_labels)
                continue; 
            end
            local_labels_cooccurrence(other_nbrs_labels(lbl)) = sum(cooccur(other_nbrs_labels(lbl), nearest_nbr_labels));
        end
        local_labels_priority = local_labels_freq + 0.5*local_labels_cooccurrence;
        transferrable_labels_cnt = numel(other_nbrs_labels);
        [~, other_lbls_sort] = sort(local_labels_priority, 'descend');
        labels(other_lbls_sort(1:min(labels_per_image-sz, transferrable_labels_cnt))) = 1;

        
    end
    test_labels(i, :) = labels;


end


csvwrite("iaprtc12_knn_output_add05.csv",test_labels);