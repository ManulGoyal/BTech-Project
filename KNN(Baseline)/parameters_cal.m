function  [results] = parameters_cal(test_labels)


% sets = {'train', 'test'};
% datasetsCap = {'Corel5k', 'ESPGame', 'IAPRTC12'};
% datasets = {'corel5k', 'espgame', 'iaprtc12'};
test_image_count = [499 2081 1962];
train_image_count = [4500 18689 17665];
dict_size = [260 268 291];

ids = 3;

% iaprtc12_test_annot = double(vec_read(['datasets/' datasetsCap{ids} '/' datasets{ids} '_test_annot.hvecs']));
% iaprtc12_train_annot = double(vec_read(['datasets/' datasetsCap{ids} '/' datasets{ids} '_train_annot.hvecs']));
% iaprtc12_label_train_freq = sum(iaprtc12_train_annot);     

load('iaprtc12_semantic_hierarchy_structure.mat');
iaprtc12_test_annot = full(semantic_hierarchy_structure.label_test_SH_augmented);


mean_precision = 0;
mean_recall = 0;
n_plus = 0;
for l = 1:dict_size(ids)
    ground_truth = sum(iaprtc12_test_annot(:, l));
    predicted = sum(test_labels(:, l));
    correct = sum(iaprtc12_test_annot(1:test_image_count(ids), l) & test_labels(:, l));
    if correct > 0
        n_plus = n_plus + 1;
    end
    mean_precision = mean_precision + correct/(predicted+1e-10);
    mean_recall = mean_recall + correct/ground_truth;
end

mean_precision = 100*mean_precision/dict_size(ids);
mean_recall = 100*mean_recall/dict_size(ids);
f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-10);
% results = [mean_precision mean_recall f1_score n_plus];

[semantic_precision,semantic_recall,semantic_f1] = semantic(dict_size(ids),test_labels,iaprtc12_test_annot);
results = [mean_precision,mean_recall,f1_score,n_plus,semantic_precision,semantic_recall,semantic_f1];

T = table(mean_precision,mean_recall,f1_score,n_plus,semantic_precision,semantic_recall,semantic_f1);
writetable(T,'results1.txt');

% save([datasets{ids} '_results_p1.mat'], 'results');
end % of funtion