function [mean_pre,mean_rec,mean_f1] = semantic(num_instance_test,...
    test_labels,label_test_gt)


% clear
% clear all
% base_path = 'code';
% chdir(base_path)
% addpath(genpath(pwd))

% data_name = 'iaprtc12'; % 'espgame'; 

% %% training 
% DPP_loss_training(data_name);

% %% inference using DPP sampling with weighted semantic paths
% dataset_test= dlmread([data_name, '_data_vggf_pca_test.txt']); 
% options = struct('k1', 8, 'k2', 5, 'num_trials', 10);
% test_result_DIA_inference = DIA_inference(data_name, dataset_test, options); 


%% evaluation using the proposed semantic precision, recall and F1
% load('iaprtc12_semantic_hierarchy_structure');
% label_test_gt = full(semantic_hierarchy_structure.label_test_SH_augmented);

% num_instance_test = size(dataset_test, 1);

prec = zeros(num_instance_test,1);
rec  = zeros(num_instance_test,1);
F1   = zeros(num_instance_test,1);
for i = 1:num_instance_test
    % pred_subset_i = test_result_DIA_inference.sampled_label_subsets{i};

    pred_subset_i = find(test_labels(:,i)==1);
    gt_subset_i = find(label_test_gt(:,i)==1);
    [prec(i), rec(i), F1(i)] = semantic_parameters(pred_subset_i, gt_subset_i);
end

mean_pre = mean(prec)
mean_rec = mean(rec)
mean_f1 = mean(F1)

% [mean(prec), mean(rec), mean(F1); std(prec), std(rec), std(F1)]

end % of funtion