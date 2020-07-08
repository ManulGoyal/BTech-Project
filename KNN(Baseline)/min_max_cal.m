features = {'DenseHue.hvecs', 'DenseHueV3H1.hvecs', 'DenseSift.hvecs', 'DenseSiftV3H1.hvecs', 'Gist.fvec', ...
    'HarrisHue.hvecs', 'HarrisHueV3H1.hvecs', 'HarrisSift.hvecs', 'HarrisSiftV3H1.hvecs', 'Hsv.hvecs32', ...
    'HsvV3H1.hvecs32', 'Lab.hvecs32', 'LabV3H1.hvecs32', 'Rgb.hvecs32', 'RgbV3H1.hvecs32'};
dist_metrics = {'chi_square', 'chi_square', 'chi_square', 'chi_square', 'l2', 'chi_square', ...
    'chi_square', 'chi_square', 'chi_square', 'l1', 'l1', 'l1', 'l1', 'l1', 'l1'};
sets = {'train', 'test'};
datasetsCap = {'Corel5k', 'ESPGame', 'IAPRTC12'};
datasets = {'corel5k', 'espgame', 'iaprtc12'};
test_image_count = [499 2081 1962];
train_image_count = [4500 18689 17665];
dict_size = [260 268 291];

data_features = cell(numel(datasets)*numel(sets), numel(features));

ids = 3;   
  
for i = 1:numel(features)
    data_features{2*ids-1,i} = double(vec_read(['datasets/' datasetsCap{ids} '/' datasets{ids} '_train_' features{i}]));
end


mini = Inf(numel(datasets),numel(features));
maxi = zeros(numel(datasets),numel(features));

train_set_distance = 0;

for k = 1:numel(features)   
    
    for i = 1:(train_image_count(ids)-1)
    
        for j = i+1:train_image_count(ids)
    
            train1_ft = data_features{2*ids-1,k}(i,:);
            train2_ft = data_features{2*ids-1,k}(j,:);
            
            switch dist_metrics{k}
                case 'chi_square'
                    train1_ft = train1_ft / norm(train1_ft);
                    train2_ft = train2_ft / norm(train2_ft);
                    train_set_distance = (sum(((train1_ft-train2_ft).^2)./(train1_ft+train2_ft+0.0000000001)));
                case 'l1'
                    train1_ft = train1_ft / sum(abs(train1_ft));
                    train2_ft = train2_ft / sum(abs(train2_ft));
                    train_set_distance = sum(abs(train1_ft-train2_ft));
                case 'l2'
                    train1_ft = train1_ft / norm(train1_ft);
                    train2_ft = train2_ft / norm(train2_ft);
                    train_set_distance = (sqrt(sum((train1_ft-train2_ft).^2)));
            end

            mini(ids,k) = min(mini(ids,k),train_set_distance);
            maxi(ids,k) = max(maxi(ids,k),train_set_distance);
        end

    end

end



save('ma.mat', 'maxi', '-v7.3');
save('mi.mat', 'mini', '-v7.3');

save(['max_' datasets{ids} '.mat'], 'maxi', '-v7.3');
save(['min_' datasets{ids} '.mat'], 'mini', '-v7.3');