function get_results(test_image_count, annot_file, output_file, save_file,choose)
% choose 1 then label based

dict_size = 291;

% %image count
% test_image_count = 1957; 

%annoted file
fileId = fopen(annot_file,'r');
test_annot = cell(1,test_image_count);

for i = 1:test_image_count
    test_annot{1,i} = fscanf(fileId,'%f,');
end

fclose(fileId);

%generated output file here
fileId = fopen(output_file,'r');
test_output = cell(1,test_image_count);

for i = 1:test_image_count
    test_output{1,i} = fscanf(fileId,'%f,');
end

fclose(fileId);

output = zeros(test_image_count,dict_size);
annot = zeros(test_image_count,dict_size);

for i = 1:test_image_count
    for j = 1:dict_size
        output(i,j) = test_output{1,i}(j,1);
        annot(i,j) = test_annot{1,i}(j,1);
    end    
end

if choose == 1 
    results = parameters_cal_label(output,annot,test_image_count,dict_size);
    %name of results file
    csvwrite(save_file,results);
else
    results = parameters_cal_image(output,annot,test_image_count,dict_size);
    %name of results file
    csvwrite(save_file,results);
end