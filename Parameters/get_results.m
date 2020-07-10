function get_results(test_image_count, annot_file, output_file, save_file)

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


results = parameters_cal(output,annot);
%name of results file
csvwrite(save_file,results);