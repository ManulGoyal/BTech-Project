function [a] = get_train_annot()

fileId = fopen('remove_images_train.txt','r');
r = fscanf(fileId,'%f,');
fclose(fileId);

fileId = fopen('train_annot.txt','r');
%   17495
sz = 17495;
szl = 17665;

% a = r(1);
% b = r(2);
a = cell(1,sz);
re = 1;
c = cell(1);
d = numel(r);
for i = 1:szl
    if re <= numel(r) &&  i == r(re)
        re=re+1;
        c = fscanf(fileId,'%f,');
    else
        a{1,i-re+1} = fscanf(fileId,'%f,');  
    end
end
fclose(fileId);

end