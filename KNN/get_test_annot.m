function [a] = get_test_annot()

fileId = fopen('remove_images_test.txt','r');
r = fscanf(fileId,'%f,');
fclose(fileId);

fileId = fopen('test_annot.txt','r');
%   1957
sz = 1957;
szl = 1962;

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
