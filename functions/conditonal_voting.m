function y_corrected = conditonal_voting(ypred);
% conditoinal voting
% condition #1
template_c1 = logical([1 1 1 1 1]);
n_c1 = length(template_c1);
% condition #2
template_c2 = logical([1 1 0 1 1]);
n_c2 = length(template_c2);
% condition #3
template_c3 = logical([1 1 1 0 0 0 0 0 0 0 0 0 1]);
n_c3 = length(template_c3);
% condition #4
template_c4 = logical([1 1 0 1 1 0 0 0 0 0 0 0 1]);
n_c4 = length(template_c4);

c1_fail = 0;
c2_fail = 0;
count = 0;
y_corrected = NaN(length(ypred),1);
activated = 0;
for i = 1 : length(ypred)
    if i < n_c1
        continue;
    end
    tmp_d = ypred(i-n_c1+1:i);
    
    % condition #1
    tmp = tmp_d(template_c1);
    if ~range(tmp) % check all values are same
        y_corrected(i) = ypred(i);
        c1_fail = 0;
    else
        c1_fail = 1;
    end
    
    %         % condition #2
    %         tmp = tmp_d(template_c2);
    %         if ~range(tmp) % check all values are same
    %              y_corrected(i) = ypred(i);
    %              c2_fail =0;
    %         else
    %             c2_fail = 1;
    %         end
    %
    if c1_fail
        y_corrected(i) = y_corrected(i-1);
    end
    
    if i < n_c3
        continue;
    end
    
    % condition #3
    tmp_d = ypred(i-n_c3+1:i);
    tmp = tmp_d(template_c3);
    if ~range(tmp(1:3))&&tmp(4)==9&&tmp(1)~=9 % check all values are same
        y_corrected(i) = ypred(i);
        c3_fail = 0;
        activated = 1;
    else
        c3_fail = 1;
    end
    
    if activated
        count = count + 1;
    end
    if count > 0 && count <=10
        y_corrected(i) = 9;
    end
    if count == 10
        count = 0;
        activated = 0;
    end
end
end