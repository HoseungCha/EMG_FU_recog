function [yp,yp_score] = majority_vote_by_score(score,idx_target2classify)
% final decision using majoriy voting
% yp has final prediction X segments(times)
% xp has <N_Seg,N_trl,N_label>
% yp has <N_Seg,N_trl,N_label>
[N_Seg,N_trl,N_label,~] = size(score);
% yp = zeros(N_label*N_trl,1);
yp = NaN(N_Seg,N_trl,N_label);
yp_score = NaN(N_Seg,N_trl,N_label);
for n_seg = 1 : N_Seg
    for i_label = 1 : N_label
        for i_trl = 1 : N_trl
            tmp1 = score(1:n_seg,i_trl,i_label,:);
            tmp1 = tmp1(:,:);
            tmp2 = mean(tmp1,1);
            maxval = max(tmp2);
            idx = tmp2 == maxval;
            if length(find(idx==1))>1
                if any(ismember(idx_target2classify(idx),9))
                    yp(n_seg,i_trl,i_label) = 9;
                else
                    idx = randi(length(find(idx==1)));
                    yp(n_seg,i_trl,i_label) = idx_target2classify(idx);
                end
            else
                yp(n_seg,i_trl,i_label) = idx_target2classify(idx);
            end
            yp_score(n_seg,i_trl,i_label) = maxval;
        end
    end
end
end