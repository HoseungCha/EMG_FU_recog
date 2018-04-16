function yp = majority_vote(xp,idx_target2classify)
% final decision using majoriy voting
% yp has final prediction X segments(times)
% xp has <N_Seg,N_trl,N_label>
% yp has <N_Seg,N_trl,N_label>
[N_Seg,N_trl,N_label] = size(xp);
% yp = zeros(N_label*N_trl,1);
yp = zeros(N_Seg,N_trl,N_label);
for n_seg = 1 : N_Seg
    maxv = zeros(N_label,N_trl); final_predict = zeros(N_label,N_trl);
    for i_label = 1 : N_label
        for i_trl = 1 : N_trl
            [maxv(i_label,i_trl),tmp] = max(countmember(idx_target2classify,...
                xp(1:n_seg,i_trl,i_label)));
%             final_predict(i_label,i_trl) = idx_target2classify(tmp);
            yp(n_seg,i_trl,i_label) = idx_target2classify(tmp);
        end
    end
%     yp(:,n_seg) = final_predict(:);
%     acc(n_seg,N_comp+1) = sum(repmat((1:label)',[N_trl,1])==final_predict)/(label*N_trial-label*n_pair)*100;
end
% yp = reshape(yp,N_label,N_trl,N_Seg);
% yp = permute(yp,[3 2 1]);
end