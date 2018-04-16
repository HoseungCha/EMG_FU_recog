%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
% 2017.09.13 DTW변환 100개 늘림(함수로 간략화시킴)
%--------------------------------------------------------------------------
clc; close all; clear all;
% get tool box
addpath(genpath(fullfile('E:\Hanyang\연구','_matlab_toolbox')));
parentdir=(fileparts(pwd));
addpath(genpath(fullfile(cd,'functions')));

%% Feature SET 가져오기 (for analysis)
name_feat_file = 'Feature_set_wininc_time_0.10_winsize_204';
path_feat = fullfile(parentdir,'DB','DB_processed',name_feat_file);
load(fullfile(path_feat,'feat_set'));

%% DB set 가져오기 (for train-less DB )
path_DB = 'E:\Hanyang\연구\EMG_TrainLess_Expression\코드\DB\ProcessedDB';
name_DB_file = 'feat_set_combined_seg_30_using_ch4';
load(fullfile(path_DB,name_DB_file));
features_DB = feat_set_combined; clear feat_set_combined;
%% 실험 정보
[i_seg, n_feat, n_FE, n_trial, n_sub , n_emg_pair] = size(Features); % DB to be analyzed
n_sub_DB = size(features_DB,5); % Database
idx_sub = 1 : n_sub;
idx_trl = 1 : n_trial;
% trigger singals corresponding to each facial expression(emotion)
name_Trg = {"무표정",1,1;"눈썹 세게 치켜 올리기",1,2;"눈썹 세게 내려서 인상쓰기",1,3;...
    "양쪽 입고리 세게 올리기",1,4;"코 찡그리기",1,5;"'ㅏ' 모양으로 입모양 짓기",1,6;...
    "'ㅓ' 모양으로 입모양 짓기",1,7;"'ㅗ' 모양으로 입모양 짓기",2,1;...
    "'ㅜ' 모양으로 입모양 짓기",2,2;"'ㅡ' 모양으로 입모양 짓기",2,3;...
    "'l' 모양으로 입모양 짓기",2,4};
name_classifier = {'cf_eyebrow','cf_cheeck','cf_lipshapes'};
n_cf = length(name_classifier);
idx_FE4CF = {[1,2,3],[1,4,5],[6,7,8,10,11]};
%% feature indexing
% when using DB of ch4 ver
idx_feat.RMS = 1:4;
idx_feat.WL = 5:8;
idx_feat.SampEN = 9:12;
idx_feat.CC = 13:28;
%% feat names and indices
names_feat = fieldnames(idx_feat);
idx_feat = struct2cell(idx_feat);
n_ftype = length(names_feat);
%% decide how many number of tranfored feat from DB 
n_transforemd = 0;
n_emg_pair = 3;
% makeing folder for results 결과 저장 폴더 설정
% folder_name2make = ['T5_',name_feat_file]; % 폴더 이름
% path_made = make_path_n_retrun_the_path(fullfile(parentdir,...
%     'DB','dist'),folder_name2make); % 결과 저장 폴더 경로
%% memory allocation for reults
result.acc = zeros(i_seg,n_trial,n_sub,n_emg_pair,n_cf,n_transforemd+1);
result.output_n_target = cell(i_seg,n_trial,n_sub,n_emg_pair,n_cf,n_transforemd+1);   
result.model = cell(n_trial,n_sub,n_emg_pair,n_transforemd+1);  
% for i_emg_pair = 1 : n_emg_pair
for i_emg_pair = 1
for i_sub = 1 : n_sub
    for i_trial = 1 : n_trial
        fprintf('i_sub:%d i_trial:%d\n',i_sub,i_trial);
        %% get similar feature from DB
        if n_transforemd~=0
        t = cell(i_seg,n_FE);
        for i_seg = 1 : i_seg
            for i_FE = 1 : n_FE
                t{i_seg,i_FE} = cell(1,n_ftype);
                for i_FeatName = 1 : n_ftype
                    %% get DB with a specific feature
                    n_feat_interested = length(idx_feat{i_FeatName});
                    feat_ref = Features(i_seg,idx_feat{i_FeatName} ,i_FE,...
                        i_trial,i_sub,i_emg_pair)';
                    feat_DB = features_DB(:,idx_feat{i_FeatName} ,:,:,:);
%                     feat_ref = feat(i_seg,:,i_FE,i_trial,i_sub)';
                    feat_compr = feat_DB(i_seg,:,i_FE,:,:);
                    feat_compr = reshape(feat_compr,...
                        [n_feat_interested, n_trial*n_sub_DB]);
                    % just get 5 similar features
                    t{i_seg,i_FE}{i_FeatName} = ...
                        dtw_search_n_transf(feat_ref, feat_compr, n_transforemd)';
                end
            end
        end
        % concatinating features with types
        t = cellfun(@(x) cat(2,x{:}),t,'UniformOutput',false);
        else
        t = [];
        end
        %% arrange feat transformed and target
        % validate with number of transformed DB
        for n_trans = 0: n_transforemd
            if n_trans~=0
            % get feature-transformed with number you want
            feat_trans = cellfun(@(x) x(1:n_trans,:),t,...
                'UniformOutput',false);
            % get size to have target
            size_temp = cell2mat(cellfun(@(x) size(x,1),...
                feat_trans(:,1),'UniformOutput',false));
            % feature transformed 
            feat_trans = cell2mat(feat_trans(:));
            % target for feature transformed 
            target_feat_trans = repmat(1:n_FE,sum(size_temp,1),1);
            target_feat_trans = target_feat_trans(:); 
            else
            feat_trans = [];
            target_feat_trans = [];
            end
            % feat for anlaysis
            feat_ref = reshape(permute(Features(:,:,:,i_trial,i_sub,i_emg_pair),...
                [1 3 2]),[i_seg*n_FE,n_feat]);
            target_feat_ref = repmat(1:n_FE,i_seg,1);
            target_feat_ref = target_feat_ref(:);
            % get input and targets for train DB
            input_train = cat(1,feat_ref,feat_trans);
            target_train = cat(1,target_feat_ref,target_feat_trans);
            % get input and targets for test DB
            % [n_seg, n_feat, n_FE, n_trial, n_sub=1 , n_emg_pair=1] -->
            % [n_seg, n_trial, n_FE, n_feat, n_sub=1 , n_emg_pair=1] -->
            % [n_seg*n_trial*n_FE, n_feat]
            input_test = reshape(permute(Features(:,: ,:,idx_trl~=i_trial,...
                i_sub,i_emg_pair),[1 4 3 2]),[i_seg*(n_trial-1)*n_FE,n_feat]);
            target_test = repmat(1:n_FE,i_seg*(n_trial-1),1);
            target_test = target_test(:);
            % train with three classfiers
            model = cell(n_cf,1);
            for i_cf = 1 : n_cf
            % get index of train samples
            tmp = find(countmember(target_train,idx_FE4CF{i_cf})==1);
            n_class = length(idx_FE4CF{i_cf});
            % train
            model{i_cf} = fitcdiscr(input_train(tmp,:),target_train(tmp));
            % test
            % get index of train samples
            tmp = find(countmember(target_test,idx_FE4CF{i_cf})==1);
            output_test = predict(model{i_cf},input_test(tmp,:));
            % reshape ouput_test as <seg, trl, FE>
            output_test = reshape(output_test,[i_seg,(n_trial-1),n_class]);
            output_mv_test = majority_vote(output_test,idx_FE4CF{i_cf});
            % reshape target test for acc caculation
            target_test4cf = repmat(idx_FE4CF{i_cf},(n_trial-1),1);
            target_test4cf = target_test4cf(:);
            for i_seg = 1 : i_seg
                % <N_Seg,N_trl,N_Class> -> <N_trl*N_Class,1>
                ouput_seg = output_mv_test(i_seg,:)';
                tmp = sum(target_test4cf==ouput_seg)/(n_class*(n_trial-1))*100;
%                 disp(tmp); % dispay of acc
                result.acc(i_seg,i_trial,i_sub,i_emg_pair,i_cf,n_trans+1) = tmp;
                result.output_n_target{i_seg,i_trial,i_sub,i_emg_pair,i_cf,n_trans+1}...
                    = [ouput_seg,target_test4cf];
            end
            end
            result.model{i_trial,i_sub,i_emg_pair,n_trans+1} = model;
        end
    end
end
end
%% save results
% id of classfied expression
tmp = [];
for i = 1 : 3
tmp = [tmp,num2str(idx_FE4CF{i}),' _ '];
end
tmp(isspace(tmp)) = []; % remove blanks
tmp(end) = []; % remove end of '_'

path4result = make_path_n_retrun_the_path(path_feat,'result');
save(fullfile(path4result,['result ',tmp]),'result');
%% plot results
tmp = result.acc;
% i_seg,i_trial,i_sub,i_emg_pair,i_cf,1 -->
% mean: i_seg,1,1,i_emg_pair,i_cf,1 -->
% i_seg,i_cf,i_emg_pair
tmp = permute(mean(mean(tmp,2),3),[1 5 4 2 3]);
for i_emg_pair = 1 : n_emg_pair
    figure(i_emg_pair)
    plot(tmp(:,:,i_emg_pair))
end
%% analysis of confustion matrix
for i_emg_pair = 1
for i_cf = 1 : 3
n_class = length(idx_FE4CF{i_cf});
name_FE_of_class = name_Trg(idx_FE4CF{i_cf},1);
for i_seg = 15
    % get ouputs and targets
    tmp = result.output_n_target(i_seg,:,:,i_emg_pair,i_cf);
    tmp = cell2mat(tmp(:));
    output_tmp = full(ind2vec(tmp(:,1)'));
    target_tmp = full(ind2vec(tmp(:,2)'));
    tmp = countmember(1:max(idx_FE4CF{i_cf}),idx_FE4CF{i_cf})==0;
    output_tmp(tmp,:) = [];
    target_tmp(tmp,:) = [];
    
    [~,mat_conf,idx_of_samps_with_ith_target,~] = confusion(target_tmp,output_tmp);
    figure(i_cf);
    plotConfMat(mat_conf, name_FE_of_class)
    mat_n_samps = cellfun(@(x) size(x,2),idx_of_samps_with_ith_target);
    mat_n_samps(logical(eye(size(mat_n_samps)))) = 0;
    fn_sum_of_each_class = sum(mat_n_samps,1);
    
end
end
end
