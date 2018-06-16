%-------------------------------------------------------------------------%
% 1. feat_extraction.m
% 2. classficiation_using_DB.m  %---current code---%
%-------------------------------------------------------------------------%
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%-------------------------------------------------------------------------%
clc; close all; clear all;

%-----------------------Code anlaysis parmaters---------------------------%
% name of raw DB
name_DB_raw = 'DB_raw3';

% name of process DB to analyze in this code
name_DB_process = 'DB_processed3';

% load feature set, which was extracted by feat_extration.m
name_DB_analy = 'feat_set_DB_raw3_n_sub_2_n_seg_30_n_wininc_204_winsize_204';

% decide number of tranfored feat from DB
n_transforemd = 0;

% decide which attibute to be compared when applying train-less algoritm
% [n_seg:30, n_feat:28, n_fe:8, n_trl:20, n_sub:30]
% 'all' : [:,:,:,:,:], 'Only_Seg' : [i_seg,:,:,:,:], 'Seg_FE' : [i_seg,:,i_FE,:,:]
id_att_compare = 'Seg_FE'; % 'all', 'Only_Seg', 'Seg_FE'

idx_fe2reject = [3,9,17];
%-------------------------------------------------------------------------%

%-------------set paths in compliance with Cha's code structure-----------%
% path of research, which contains toolbox
path_research = fileparts(fileparts(fileparts(fullfile(cd))));

% path of code, which
path_code = fileparts(fullfile(cd));
path_DB = fullfile(path_code,'DB');
path_DB_raw = fullfile(path_DB,name_DB_raw);
path_DB_process = fullfile(path_DB,name_DB_process);
path_DB_analy = fullfile(path_DB_process,name_DB_analy);
%-------------------------------------------------------------------------%

%-------------------------add functions-----------------------------------%
% get toolbox
addpath(genpath(fullfile(path_research,'_toolbox')));

% add functions
addpath(genpath(fullfile(cd,'functions')));
%-------------------------------------------------------------------------%

%-----------------------------load DB-------------------------------------%
% load feature set, from this experiment
tmp = load(fullfile(path_DB_process,name_DB_analy,'feat_set_pair_1'));
tmp_name = fieldnames(tmp);
feat = getfield(tmp,tmp_name{1}); %#ok<GFLD>
feat(:,:,idx_fe2reject,:,:) = [];

% check if there are feat from other experiment and if thre is, concatinate
if exist('feat_DB','var')
    feat = cat(5,feat,feat_DB);
end

% LOAD MODEL of EMG ONSET DETECTION
load(fullfile(path_DB_process,'model_tree_emg_onset.mat'));
%-------------------------------------------------------------------------%

%-----------------------experiment information----------------------------%
% trigger singals corresponding to each facial expression(emotion)
name_fe = {'neutral-neutral'
    'eye_brow_down-lip_open'
    'eye_brow_down-lip_stretch_down'
    'eye_brow_down-lip_sulky'
    'eye_brow_down-lip_tighten'
    'eye_brow_down-neutral'
    'eye_brow_happy-lip_happy'
    'eye_brow_sad-lip_open'
    'eye_brow_sad-lip_stretch_down'
    'eye_brow_sad-lip_sulky'
    'eye_brow_sad-lip_tighten'
    'eye_brow_sad-neutral'
    'eye_brow_up-lip_open'
    'eye_brow_up-neutral'
    'neutral-lip_happy'
    'neutral-lip_open'
    'neutral-lip_stretch_down'
    'neutral-lip_sulky'
    'neutral-lip_tighten'
    };
for i = 1 : length(name_fe)
    tmp = strsplit(name_fe{i},'-');
    name_fe_eb{i,1} = tmp{1};
    name_fe_lp{i,1} = tmp{2};
end
name_emo = {'neutral'
    'angry'
    'angry'
    'angry'
    'angry'
    'angry'
    'happy'
    'sad'
    'sad'
    'sad'
    'sad'
    'sad'
    'surprised'
    'surprised'
    'happy'
    'surprised'
    'sad'
    'sad'
    'sad'
    };
name_fe(idx_fe2reject) = [];
name_emo(idx_fe2reject) = [];
name_fe_eb(idx_fe2reject) = [];
name_fe_lp(idx_fe2reject) = [];

name_fe2cfy = {'neutral-neutral';'eye_brow_down-lip_tighten';'eye_brow_happy-lip_happy';...
    'eye_brow_sad-lip_sulky';'eye_brow_up-lip_open'};

idx_fe2cfy = find(contains(name_fe,name_fe2cfy)==1);
n_fe_cfy = length(idx_fe2cfy);

% get name list of subjects
[name_subject,~] = read_names_of_file_in_folder(path_DB_raw);

% classifier of each facial part
name_classifier = {'cf_eyebrow','cf_lipshapes'};
n_part = 2;
n_cf = 2;
n_class = [5,5];
% mapped avartar gesture of each facial part of each emotion
tmp = name_fe(idx_fe2cfy);
name_gesture_clfr = cell(n_fe_cfy,n_part);
for i = 1 : n_fe_cfy
    name_gesture_clfr(i,:) = strsplit(tmp{i},'-');
end

% name of types of features
name_feat = {'RMS';'WL';'SampEN';'CC'};
% % correct_label
% fe_target.angry = [1,1;1,2;1,6;1,9;1,10;1,11;9,1;2,1;3,1;4,1;8,1;2,1;3,1;4,1;8,1];
% fe_target.clench = [9,2;2,2;3,2;4,2;8,2];
% fe_target.contempt = [9,3;9,4;2,3;2,4;3,3;3,4;4,3;4,4;8,3;8,4];
% fe_target.frown = [1,5];
% fe_target.fear_sad = [6,1;6,2;6,6;6,9;6,10;6,11;10,1;10,2;10,6;10,9;10,10;10,11;9,6;9,10;2,6;2,10;3,6;3,10;4,6;4,10;8,6;8,10];
% fe_target.happy = [7,5;7,7;9,5;2,5;3,5;4,5;8,5;2,5;3,5;4,5;5,5];
% fe_target.kiss = [9,8;2,8;3,8;4,8;8,8];
% fe_target.neutral = [9,9;2,9;3,9;4,9;8,9];
% fe_target.surprised = [11,9;11,11;9,11;2,11;3,11;4,11;8,11];

% name_target = fieldnames(fe_target);
% fe_target = struct2cell(fe_target);
% n_target = length(fe_target);

%=============numeber of variables
[n_seg, n_feat, n_fe, n_trl, n_sub , n_emg_pair] = size(feat);
n_sub_compared = n_sub - 1;
n_ftype = length(name_feat);
n_bip_ch = 4;
n_fe_cf = {6,8};
%=============indices of variables
idx_sub = 1 : n_sub;
idx_trl = 1 : n_trl;
% idx_fe2classfy = {[1,6,7,9,10,11],[1,2,3,4,6,7,8,11]};
idx_feat = {[1,2,3,4];[5,6,7,8];[9,10,11,12];13:28};
idx_ch_fe2classfy = {[2,6,10,14,18,22,26,3,7,11,15,19,23,27,31:34],...
    [1,5,9,13,17,21,25,4,8,12,16,20,24,28,29:30,35:36]};
% idx_emg_onest = [0,1];
idx_ch_face_parts = {[2,3],[1,4]};

% check numeber of subject from DB processed and DB raw
if length(name_subject) ~= n_sub
    error('please first do analysis of raw DB')
end
%-------------------------------------------------------------------------%

%----------------------set saving folder----------------------------------%
% set folder for saving
name_folder_saving = [id_att_compare,'_',num2str(n_transforemd)];

% set saving folder for windows
path_saving = make_path_n_retrun_the_path(path_DB_analy,name_folder_saving);
%-------------------------------------------------------------------------%

%----------------------memory allocation for results----------------------%
% memory allocatoin for accurucies
r.acc = NaN(n_emg_pair,n_sub,n_trl,n_transforemd+1,n_seg,(n_trl-1),n_fe);
r.acc_emo = NaN(n_emg_pair,n_sub,n_trl,n_transforemd+1,n_seg,(n_trl-1),n_fe);

% memory allocatoin for output and target
r.output_n_target = cell(n_emg_pair,n_sub,n_trl,n_transforemd+1,n_seg,(n_trl-1),n_fe);
r.output_n_target_emo = cell(n_emg_pair,n_sub,n_trl,n_transforemd+1,n_seg,(n_trl-1),n_fe);
%-------------------------------------------------------------------------%

%------------------------------------main---------------------------------%

% get accrucies and output/target (for confusion matrix) with respect to
% subject, trial, number of segment, FE,
for i_emg_pair = 1 : n_emg_pair
for i_sub = idx_sub
for i_trl = 1 : n_trl
%display of subject and trial in progress
fprintf('i_emg_pair:%d i_sub:%d i_trial:%d\n',i_emg_pair,i_sub,i_trl);
if n_transforemd>=1
    % memory allocation similarily transformed feature set
    feat_t = cell(n_seg,n_fe);
    for i_seg = 1 : n_seg
        for i_FE = 1 : n_fe

            % memory allocation feature set
            feat_t{i_seg,i_FE} = cell(1,n_ftype);

            % you should get access to DB of other experiment with each
            % features
            for i_FeatName = 1 : n_ftype

                % number of feature of each type
                n_feat_type = length(idx_feat{i_FeatName});

                % feat from this experiment
                feat_ref = feat(i_seg,idx_feat{i_FeatName},i_FE,...
                    i_trl,i_sub,i_emg_pair)';

                %---------feat to be compared from this experiment----%
                % [n_seg:30, n_feat:28, n_fe:8, n_trl:20, n_sub:30, n_emg_pair:3]

                % compare ohter subject except its own subject
                idx_sub_DB = find(countmember(idx_sub_val,i_sub)==0==1);
                n_sub_DB = length(idx_sub_DB);
                switch id_att_compare
                    case 'all'
                        feat_compare = feat(:,idx_feat{i_FeatName},...
                            :,:,idx_sub_DB,i_emg_pair);

                    case 'Only_Seg'
                        feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                            :,:,idx_sub_DB,i_emg_pair);

                    case 'Seg_FE'
                        feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                            i_FE,:,idx_sub_DB,i_emg_pair);
                end

                % permutation giving [n_feat, n_fe, n_trl, n_sub ,n_seg]
                % to bring about formation of [n_feat, others]
                feat_compare = permute(feat_compare,[2 3 4 5 1]);

                %  size(2):FE, size(5):seg
                feat_compare = reshape(feat_compare,...
                    [n_feat_type, size(feat_compare,2)*n_trl*n_sub_DB*...
                    size(feat_compare,5)]);

                % get similar features by determined number of
                % transformed DB
                feat_t{i_seg,i_FE}{i_FeatName} = ...
                    dtw_search_n_transf(feat_ref, feat_compare, n_transforemd)';
                %-----------------------------------------------------%

            end
        end
    end

    % arrange feat transformed and target
    % concatinating features with types
    feat_t = cellfun(@(x) cat(2,x{:}),feat_t,'UniformOutput',false);
end
% validate with number of transformed DB
for n_t = 0:n_transforemd
    if n_t >= 1
        % get feature-transformed with number you want
        feat_trans = cellfun(@(x) x(1:n_t,:),feat_t,...
            'UniformOutput',false);

        % get size to have target
        size_temp = cell2mat(cellfun(@(x) size(x,1),...
            feat_trans(:,1),'UniformOutput',false));

        % feature transformed
        feat_trans = cell2mat(feat_trans(:));

        % target for feature transformed
        target_feat_trans = repmat(1:n_fe,sum(size_temp,1),1);
        target_feat_trans = target_feat_trans(:);
    else
        feat_trans = [];
        target_feat_trans = [];
    end

    % feat for anlaysis
    feat_ref = reshape(permute(feat(:,:,:,i_trl,i_sub,i_emg_pair),...
        [1 3 2]),[n_seg*n_fe,n_feat]);
    target_feat_ref = repmat(1:n_fe,n_seg,1);
    target_feat_ref = target_feat_ref(:);
    
    %=================PREPARE DB FOR TRAIN====================%
    input_train = cat(1,feat_ref,feat_trans);
    target_train = cat(1,target_feat_ref,target_feat_trans);
    %=========================================================%

    %============EMG ONSET FEATURES================%
    score  = cell(n_bip_ch,1);
    for i_ch = 1 : 4
        [~,score{i_ch}] = predict(model_tree_emg_onset,input_train(:,i_ch));
    end
    score = cat(2,score{:});
    %===============================================%
    input_train = [input_train,score];
    
    idx_train = countmember(target_train,idx_fe2cfy)==1;
    %==================TRAIN EACH EMOTION=====================%
    model.emotion = fitcdiscr(...
        input_train(idx_train,:),...
        target_train(idx_train),...
        'discrimType','pseudoLinear');
    %=========================================================%

    %=================TRAIN FACIAL PARTS======================%

    % get features of determined emotions that you want to classify
    model.parts = cell(n_part,1);
    for i_part = 1 : n_part
        % train
        model.parts{i_part} = fitcdiscr(...
            input_train(idx_train,idx_ch_fe2classfy{i_part}),...
            target_train(idx_train),...
            'discrimType','pseudoLinear');
    end
    % saving models
    r.model{i_emg_pair,i_sub,i_trl,n_t+1} = model;
    %=========================================================%


    %================= TEST=====================%
    % get input and targets for test DB
    idx_trl_test = find(idx_trl~=i_trl==1);

    c_t = 0;
    for i_trl_test = idx_trl_test
        c_t = c_t + 1;
        for i_fe = 1 : n_fe
%         for i_fe = [6,13]
            % save score(likelihood) in circleque
            score_matrix_cq{1,1} = circlequeue(n_seg,n_fe_cfy);
            score_matrix_cq{2,1} = circlequeue(n_seg,n_fe_cfy);
            score_matrix_cq{1,2} = circlequeue(n_seg,n_fe_cfy);
            score_matrix_cq{2,2} = circlequeue(n_seg,n_fe_cfy);

            for i_seg = 1 : n_seg
%                 if i_seg == 30
%                    keyboard;
%                 end
                   
                % get feature during real-time
                f = feat(i_seg,:,i_fe,i_trl_test,i_sub,i_emg_pair);

                %====PASS THE TEST FEATURES TO CLASSFIERS=============%
                %----EMG ONSET
                f_onset = f(idx_feat{1});
                s_ons = cell(n_bip_ch,1);
                for i_ch = 1 : n_bip_ch
                    [~,s_ons{i_ch}] = ...
                        predict(model_tree_emg_onset,f_onset(i_ch));
                end
                f = [f,cat(2,s_ons{:})];
                %=======EMG ONSET FEATURE ADDITION
                
                
                %----EMOTION CLASSFIER
                [~,s_emo] = predict(model.emotion,f);
                score_matrix_cq{1,1}.add(s_emo);
                score_matrix_cq{2,1}.add(s_emo);

                %----FACE PARTS CLASSFIER
                for i_part = 1 : n_part
                    [~,s_par] = predict(model.parts{i_part},f(idx_ch_fe2classfy{i_part}));
                    score_matrix_cq{i_part,2}.add(s_par);
                end
                %=====================================================%

                score_matrix = NaN(n_part,n_cf);
                output_matrix = NaN(n_part,n_cf);
                for i_part = 1 : n_part
                for i_cf = 1 : n_cf
                    if i_seg<15
                        tmp = mean(score_matrix_cq{i_part,i_cf}.getLastN(i_seg),1);
                    else
                        tmp = mean(score_matrix_cq{i_part,i_cf}.getLastN(15),1);
                    end
                    % get max value
                    [max_v,max_idx] = get_max_and_idx(tmp);

                    % get socre matrix
                    score_matrix(i_part,i_cf) = max_v*n_class(i_cf);
                    output_matrix(i_part,i_cf) = max_idx;
                end
                end
                disp(score_matrix);
                disp(output_matrix);
                
                % temperilly test
                score_matrix_test =  score_matrix;
                score_matrix_test(:,2)  = 0;
                % select classfier with p.p. with hightest value
                score_ouput = NaN(n_part,2);
                for i_part = 1 : n_part
                    % get max value
                    [max_v,max_idx] = get_max_and_idx(score_matrix_test(i_part,:));
                    score_ouput(i_part,:) = [max_v,output_matrix(i_part,max_idx)];
                end
                
                % avarter expression
                output_brow = name_gesture_clfr{score_ouput(1,2),1};
                output_lip = name_gesture_clfr{score_ouput(2,2),2};
                
                % restriction based on score of ouput
                if score_ouput(1,1) >= score_ouput(2,1) % 눈썹 기준으로 입모양 결정
                    posbl_output = name_fe_lp(~cellfun(@isempty,strfind(name_fe_eb,output_brow))); %#ok<STRCLFH>
                    posbl_output = unique(posbl_output);
                    if any(ismember(posbl_output,output_lip)==1)
                        output_lip = posbl_output{ismember(posbl_output,output_lip)};
                    else
                        output_lip = posbl_output{randi(length(posbl_output))};
                    end
                    
                else % 입모양 기준으로 눈썹 모양 결정
                    posbl_output = name_fe_eb(~cellfun(@isempty,strfind(name_fe_lp,output_lip))); %#ok<STRCLFH>
                    posbl_output = unique(posbl_output);
                    if any(ismember(posbl_output,output_brow)==1)
                        output_brow = posbl_output{ismember(posbl_output,output_brow)};
                    else
                        output_brow = posbl_output{randi(length(posbl_output))};
                    end
                end
              
                % 눈썹 무표정, 입모양 행복한 표정 보완
%                 if strcmp(output_brow,'eye_brow_happy')&&strcmp(output_lip,'lip_happy')
%                     if output_matrix(1,1) == 1 
%                         output_brow = 'neutral';
%                     end
%                 end
                
                
                % output/target facial gesutres
                output = {output_brow,output_lip};
                target = strsplit(name_fe{i_fe},'-');
                
                % output/target emotion
%                 ouput_emo = name_emo{contains(name_fe,strcat(output{1},'-',output{2}))};
%                 target_emo = name_emo{contains(name_fe,strcat(target{1},'-',target{2}))};

                % acc
                acc = all([strcmp(target{1},output{1}),strcmp(target{2},output{2})]);
%                 acc_emo = strcmp(ouput_emo,target_emo);
                
                disp(target);
                disp({output_brow,output_lip});
                % get accurcy
                r.acc(i_emg_pair,i_sub,i_trl,n_t+1,i_seg,c_t,i_fe)...
                    = acc;
%                 r.acc_emo(i_emg_pair,i_sub,i_trl,n_t+1,i_seg,c_t,i_fe)...
%                     = acc_emo;
                r.output_n_target{i_emg_pair,i_sub,i_trl,n_t+1,i_seg,c_t,i_fe}...
                    = {output_brow,output_lip,target{1},target{2}};
%                 r.output_n_target_emo{i_emg_pair,i_sub,i_trl,n_t+1,i_seg,c_t,i_fe}...
%                     = {ouput_emo,target_emo};
            end
        end
    end
end
end
end
end
%-----------------^--------------------------------------------------------%

%-------------------------preprocessing of results------------------------%

%-------------------------------------------------------------------------%

%-------------------------------save results------------------------------%
save(fullfile(path_saving,'result'),'r');
%-------------------------------------------------------------------------%

%==============================부위별 분류 결과============================%
close all
tmp1 = squeeze(r.output_n_target(1,:,:,1,30,:,:));
tmp1 = tmp1(:);
tmp1 = cat(1,tmp1{:});



for i_part = 1 : n_part
% name_neutral = {'neutral','neutral'};
name_gesture = [name_gesture_clfr];
n = length(name_gesture(:,i_part));
    
output = strcat(tmp1(:,i_part));
target = strcat(tmp1(:,i_part+2));



% name_fe_eye = {'neutral';'eye_brow_down';'eye_brow_happy';'eye_brow_sad'};
[~,tmp]  = ismember(output,name_gesture(:,i_part));
idx2delete = find(tmp==0);
tmp(idx2delete) =[];

B = unique(tmp);
out = [B,histc(tmp,B)];

output_tmp = full(ind2vec(tmp',n));

[~,tmp]  = ismember(target,name_gesture(:,i_part));
tmp(idx2delete) =[];
target_tmp = full(ind2vec(tmp',n));

B = unique(tmp);
out = [B,histc(tmp,B)];
% compute confusion
[~,mat_conf,idx_of_samps_with_ith_target,~] = ...
    confusion(target_tmp,output_tmp);

figure;
name_conf = strrep(name_gesture(:,i_part),'_',' ');
h = plotconfusion(target_tmp,output_tmp);
h.Children(2).XTickLabel(1:n) = name_conf;
h.Children(2).YTickLabel(1:n)  = name_conf;

% plotConfMat(mat_conf', name_conf)
end
%=========================================================================%

%------------------------------results processing-------------------------%
acc = NaN(n_fe,1);
for i_fe = 1 : n_fe
tmp = squeeze(r.acc(1,:,:,1,30,:,i_fe));
tmp = tmp(:);
acc(i_fe) = length(find(tmp==1))/length(tmp);
end
%-------------------------------------------------------------------------%

%=============================표정 합성 분류 결과==========================%
% 
% tmp1 = squeeze(r.output_n_target(1,:,:,1,30,:,:));
% tmp1 = tmp1(:);
% tmp1 = cat(1,tmp1{:});
% 
% output = strcat(tmp1(:,1),'-',tmp1(:,3));
% target = strcat(tmp1(:,2),'-',tmp1(:,4));
% 
% % name_fe_eye = {'neutral';'eye_brow_down';'eye_brow_happy';'eye_brow_sad'};
% [~,tmp]  = ismember(output,name_fe);
% idx2delete = tmp==0;
% tmp(idx2delete) =[];
% output_tmp = full(ind2vec(tmp',n_fe));
% 
% [~,tmp]  = ismember(target,name_fe);
% tmp(idx2delete) =[];
% target_tmp = full(ind2vec(tmp',n_fe));
% 
% % compute confusion
% [~,mat_conf,idx_of_samps_with_ith_target,~] = ...
%     confusion(target_tmp,output_tmp);
% 
% figure;
% name_fe = strrep(name_fe,'_',' ');
% plotConfMat(mat_conf, name_fe)
%=========================================================================%




%============================FUNCTIONS====================================%
function [max_v,max_idx] = get_max_and_idx(tmp)
[~,idx] = max(tmp);

% check multiple max values occur
idx_max_v = find(tmp(idx)==tmp==1);
if length(idx_max_v)>1
    idx = randi(length(idx_max_v));
    idx = idx_max_v(idx);
end
max_v = tmp(idx);
max_idx = idx;
end
%=========================================================================%