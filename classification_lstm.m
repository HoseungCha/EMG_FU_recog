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
name_DB_raw = 'DB_raw2';

% name of process DB to analyze in this code
name_DB_process = 'DB_processed2';

% load feature set, which was extracted by feat_extration.m
name_DB_analy = 'feat_set_DB_raw2_n_sub_39_n_seg_30_n_wininc_204_winsize_204';

% decide if validation datbase set
id_DB_val = 'myoexp2'; % myoexp1, myoexp2, both

% decide number of tranfored feat from DB
n_transforemd = 0;

% decide whether to use emg onset fueatre
id_use_emg_onset_feat = 0;

% decide which attibute to be compared when applying train-less algoritm
% [n_seg:30, n_feat:28, n_fe:8, n_trl:20, n_sub:30]
% 'all' : [:,:,:,:,:], 'Only_Seg' : [i_seg,:,:,:,:], 'Seg_FE' : [i_seg,:,i_FE,:,:]
id_att_compare = 'Seg_FE'; % 'all', 'Only_Seg', 'Seg_FE'
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


% check if there are feat from other experiment and if thre is, concatinate
if exist('feat_DB','var')
feat = cat(5,feat,feat_DB);
end


% LOAD MODEL of EMG ONSET DETECTION
load(fullfile(path_DB_process,'model_tree_emg_onset.mat'));
%-------------------------------------------------------------------------%

%-----------------------experiment information----------------------------%
% trigger singals corresponding to each facial expression(emotion)
% name_fe = ["angry";...
%     "clench";"contemptous left";"contemptous right";...
%     "nose wrinkled";"fear";"happy";...
%     "kiss";"neutral";"sad";...
%     "surprised"];
name_fe = {'eye_brow_down-lip_tighten';'neutral-clench';...
'neutral-lip_corner_up_left';'neutral-lip_corner_up_right';...
'eye_brow_down-lip_corner_up_both';'eye_brow_sad-lip_stretch_down';...
'eye_brow_happy-lip_happy';'neutral-kiss';'neutral-neutral';...
'eye_brow_sad-lip_sulky';'eye_brow_up-lip_open'};

% get name list of subjects
[name_subject,~] = read_names_of_file_in_folder(path_DB_raw);

% mapped avartar gesture of each facial part of each emotion
for i = 1 : length(name_fe)
tmp = strsplit(name_fe{i},'-');
name_gesture_c1{1}{i,1} = tmp{1};
name_gesture_c1{2}{i,1} = tmp{2};
end


% name of types of features
name_feat = {'RMS';'WL';'SampEN';'CC'};
% correct_label
% fe_target.angry = [1,1;1,2;1,6;1,9;1,10;1,11;9,1;2,1;3,1;4,1;8,1;2,1;3,1;4,1;8,1];
% fe_target.clench = [9,2;2,2;3,2;4,2;8,2];
% fe_target.contempt = [9,3;9,4;2,3;2,4;3,3;3,4;4,3;4,4;8,3;8,4];
% fe_target.frown = [1,5];
% fe_target.fear_sad = [6,1;6,2;6,6;6,9;6,10;6,11;10,1;10,2;10,6;10,9;10,10;10,11;9,6;9,10;2,6;2,10;3,6;3,10;4,6;4,10;8,6;8,10];
% fe_target.happy = [7,5;7,7;9,5;2,5;3,5;4,5;8,5;2,5;3,5;4,5;5,5];
% fe_target.kiss = [9,8;2,8;3,8;4,8;8,8];
% fe_target.neutral = [9,9;2,9;3,9;4,9;8,9];
% fe_target.surprised = [11,9;11,11;9,11;2,11;3,11;4,11;8,11];
%
% name_target = fieldnames(fe_target);
% fe_target = struct2cell(fe_target);
% n_target = length(fe_target);

%=============numeber of variables
[n_seg, n_feat, n_fe, n_trl, n_sub , n_emg_pair] = size(feat);
n_part =2;
n_cf = 2;
n_emg_pair = 1;
n_sub_compared = n_sub - 1;
n_ftype = length(name_feat);
n_bip_ch = 4;
n_fe_cf = {6,8};
%=============indices of variables
idx_sub = 1 : n_sub;
idx_sub_exp1 = idx_sub(end-15:end);
idx_sub_exp2 = find(countmember(idx_sub,idx_sub_exp1)==0==1);
switch id_DB_val
case 'myoexp1'
idx_sub_val = idx_sub_exp1;
case 'myoexp2'
idx_sub_val = idx_sub_exp2;
case 'both'
idx_sub_val = idx_sub;
end
n_sub_val = length(idx_sub_val);
idx_trl = 1 : n_trl;
idx_feat = {[1,2,3,4];[5,6,7,8];[9,10,11,12];13:28};
% idx_ch_fe2classfy = {[2,6,10,14,18,22,26,3,7,11,15,19,23,27],...
%     [1,5,9,13,17,21,25,4,8,12,16,20,24,28]};
idx_ch_fe2classfy = {[2,6,10,14,18,22,26,3,7,11,15,19,23,27],...
[1,5,9,13,17,21,25,4,8,12,16,20,24,28,]};
if (id_use_emg_onset_feat)
idx_ch_fe2classfy{1} = [idx_ch_fe2classfy{1},31:34];
idx_ch_fe2classfy{2} = [idx_ch_fe2classfy{2},29:30,35:36];
end
idx_emg_onest = [0,1];
idx_ch_face_parts = {[2,3],[1,4]};

for i_part = 1 : n_part
[name_gesture_c2{i_part},~,label_gesture{i_part}] = unique(name_gesture_c1{i_part});
n_class(i_part) = length(name_gesture_c2{i_part});
end
name_gesture = [name_gesture_c1',name_gesture_c2'];

% check numeber of subject from DB processed and DB raw
if length(name_subject) ~= n_sub
error('please first do analysis of raw DB')
end
%-------------------------------------------------------------------------%

%----------------------set saving folder----------------------------------%
% set folder for saving
name_folder_saving = [id_att_compare,'_',id_DB_val,'_',num2str(n_transforemd)];

% set saving folder for windows
path_saving = make_path_n_retrun_the_path(path_DB_analy,name_folder_saving);
%-------------------------------------------------------------------------%

%----------------------memory allocation for results----------------------%
% memory allocatoin for accurucies
r.acc = NaN(n_emg_pair,n_sub_val,n_trl,n_transforemd+1,n_seg,(n_trl-1),n_fe,n_cf+1);

% memory allocatoin for output and target
r.output_n_target = cell(n_emg_pair,n_sub_val,n_trl,n_transforemd+1,n_seg,(n_trl-1),n_fe,n_cf+1);
% memory allocatoin for output and target
% r.model= cell(n_emg_pair,n_sub_val,n_trl,n_transforemd+1);

%-------------------------------------------------------------------------%

%------------------------------------main---------------------------------%

% get accrucies and output/target (for confusion matrix) with respect to
% subject, trial, number of segment, FE,
for i_emg_pair = 1 : n_emg_pair
for i_sub = idx_sub_val
%     for i_sub = 10:23
%         for i_trl = 6
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

%=================EMG ONSET FEATURE=======================%
score  = cell(n_bip_ch,1);
for i_ch = 1 : n_bip_ch
    [~,score{i_ch}] = predict(model_tree_emg_onset,...
        input_train(:,i_ch));
end
score = cat(2,score{:});
if (id_use_emg_onset_feat)
input_train = [input_train,score];
end
%=========================================================%


%==================TRAIN EACH EMOTION=====================%
model.emotion = fitcdiscr(...
    input_train,...
    target_train);
%=========================================================%

%=================TRAIN FACIAL PARTS======================%
% get features of determined emotions that you want to classify
model.parts = cell(n_part,1);
for i_part = 1 : n_part
    tmp_target = repmat(label_gesture{i_part}',n_seg,1);
    tmp_target = tmp_target(:);

    % train
    model.parts{i_part} = fitcdiscr(...
        input_train(:,idx_ch_fe2classfy{i_part}),...
        tmp_target,...
        'discrimType','pseudoLinear');
end
% saving models
%     r.model{i_emg_pair,i_sub,i_trl,n_t+1} = model;
%=========================================================%

%================= TEST=====================%
% get input and targets for test DB
idx_trl_test = find(idx_trl~=i_trl==1);

c_t = 0;
for i_trl_test = idx_trl_test
    c_t = c_t + 1;
    for i_fe = 1 : n_fe
        %         for i_fe = [15]
        % save score(likelihood) in circleque
        score_matrix_cq{1,1} = circlequeue(n_seg,n_fe);
        score_matrix_cq{2,1} = circlequeue(n_seg,n_fe);
        score_matrix_cq{1,2} = circlequeue(n_seg,5);
        score_matrix_cq{2,2} = circlequeue(n_seg,n_fe);

        for i_seg = 1 : n_seg

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
            if (id_use_emg_onset_feat)
                f = [f,cat(2,s_ons{:})];
            end
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
%                 disp(score_matrix);
%                 disp(output_matrix);

            % output/target facial gesutres
            %
            for i_clf_method = 1 : n_cf+1
            score_matrix_test = score_matrix;
            output_matrix_test = output_matrix;
            if any(ismember(1:n_cf,i_clf_method))
                score_matrix_test(:,1:n_cf~=i_clf_method) = 0;
                output_matrix_test(:,1:n_cf~=i_clf_method) = 0;
            end

            % get out and target
            output = possible_fe_selector(score_matrix_test,output_matrix_test,...
                 name_gesture,name_gesture_c1,n_part);  % Train 한 표정에서 제한 시켜버림 ( Train 한 표정에서 안나왔을 경우
            target = {name_gesture_c1{1}{i_fe},name_gesture_c1{2}{i_fe}};
            
            % change ouput to number index
            output_n(1) = find(ismember(name_gesture_c2{1},output{1})==1);
            output_n(2) = find(ismember(name_gesture_c2{2},output{2})==1);
            target_n(1) = find(ismember(name_gesture_c2{1},target{1})==1);
            target_n(2) = find(ismember(name_gesture_c2{2},target{2})==1);
            
            % results
            acc = all([isequal(target_n(1),output_n(1)),isequal(target_n(2),output_n(2))]);
            output_n_target = [output_n(1),target_n(1),output_n(2),target_n(2)];


            % result saving
            r.acc(i_emg_pair,i_sub,i_trl,n_t+1,i_seg,c_t,i_fe,i_clf_method)...
                = acc;
            r.output_n_target{i_emg_pair,i_sub,i_trl,n_t+1,i_seg,c_t,i_fe,i_clf_method}...
             = output_n_target;
            end

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
name_saving = sprintf('DB-%s_ntrans-%d_onset-%d-compmethod-%s',...
id_DB_val,n_transforemd,id_use_emg_onset_feat,id_att_compare);

save(fullfile(path_saving,name_saving),'r');

load(fullfile(path_saving,name_saving));
%-------------------------------------------------------------------------%

%==============================부위별 분류 결과============================%
close all
tmp1 = squeeze(r.output_n_target(1,:,:,1,30,:,:));
tmp1 = tmp1(:);
tmp1 = cat(1,tmp1{:});



for i_part = 1 : n_part
% name_neutral = {'neutral','neutral'};
n_fe_result = length(name_gesture_c1{i_part});
output = strcat(tmp1(:,i_part));
target = strcat(tmp1(:,i_part+2));



% name_fe_eye = {'neutral';'eye_brow_down';'eye_brow_happy';'eye_brow_sad'};
[~,tmp]  = ismember(output,name_gesture{i_part,2});
idx2delete = find(tmp==0);
tmp(idx2delete) =[];

B = unique(tmp);
out = [B,histc(tmp,B)];

output_tmp = full(ind2vec(tmp',n_fe_result));

[~,tmp]  = ismember(target,name_gesture{i_part,2});
tmp(idx2delete) =[];
target_tmp = full(ind2vec(tmp',n_fe_result));

B = unique(tmp);
out = [B,histc(tmp,B)];
% compute confusion
[~,mat_conf,idx_of_samps_with_ith_target,~] = ...
confusion(target_tmp,output_tmp);

figure;
h = plotconfusion(target_tmp,output_tmp);
name_conf = strrep(name_gesture{i_part,2},'_',' ');

h.Children(2).XTickLabel(1:n_fe_result) = name_conf;
h.Children(2).YTickLabel(1:n_fe_result)  = name_conf;

% plotConfMat(mat_conf', name_conf)
end
%=========================================================================%

%------------------------------results processing-------------------------%
acc = NaN(n_fe,3);
for i_clf_method = 1 : 3
for i_fe = 1 : n_fe
tmp = squeeze(r.acc(1,:,:,1,30,:,i_fe,i_clf_method));
tmp = tmp(:);
acc(i_fe,i_clf_method) = length(find(tmp==1))/length(tmp);
end
end
%-------------------------------------------------------------------------%



%============================FUNCTIONS====================================%

%=========================================================================%
%=========================================================================%