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
idx_fe2cfy = [1, 5, 7, 9, 10];
n_fe_cfy = length(idx_fe2cfy);

% get name list of subjects
[name_subject,~] = read_names_of_file_in_folder(path_DB_raw);

% classifier of each facial part
name_classifier = {'cf_eyebrow','cf_lipshapes'};
n_cf = 2;

% mapped avartar gesture of each facial part of each emotion
tmp = name_fe(idx_fe2cfy);
name_gesture_clfr = cell(n_fe_cfy,n_cf);
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
idx_ch_fe2classfy = {[2,6,10,14,18,22,26,3,7,11,15,19,23,27],...
    [1,5,9,13,17,21,25,4,8,12,16,20,24,28]};
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

% memory allocatoin for output and target
r.output_n_target = cell(n_emg_pair,n_sub,n_trl,n_transforemd+1,n_seg,(n_trl-1),n_fe);

% memory allocatoin for output and target
r.model= cell(n_emg_pair,n_sub,n_trl,n_transforemd+1);

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
                
                %=================TRAIN FACIAL PARTS======================%
                
                % get features of determined emotions that you want to classify
                model.parts = cell(n_cf,1);
                for i_cf = 1 : n_cf
                    idx_train_samples_2_classify =...
                        countmember(target_train,idx_fe2cfy)==1;
                    
                    % train
                    model.parts{i_cf} = fitcdiscr(...
                        input_train(idx_train_samples_2_classify,idx_ch_fe2classfy{i_cf}),...
                        target_train(idx_train_samples_2_classify));
                end
                % saving models
                r.model{i_emg_pair,i_sub,i_trl,n_t+1} = model;
               
                
                %=================PREPARE DB FOR TEST=====================%
                % get input and targets for test DB
                input_test = reshape(permute(feat(:,: ,:,idx_trl~=i_trl,...
                    i_sub,i_emg_pair),[1 4 3 2]),[n_seg*(n_trl-1)*n_fe,n_feat]);
                target_test = repmat(1:n_fe,n_seg*(n_trl-1),1);
                target_test = target_test(:);
                %=========================================================%
                
                %=========================TEST============================%
                % TEST SHOULD BE EVALUATED BY AVARTAR EXPRESSION
                
                % +++++CLASSIFICATION OF EMG ONSET
                % get RMS features of TEST
                input_test_rms = input_test(:,idx_feat{1});
                output_emg_onset = cell(n_bip_ch,1);
                score_emg_onset = cell(n_bip_ch,1);
                for i_bip_ch = 1 : n_bip_ch
                    [tmp,tmp2] = ...
                        predict(model_tree_emg_onset,input_test_rms(:,i_bip_ch));
                    output_emg_onset{i_bip_ch} = reshape(tmp,[n_seg,(n_trl-1),n_fe]);
                    score_emg_onset{i_bip_ch} = reshape(tmp2,[n_seg,(n_trl-1),n_fe,2]);
                end
                
                % +++++CLASSIFICATION BY emotion model+++++++++
                [~,score_test_emotion] = predict(model.emotion,input_test);
                score_test_emotion = reshape(score_test_emotion,[n_seg,(n_trl-1),n_fe,n_fe]);
                
                % +++++CLASSIFICATION BY faical parts model+++++++++
                output_test_parts = cell(n_cf,1);
                score_test_parts = cell(n_cf,1);
                for i_cf = 1 : n_cf
                [tmp,tmp2] = predict(model.parts{i_cf},...
                    input_test(:,idx_ch_fe2classfy{i_cf}));
                output_test_parts{i_cf} = reshape(tmp,[n_seg,(n_trl-1),n_fe]);
                score_test_parts{i_cf} = reshape(tmp2,[n_seg,(n_trl-1),n_fe,n_fe_cf{i_cf}]);
                end                
                
                %===============APPLYING MAJORITY VOTING==================%
                for i_bip_ch = 1 : n_bip_ch
                    [output_emg_onset{i_bip_ch},score_emg_onset{i_bip_ch}]  = ...
                        majority_vote_by_score(score_emg_onset{i_bip_ch},[0,1]);
                end
                
                score_onset_parts{1} = mean(cat(4,score_emg_onset{2},score_emg_onset{3}),4);
                score_onset_parts{2} = mean(cat(4,score_emg_onset{1},score_emg_onset{4}),4);
                
                [output_test_emotion,score_test_emotion] = ...
                    majority_vote_by_score(score_test_emotion,1:n_fe);
                
                for i_cf = 1 : n_cf
                    [output_test_parts{i_cf},score_test_parts{i_cf}] = ...
                    majority_vote_by_score(score_test_parts{i_cf},idx_fe2cfy{i_cf});    
                end
                %=========================================================%
                
                
                %====================MAKE AVARTAR=========================%
                % the acc can be computed according to 'NATURAL AVARTAR (
                % if the avartar made an expression in relation to emotion,
                % regard it as corrected emotion)
                
                
                for i_fe = 1 : n_fe
%                 for i_trl_test = 1
                for i_trl_test = 1 : (n_trl-1)
                for i_seg = 1 : n_seg
%                 for i_seg = 15
                % compare P.P. of each classifiers !!!!!!!!!!!!!!!!!!
                score_matrix = NaN(n_cf,3);
                output_matrix = NaN(n_cf,3);
                for i_cf = 1 : n_cf
                    % score (P.P) matrix
                    score_matrix(i_cf,1) = score_onset_parts{i_cf}(i_seg,i_trl_test,i_fe)*2;     
                    score_matrix(i_cf,2) = score_test_emotion(i_seg,i_trl_test,i_fe)*n_fe;
                    score_matrix(i_cf,3) = score_test_parts{i_cf}(i_seg,i_trl_test,i_fe)*n_fe_cf{i_cf};
                    
                    % ouput matrix
                    output_matrix(i_cf,1) = output_emg_onset{i_cf}(i_seg,i_trl_test,i_fe);     
                    output_matrix(i_cf,2) = output_test_emotion(i_seg,i_trl_test,i_fe);
                    output_matrix(i_cf,3) = output_test_parts{i_cf}(i_seg,i_trl_test,i_fe);
                end
                disp(score_matrix);
                disp(output_matrix);
                
                % select classfier with p.p. with hightest value
                max_scores = max(score_matrix,[],2);
                idx = score_matrix == max_scores;
                
                idx_clf = NaN(n_cf,1);
                for i_cf = 1 : n_cf
                    idx_clf(i_cf) = find(idx(i_cf,:));
                end                
                
                maxval = max(max_scores);
                idx2 = find(max_scores == maxval==1);
                if length(idx2) == 2
                    part_criteria = randi(2);
                else
                    part_criteria = idx2;
                end
                
                % decide final output of facial part which will be used as
                % crietria
                output_final = NaN(2,1);                
                output_final(part_criteria) = output_matrix(part_criteria,idx_clf(part_criteria));
                idx_part_2_be_searched = find(isnan(output_final)==1);
                
                % decide final ouput of other facial part based on 
                % facial expressions generated naturally
                output_final(idx_part_2_be_searched) = ...
                    output_matrix(idx_part_2_be_searched,idx_clf(idx_part_2_be_searched));
                
                % if output has zero(which comes from EMG onset detection)
                output_final(output_final==0) = 9;
                
                if part_criteria == 1 % criterian has been set with eye brow
                    % get possible lip expressions based on eye brow 
                    possible_exps = ...
                        natural_lip_exp_selctor(output_final(part_criteria));
                elseif part_criteria == 2 % criterian has been set with lips
                    % get possible eye brow expressions based on lip shapes
                    possible_exps = ...
                        natural_eye_brow_exp_selctor(output_final(part_criteria));
                end
                
                % set eye brow within possible eye brow expression
                output_final(idx_part_2_be_searched) = get_output_within_possible_outcomes(...
                    possible_exps,output_final(idx_part_2_be_searched));
                disp(output_final);
                
                
                % named gestures of each facial part
                gest_avartar = cell(n_cf,1);
                for i_cf = 1 : n_cf
                gest_avartar{i_cf} = name_gesture_clfr{i_cf}{output_final(i_cf)};
                end
                
                % composing avartar by each classfied final ouput
%                 plot_avartar(gest_avartar{1},'neutral',gest_avartar{2});
%                 title(name_fe{i_fe})
%                 set(gcf,'Position',[2613 242 560 420]);
%                 drawnow;
                
                % validation
                output = [];
                for i_target = 1 : n_target
                    if any(ismember(fe_target{i_target},output_final','rows'))
                        output = i_target;
                        break;
                    end
                end
                if isempty(output)
                    keyboard;
                end
                
                if i_fe ==3 || i_fe==4
                    target = 3;
                elseif i_fe == 5
                    target = 4;
                elseif i_fe == 6 || i_fe == 10
                    target = 5;
                elseif i_fe == 7
                    target = 6;
                elseif i_fe == 8
                    target = 7;
                elseif i_fe == 9
                    target = 8;
                elseif i_fe == 11
                    target =9;
                else
                    target = i_fe;
                end
                    
                r.acc(i_emg_pair,i_sub,i_trl,n_t+1,i_seg,i_trl_test,i_fe)...
                    = output==target;
                r.output_n_target{i_emg_pair,i_sub,i_trl,n_t+1,i_seg,i_trl_test,i_fe}...
                    = [output,target];
                disp([output,target]);
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

%------------------------------results processing-------------------------%
tmp = mean(mean(r.acc(1,:,:,:,:,:,:),2),3);
tmp = squeeze(tmp);
%-------------------------------------------------------------------------%




%============================FUNCTIONS====================================%

%=========================================================================%