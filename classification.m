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
name_DB_analy = 'feat_set_DB_raw2_n_sub_16_n_seg_30_n_wininc_204_winsize_204';
% 
% % load feature set, from different DB, which was extracted by
% % feat_extraion.m in git_EMG_train_less_FE_recog
% name_DB_from_other_exp = 'feat_set_combined_seg_30_using_ch4';

% decide if you applied train-less algorithm using DB, or DB of other expreiment
id_DBtype = 'DB_own'; % 'DB_own' , 'DB_other' , 'DB_both'

% decide expression to classify
% idx_FE2classfy =[1:11];

% decide number of tranfored feat from DB 
n_transforemd = 1;

% decide which attibute to be compared when applying train-less algoritm
% [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
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
tmp = load(fullfile(path_DB_process,name_DB_analy,'feat_set')); 
tmp_name = fieldnames(tmp);
feat = getfield(tmp,tmp_name{1}); %#ok<GFLD>

% load feature setfrom another experiment(train-less code)
% tmp = load(fullfile(path_DB_process,'feat_set_from_trainless',name_DB_from_other_exp));
% tmp_name = fieldnames(tmp);
% feat_DB = getfield(tmp,tmp_name{1}); %#ok<GFLD>
%-------------------------------------------------------------------------%

%-----------------------experiment information----------------------------%
% DB to be analyzed
[n_seg, n_feat, n_FE, n_trl, n_sub , n_emg_pair] = size(feat); 

% DB to be used from other experiment
% n_sub_DB = size(feat_DB,5); % Database

% trigger singals corresponding to each facial expression(emotion)
name_trg = {"angry(eye brow wrinkled, lip tightening)",...
    1;"어금니깨물기(teeth clenching)",2;"비웃음(왼쪽)",3;"비웃음(오른쪽)",4;...
    "눈 세게 감기(nose wrinkled)",5;"두려움(lip strecher)",...
    6;"행복(lip corner)",7;"키스(lip kiss)",8;"무표정",9;"슬픔(lip protrude",...
    10;"놀람(eye brow up, lip open)",11};
name_FE = name_trg(:,1);
idx_trg = cell2mat(name_trg(:,2));
clear name_trg;

% get name list of subjects
[name_subject,~] = read_names_of_file_in_folder(path_DB_raw);

% check numeber of subject from DB processed and DB raw
if length(name_subject) ~= n_sub
    error('please first do analysis of raw DB')
end
    
% name of classfier to use in facial unit
% eyebrow: angry(1),fear(6), neutral(9), sadness(10), surprised(11)
% 6번과 10번의 경우 둘중 1개로 분류되면 아바타는 동일한 표정으로 합성
% zygomatric: frown(3), contemptuous(left)(4), contemptuous(right)(5),
% happy(7), 
% lips : angry(lip tightening) (1), clench(2), fear(lip_stretch_down)(6)
% kiss(8), surprise(open)(11)
name_classifier = {'cf_eyebrow','cf_cheeck','cf_lipshapes'};
n_cf = length(name_classifier);
idx_FE2classfy = {[1,6,9,10,11],[3,9],[1,2,3,4,6,7,8,11]};

name_FE2classfy = cell(n_cf,1);
for i_cf = 1 : n_cf
    name_FE2classfy{i_cf} = name_FE(idx_FE2classfy{i_cf});
end
idx_sub = 1 : n_sub;
idx_trl = 1 : n_trl;

name_gesture_clfr = cell(n_cf,1);
name_gesture_clfr{1} = {'eye_brow_down','eye_brow_sad','neutral',...
    'eye_brow_sad','eye_brow_up'};
name_gesture_clfr{2} = {'neutral','neutral'};
name_gesture_clfr{3} = {'lip_tighten','clench','lip_corner_up_left',...
    'lip_corner_up_right','lip_stretch_down',...
    'lip_corner_up_both','kiss','lip_open'};

% name_gesture_list{1} = {'eye_brow_down','neutral','lip_tighten'};
% name_gesture_list{2} = {'neutral','neutral','clench'};
% name_gesture_list{3} = {'neutral','neutral','lip_corner_up_left'};
% name_gesture_list{4} = {'neutral','neutral','contemptuous_right'};
% name_gesture_list{5} = {'neutral','nose_wrinkle','lip_corner_up_both'};
% name_gesture_list{6} = {'neutral','neutral','lip_corner_up_left'};
% name_gesture_list{7} = {'neutral','neutral','lip_corner_up_left'};
% name_gesture_list{8} = {'neutral','neutral','lip_corner_up_left'};
% name_gesture_list{9} = {'neutral','neutral','lip_corner_up_left'};
% name_gesture_list{10} = {'neutral','neutral','lip_corner_up_left'};
% name_gesture_list{11} = {'neutral','neutral','lip_corner_up_left'};


% feature indexing when using DB of ch4 ver
idx_feat.RMS = 1:4;
idx_feat.WL = 5:8;
idx_feat.SampEN = 9:12;
idx_feat.CC = 13:28;
n_feat = 28;
n_sub_compared = n_sub - 1;

% feat names and indices
name_feat = fieldnames(idx_feat);
idx_feat = struct2cell(idx_feat);
n_ftype = length(name_feat);

% channel indexing of features
idx_ch.RZ = 1:4:n_feat;
idx_ch.RF = 2:4:n_feat;
idx_ch.LF = 3:4:n_feat;
idx_ch.LZ = 4:4:n_feat;

idx_Zy = cellfun(@isempty,strfind(fieldnames(idx_ch),'Z')); %#ok<STRCLFH>
idx_Fr = cellfun(@isempty,strfind(fieldnames(idx_ch),'F')); %#ok<STRCLFH>

idx_ch = struct2cell(idx_ch);

idx_ch_zy = cat(2,idx_ch{idx_Zy});
idx_ch_fr = cat(2,idx_ch{idx_Fr});

% {'cf_eyebrow','cf_cheeck','cf_lipshapes'};
idx_ch_FE2classfy{1} = idx_ch_fr;
idx_ch_FE2classfy{2} = idx_ch_zy;
idx_ch_FE2classfy{3} = idx_ch_zy;

% idx_ch_FE2classfy{1} = 1:28;
% idx_ch_FE2classfy{2} = 1:28;
% idx_ch_FE2classfy{3} = 1:28;
%-------------------------------------------------------------------------%

%----------------------set saving folder----------------------------------%
% set folder for saving
name_folder_saving = [id_att_compare,'_',id_DBtype,'_n_trans_',...
    num2str(n_transforemd)];

% set saving folder for windows
path_saving = make_path_n_retrun_the_path(path_DB_analy,name_folder_saving);
%-------------------------------------------------------------------------%

%----------------------memory allocation for results----------------------%
% memory allocatoin for accurucies
r.acc = zeros(n_seg,n_trl,n_sub,n_transforemd+1,n_emg_pair,n_cf);

% memory allocatoin for output and target
r.output_n_target = cell(n_seg,n_trl,n_sub,n_transforemd+1,n_emg_pair,n_cf);    

% memory allocatoin for output and target
r.model= cell(n_trl,n_sub,n_transforemd+1,n_emg_pair,n_cf);    

%-------------------------------------------------------------------------%

%------------------------------------main---------------------------------%

% get accrucies and output/target (for confusion matrix) with respect to
% subject, trial, number of segment, FE,
for i_emg_pair = 1 : n_emg_pair
for i_sub = 1 : n_sub
for i_trl = 1 : n_trl
    %display of subject and trial in progress
    fprintf('i_emg_pair:%d i_sub:%d i_trial:%d\n',i_emg_pair,i_sub,i_trl);

    if n_transforemd>=1
    % memory allocation similarily transformed feature set
    feat_t = cell(n_seg,n_FE);

    for i_seg = 1 : n_seg
        for i_FE = 1 : n_FE

            % memory allocation feature set from other experiment
            feat_t{i_seg,i_FE} = cell(1,n_ftype);

            % you should get access to DB of other experiment with each
            % features
            for i_FeatName = 1 : n_ftype

                % number of feature of each type
                n_feat_each = length(idx_feat{i_FeatName});

                % feat from this experiment
                feat_ref = feat(i_seg,idx_feat{i_FeatName} ,i_FE,...
                    i_trl,i_sub,i_emg_pair)';

            switch id_DBtype
            case 'DB_own'
                %---------feat to be compared from this experiment----%
                % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30, n_emg_pair:3]

                % compare ohter subject except its own subject
                idx_sub_compared = countmember(idx_sub,i_sub)==0;
                switch id_att_compare
                case 'all'
                    feat_compare = feat(:,idx_feat{i_FeatName},...
                        :,:,idx_sub_compared,i_emg_pair);

                case 'Only_Seg'
                    feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                        :,:,idx_sub_compared,i_emg_pair);

                case 'Seg_FE'
                    feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                    i_FE,:,idx_sub_compared,i_emg_pair);
                end

                % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
                feat_compare = permute(feat_compare,[2 3 4 5 1]);

                %  size(2):FE, size(5):seg
                feat_compare = reshape(feat_compare,...
                    [n_feat_each, size(feat_compare,2)*n_trl*n_sub_compared*...
                    size(feat_compare,5)]);

                % get similar features by determined number of
                % transformed DB
                feat_t{i_seg,i_FE}{i_FeatName} = ...
                    dtw_search_n_transf(feat_ref, feat_compare, n_transforemd)';
                %-----------------------------------------------------%

            case 'DB_other'
                 %---------feat to be compared from other experiment---%
                % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
                switch id_att_compare
                case 'all'
                    feat_compare_DB = feat_DB(:,idx_feat{i_FeatName},:,:,:);

                case 'Only_Seg'
                    feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},:,:,:);

                case 'Seg_FE'
                    feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},i_FE,:,:);
                end

                % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
                feat_compare_DB = permute(feat_compare_DB,[2 3 4 5 1]);

                %  size(2):FE, size(5):seg
                feat_compare_DB = reshape(feat_compare_DB,...
                    [n_feat_each, size(feat_compare_DB,2)*n_trl*n_sub_DB*...
                    size(feat_compare_DB,5)]);

                % get similar features by determined number of
                % transformed DB
                feat_t{i_seg,i_FE}{i_FeatName} = ...
                    dtw_search_n_transf(feat_ref, feat_compare_DB, n_transforemd)';
                %-----------------------------------------------------%

            case 'DB_both'
                %---------feat to be compared from both experiment---%
                % compare ohter subject except its own subject in this
                % experiment
                idx_sub_compared = countmember(idx_sub,i_sub)==0;
                % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
                switch id_att_compare
                case 'all'
                    feat_compare = feat(:,idx_feat{i_FeatName},...
                        :,:,idx_sub_compared,i_emg_pair);
                    feat_compare_DB = feat_DB(:,idx_feat{i_FeatName},:,:,:);
                case 'Only_Seg'
                    feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                        :,:,idx_sub_compared,i_emg_pair);
                    feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},:,:,:);

                case 'Seg_FE'
                    feat_compare = feat(i_seg,idx_feat{i_FeatName},...
                    i_FE,:,idx_sub_compared,i_emg_pair);
                    feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},i_FE,:,:);
                end
                % concatinating both DB
                feat_both_DB = cat(5,feat_compare,feat_compare_DB);
                % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
                feat_both_DB = permute(feat_both_DB,[2 3 4 5 1]);

                %  size(2):FE, size(4):sub, size(5):seg
                feat_both_DB = reshape(feat_both_DB,...
                    [n_feat_each, size(feat_both_DB,2)*n_trl*...
                    size(feat_both_DB,4)*...
                    size(feat_both_DB,5)]);

                % get similar features by determined number of
                % transformed DB
                feat_t{i_seg,i_FE}{i_FeatName} = ...
                    dtw_search_n_transf(feat_ref, feat_both_DB, n_transforemd)';
                %-----------------------------------------------------%
            end
            end
        end
    end

    % arrange feat transformed and target
    % concatinating features with types
    feat_t = cellfun(@(x) cat(2,x{:}),feat_t,'UniformOutput',false);
    end
    % validate with number of transformed DB
    for n_t = 0: n_transforemd
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
        target_feat_trans = repmat(1:n_FE,sum(size_temp,1),1);
        target_feat_trans = target_feat_trans(:); 
        else
        feat_trans = [];    
        target_feat_trans = [];
        end

        % feat for anlaysis
        feat_ref = reshape(permute(feat(:,:,:,i_trl,i_sub,i_emg_pair),...
            [1 3 2]),[n_seg*n_FE,n_feat]);
        target_feat_ref = repmat(1:n_FE,n_seg,1);
        target_feat_ref = target_feat_ref(:);

        % get input and targets for train DB
        input_train = cat(1,feat_ref,feat_trans);
        target_train = cat(1,target_feat_ref,target_feat_trans);

        % get input and targets for test DB
        input_test = reshape(permute(feat(:,: ,:,idx_trl~=i_trl,...
            i_sub,i_emg_pair),[1 4 3 2]),[n_seg*(n_trl-1)*n_FE,n_feat]);
        target_test = repmat(1:n_FE,n_seg*(n_trl-1),1);
        target_test = target_test(:);

        % get features of determined emotions that you want to classify
        for i_cf = 1 : n_cf
            n_FE2classfy = length(idx_FE2classfy{i_cf});
            idx_train_samples_2_classify = countmember(target_train,idx_FE2classfy{i_cf})==1;
            idx_test_samples_2_classify = countmember(target_test,idx_FE2classfy{i_cf})==1;
            
%             input_train = input_train(idx_train_samples_2_classify,idx_ch_FE2classfy{i_cf});
%             target_train = target_train(idx_train_samples_2_classify,idx_ch_FE2classfy{i_cf});

            
%             input_test = input_test(idx_test_samples_2_classify,:);
    %             target_test = target_test(idx_test_samples_2_classify,:);

            % train
            model.lda = fitcdiscr(...
                input_train(idx_train_samples_2_classify,idx_ch_FE2classfy{i_cf}),...
                target_train(idx_train_samples_2_classify));

            % test
            output_test = predict(model.lda,...
                input_test(idx_test_samples_2_classify,idx_ch_FE2classfy{i_cf}));

            r.model{i_trl,i_sub,n_t+1,i_emg_pair,i_cf} = model.lda;
            % reshape ouput_test as <seg, trl, FE>
            output_test = reshape(output_test,[n_seg,(n_trl-1),n_FE2classfy]);
            output_mv_test = majority_vote(output_test,idx_FE2classfy{i_cf});

            % reshape target test for acc caculation
            target_test4acc = repmat(idx_FE2classfy{i_cf},(n_trl-1),1);
            target_test4acc = target_test4acc(:);
            for i_seg = 1 : n_seg
                ouput_seg = output_mv_test(i_seg,:)';
%                 disp(ouput_seg);
                r.acc(i_seg,i_trl,i_sub,n_t+1,i_emg_pair,i_cf) = ...
                    sum(target_test4acc==ouput_seg)/(n_FE2classfy*(n_trl-1))*100;
                r.output_n_target{i_seg,i_trl,i_sub,n_t+1,i_emg_pair,i_cf} = ...
                    [ouput_seg,target_test4acc];
            end
        end
        
        %------------------------make avartar-----------------------------%
        % get input and targets for test DB
        i_seg = 15;
        for i_trl_test = find(idx_trl~=i_trl)
            tmp = feat(:,:,:,i_trl_test,i_sub,i_emg_pair);
            for i_fe = 1 : n_FE
                name_output_clfr = cell(n_cf,1);
                for i_cf = 1 : n_cf
                    % test
                    output_test = predict(r.model{i_trl,i_sub,n_t+1,i_emg_pair,i_cf},...
                    tmp(:,idx_ch_FE2classfy{i_cf},i_fe));
            
                    % reshape ouput_test as <seg, trl, FE>
                    output_mv_test = majority_vote(output_test,idx_FE2classfy{i_cf});
                    
                    % get output using number of segments you want
                    idx_output2clfr = ...
                        find(countmember(idx_FE2classfy{i_cf},output_mv_test(i_seg))==1);
                    name_output_clfr{i_cf} = name_gesture_clfr{i_cf}...
                        {idx_output2clfr};
                end
                 % composing avartar by classfied facial unit
                 plot_avartar(name_output_clfr{1},name_output_clfr{2},...
                     name_output_clfr{3})
                 title(name_FE{i_fe})
                 set(gcf,'Position',[1162 348 560 420]);
            end
        end
        %-----------------------------------------------------------------%
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

%------------------------------results of plot----------------------------%

% tmp = mean(mean(r.acc(15,:,:,1,:,:),2),3);
% tmp = reshape(tmp,n_emg_pair,n_cf);
% bar(tmp);
% mean(tmp,2)
for n_t = 0
for i_emg_pair = 1
for i_cf = 1 : n_cf
    for i_seg = 15
        tmp = cat(1,r.output_n_target{i_seg,:,:,n_t+1,i_emg_pair,i_cf});
        output_tmp = full(ind2vec(tmp(:,1)'));
        target_tmp = full(ind2vec(tmp(:,2)'));

        tmp = countmember(1:max(idx_FE2classfy{i_cf}),idx_FE2classfy{i_cf})==0;
        output_tmp(tmp,:) = [];
        target_tmp(tmp,:) = [];

        [~,mat_conf,idx_of_samps_with_ith_target,~] = ...
            confusion(target_tmp,output_tmp);
        figure;
        title(sprintf('n_t %d emg_pair %d i_cf %d i_seg %d',...
            n_t,i_emg_pair,i_cf,i_seg));
        plotConfMat(mat_conf, name_FE2classfy{i_cf})
    end
end
end
end
%-------------------------------------------------------------------------%


% data structure of accuracy: [i_seg,i_trl,i_sub,n_t+1]
% save(fullfile(path_saving,'result'),'acc_total','r_total');
% % plot with emg_pair
% for i_emg_pair = 1 : n_emg_pair
%     tmp = r_total{i_emg_pair};
%     tmp = permute(mean(mean(tmp.acc,2),3),[1 4 2 3]);
%     figure;
%     plot(tmp)
% end
% 
% % plot confusion matrix with specific subejct of a emg pair
% for i_emg_pair = 1 : n_emg_pair
%     tmp = r_total{i_emg_pair};
%     tmp = tmp.output_n_target(15,:,2,:);
%     tmp = cat(1,tmp{:});
%     
%     output_tmp = full(ind2vec(tmp(:,1)'));
%     target_tmp = full(ind2vec(tmp(:,2)'));
%     
%     tmp = countmember(1:max(idx_FE2classfy),idx_FE2classfy)==0;
%     output_tmp(tmp,:) = [];
%     target_tmp(tmp,:) = [];
%     
%     [~,mat_conf,idx_of_samps_with_ith_target,~] = ...
%         confusion(target_tmp,output_tmp);
%     figure(i_emg_pair);
%     plotConfMat(mat_conf, name_FE2classfy)
% %     mat_n_samps = cellfun(@(x) size(x,2),idx_of_samps_with_ith_target);
% %     mat_n_samps(logical(eye(size(mat_n_samps)))) = 0;
% %     fn_sum_of_each_class = sum(mat_n_samps,1);
% end
% 
% 
% for i_emg_pair = 1 : n_emg_pair
%     tmp = r_total{i_emg_pair};
%     tmp = permute(mean(tmp.acc(15,:,:,:),2),[3 4 1 2]);
%     figure;
%     bar(tmp)
%     mean(tmp)
% end
% %-------------------------------------------------------------------------%
% 
% 



% 
% 
% %-------------------------------------------------------------------------%
% % 1. feat_extraction.m
% % 2. classficiation_using_DB.m  %---current code---%
% %-------------------------------------------------------------------------%
% % developed by Ho-Seung Cha, Ph.D Student,
% % CONE Lab, Biomedical Engineering Dept. Hanyang University
% % under supervison of Prof. Chang-Hwan Im
% % All rights are reserved to the author and the laboratory
% % contact: hoseungcha@gmail.com
% %-------------------------------------------------------------------------%
% clc; close all; clear all;
% 
% %-----------------------Code anlaysis parmaters----------------------------
% % name of raw DB
% name_DB_raw = 'DB_raw2';
% 
% % name of process DB to analyze in this code
% name_DB_process = 'DB_processed2';
% 
% % name of folder of DB anlaysis
% name_DB_analy = 'feat_set_n_sub_16_n_seg_30_n_wininc_204_winsize_204';
% 
% 
% % load feature set, which was extracted by feat_extration.m
% name_feat_file = 'feat_set_seg_30';
% 
% % load feature set, from different DB, which was extracted by
% % feat_extraion.m in git_EMG_train_less_FE_recog
% name_other_DB = 'feat_set_combined_seg_30_using_ch4';
% 
% % decide if you applied train-less algorithm using DB, or DB of other expreiment
% id_DBtype = 'DB_both'; % 'DB_own' , 'DB_other' , 'DB_both'
% 
% % decide expression to classify
% % % ["화남(1)";"비웃음(2)";"역겨움(3)";"두려움(4)";"행복(5)";"무표정(6)";"슬픔(7)";
% % % "놀람(8)"];
% % % idx_FE2classfy =[1,5,7,8];% 화남, 놀람, 행복(기쁨), 슬픔 <교수님 선정>
% % idx_FE2classfy =[1,3,5,8];% 화남, 놀람, 행복(기쁨), 역겨움 <교수님 선정>
% 
% % decide number of tranfored feat from DB 
% n_transforemd = 0;
% 
% % decide which attibute to be compared when applying train-less algoritm
% % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
% % 'all' : [:,:,:,:,:], 'Only_Seg' : [i_seg,:,:,:,:], 'Seg_FE' : [i_seg,:,i_FE,:,:]
% id_att_compare = 'Only_Seg'; % 'all', 'Only_Seg', 'Seg_FE'
% %-------------------------------------------------------------------------%
% 
% %-------------set paths in compliance with Cha's code structure-----------%
% 
% % path of research, which contains toolbox
% path_research = fileparts(fileparts(fileparts(fullfile(cd))));
% 
% % path of code, which 
% path_code = fileparts(fullfile(cd));
% path_DB = fullfile(path_code,'DB');
% path_DB_raw = fullfile(path_DB,name_DB_raw);
% path_DB_process = fullfile(path_DB,name_DB_process);
% path_DB_analy = fullfile(path_DB_process,name_DB_analy);
% %-------------------------------------------------------------------------%
% 
% 
% % get toolbox
% addpath(genpath(fullfile(fileparts(fileparts(fileparts(cd))),'_toolbox')));
% 
% % add functions
% addpath(genpath(fullfile(cd,'functions')));
% 
% % path for processed data
% path_research=fileparts(pwd); % parent path which has DB files
% 
% % get path
% path_DB_process = fullfile(path_research,'DB','DB_processed');
% path_DB_raw = fullfile(path_research,'DB','DB_raw');
% % load feature set, from this experiment 
% tmp = load(fullfile(path_DB_process,name_feat_file)); 
% tmp_name = fieldnames(tmp);
% feat = getfield(tmp,tmp_name{1}); %#ok<GFLD>
% 
% % load feature setfrom another experiment(train-less code)
% tmp = load(fullfile(path_DB_process,'feat_set_from_trainless',name_other_DB));
% tmp_name = fieldnames(tmp);
% feat_DB = getfield(tmp,tmp_name{1}); %#ok<GFLD>
% 
% %-----------------------experiment information----------------------------%
% [n_seg, n_feat, n_FE, n_trl, n_sub , n_emg_pair] = size(feat); % DB to be analyzed
% name_FE = ["화남";"비웃음";"역겨움";"두려움";"행복";"무표정";"슬픔";"놀람"];
% n_FE2classfy = length(idx_FE2classfy);
% name_FE2classfy = name_FE(idx_FE2classfy);
% n_sub_DB = size(feat_DB,5); % Database
% idx_sub = 1 : n_sub;
% idx_trl = 1 : n_trl;
% 
% %
% [name_subject,~] = read_names_of_file_in_folder(path_DB_raw);
% 
% % feature indexing when using DB of ch4 ver
% idx_feat.RMS = 1:4;
% idx_feat.WL = 5:8;
% idx_feat.SampEN = 9:12;
% idx_feat.CC = 13:28;
% n_feat = 28;
% n_sub_compared = n_sub - 1;
% % feat names and indices
% name_feat = fieldnames(idx_feat);
% idx_feat = struct2cell(idx_feat);
% n_ftype = length(name_feat);
% 
% %-------------------------------------------------------------------------%
% 
% % memory allocation for reults
% r_total = cell(n_emg_pair,1);
% 
% % get accrucies and output/target (for confusion matrix) with respect to
% % subject, trial, number of segment, FE,
% for i_emg_pair = 1 : n_emg_pair
%     
% % memory allocatoin for accurucies
% r.acc = zeros(n_seg,n_trl,n_sub,n_transforemd+1);
% 
% % memory allocatoin for output and target
% r.output_n_target = cell(n_seg,n_trl,n_sub,n_transforemd+1);    
% 
% for i_sub = 1 : n_sub
%     for i_trl = 1 : n_trl
%         
%         %display of subject and trial in progress
%         fprintf('i_sub:%d i_trial:%d\n',i_sub,i_trl);
% 
%         if n_transforemd>=1
%         % memory allocation similarily transformed feature set
%         feat_t = cell(n_seg,n_FE);
%         
%         for i_seg = 1 : n_seg
%             for i_FE = 1 : n_FE
% 
%                 % memory allocation feature set from other experiment
%                 feat_t{i_seg,i_FE} = cell(1,n_ftype);
%                 
%                 % you should get access to DB of other experiment with each
%                 % features
%                 for i_FeatName = 1 : n_ftype
%                     
%                     % number of feature of each type
%                     n_feat_each = length(idx_feat{i_FeatName});
%                     
%                     % feat from this experiment
%                     feat_ref = feat(i_seg,idx_feat{i_FeatName} ,i_FE,...
%                         i_trl,i_sub,i_emg_pair)';
%                     
%                 switch id_DBtype
%                 case 'DB_own'
%                     %---------feat to be compared from this experiment----%
%                     % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30, n_emg_pair:3]
% 
%                     % compare ohter subject except its own subject
%                     idx_sub_compared = countmember(idx_sub,i_sub)==0;
%                     switch id_att_compare
%                     case 'all'
%                         feat_compare = feat(:,idx_feat{i_FeatName},...
%                             :,:,idx_sub_compared,i_emg_pair);
% 
%                     case 'Only_Seg'
%                         feat_compare = feat(i_seg,idx_feat{i_FeatName},...
%                             :,:,idx_sub_compared,i_emg_pair);
% 
%                     case 'Seg_FE'
%                         feat_compare = feat(i_seg,idx_feat{i_FeatName},...
%                         i_FE,:,idx_sub_compared,i_emg_pair);
%                     end
%                     
%                     % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
%                     feat_compare = permute(feat_compare,[2 3 4 5 1]);
%                     
%                     %  size(2):FE, size(5):seg
%                     feat_compare = reshape(feat_compare,...
%                         [n_feat_each, size(feat_compare,2)*n_trl*n_sub_compared*...
%                         size(feat_compare,5)]);
%                     
%                     % get similar features by determined number of
%                     % transformed DB
%                     feat_t{i_seg,i_FE}{i_FeatName} = ...
%                         dtw_search_n_transf(feat_ref, feat_compare, n_transforemd)';
%                     %-----------------------------------------------------%
%                     
%                     
%                    
%                 case 'DB_other'
%                      %---------feat to be compared from other experiment---%
%                     % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
%                     switch id_att_compare
%                     case 'all'
%                         feat_compare_DB = feat_DB(:,idx_feat{i_FeatName},:,:,:);
% 
%                     case 'Only_Seg'
%                         feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},:,:,:);
% 
%                     case 'Seg_FE'
%                         feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},i_FE,:,:);
%                     end
%                     
%                     % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
%                     feat_compare_DB = permute(feat_compare_DB,[2 3 4 5 1]);
%                     
%                     %  size(2):FE, size(5):seg
%                     feat_compare_DB = reshape(feat_compare_DB,...
%                         [n_feat_each, size(feat_compare_DB,2)*n_trl*n_sub_DB*...
%                         size(feat_compare_DB,5)]);
%                     
%                     % get similar features by determined number of
%                     % transformed DB
%                     feat_t{i_seg,i_FE}{i_FeatName} = ...
%                         dtw_search_n_transf(feat_ref, feat_compare_DB, n_transforemd)';
%                     %-----------------------------------------------------%
%                     
%                 case 'DB_both'
%                     %---------feat to be compared from both experiment---%
%                     % compare ohter subject except its own subject in this
%                     % experiment
%                     idx_sub_compared = countmember(idx_sub,i_sub)==0;
%                     % [n_seg:30, n_feat:28, n_FE:8, n_trl:20, n_sub:30]
%                     switch id_att_compare
%                     case 'all'
%                         feat_compare = feat(:,idx_feat{i_FeatName},...
%                             :,:,idx_sub_compared,i_emg_pair);
%                         feat_compare_DB = feat_DB(:,idx_feat{i_FeatName},:,:,:);
%                     case 'Only_Seg'
%                         feat_compare = feat(i_seg,idx_feat{i_FeatName},...
%                             :,:,idx_sub_compared,i_emg_pair);
%                         feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},:,:,:);
% 
%                     case 'Seg_FE'
%                         feat_compare = feat(i_seg,idx_feat{i_FeatName},...
%                         i_FE,:,idx_sub_compared,i_emg_pair);
%                         feat_compare_DB = feat_DB(i_seg,idx_feat{i_FeatName},i_FE,:,:);
%                     end
%                     % concatinating both DB
%                     feat_both_DB = cat(5,feat_compare,feat_compare_DB);
%                     % permutation giving [n_feat, n_FE, n_trl, n_sub ,n_seg]
%                     feat_both_DB = permute(feat_both_DB,[2 3 4 5 1]);
%                     
%                     %  size(2):FE, size(4):sub, size(5):seg
%                     feat_both_DB = reshape(feat_both_DB,...
%                         [n_feat_each, size(feat_both_DB,2)*n_trl*...
%                         size(feat_both_DB,4)*...
%                         size(feat_both_DB,5)]);
%                     
%                     % get similar features by determined number of
%                     % transformed DB
%                     feat_t{i_seg,i_FE}{i_FeatName} = ...
%                         dtw_search_n_transf(feat_ref, feat_both_DB, n_transforemd)';
%                     %-----------------------------------------------------%
%                 end
%                 end
%             end
%         end
%         
%         % arrange feat transformed and target
%         % concatinating features with types
%         feat_t = cellfun(@(x) cat(2,x{:}),feat_t,'UniformOutput',false);
%         end
%         % validate with number of transformed DB
%         for n_t = 0: n_transforemd
%             if n_t >= 1
%             % get feature-transformed with number you want
%             feat_trans = cellfun(@(x) x(1:n_t,:),feat_t,...
%                 'UniformOutput',false);
%             
%             % get size to have target
%             size_temp = cell2mat(cellfun(@(x) size(x,1),...
%                 feat_trans(:,1),'UniformOutput',false));
%             
%             % feature transformed 
%             feat_trans = cell2mat(feat_trans(:));
%             
%             % target for feature transformed 
%             target_feat_trans = repmat(1:n_FE,sum(size_temp,1),1);
%             target_feat_trans = target_feat_trans(:); 
%             else
%             feat_trans = [];    
%             target_feat_trans = [];
%             end
%             
%             % feat for anlaysis
%             feat_ref = reshape(permute(feat(:,:,:,i_trl,i_sub,i_emg_pair),...
%                 [1 3 2]),[n_seg*n_FE,n_feat]);
%             target_feat_ref = repmat(1:n_FE,n_seg,1);
%             target_feat_ref = target_feat_ref(:);
%             
%             % get input and targets for train DB
%             input_train = cat(1,feat_ref,feat_trans);
%             target_train = cat(1,target_feat_ref,target_feat_trans);
% 
%             % get input and targets for test DB
%             input_test = reshape(permute(feat(:,: ,:,idx_trl~=i_trl,...
%                 i_sub,i_emg_pair),[1 4 3 2]),[n_seg*(n_trl-1)*n_FE,n_feat]);
%             target_test = repmat(1:n_FE,n_seg*(n_trl-1),1);
%             target_test = target_test(:);
%             
%             % get features of determined emotions that you want to classify
%             idx_train_samples_2_classify = countmember(target_train,idx_FE2classfy)==1;
%             input_train = input_train(idx_train_samples_2_classify,:);
%             target_train = target_train(idx_train_samples_2_classify,:);
%             
%             idx_test_samples_2_classify = countmember(target_test,idx_FE2classfy)==1;
%             input_test = input_test(idx_test_samples_2_classify,:);
%             target_test = target_test(idx_test_samples_2_classify,:);
%             
%             % train
%             model.lda = fitcdiscr(input_train,target_train);
%             
%             % test
%             output_test = predict(model.lda,input_test);
%             
%             % reshape ouput_test as <seg, trl, FE>
%             output_test = reshape(output_test,[n_seg,(n_trl-1),n_FE2classfy]);
%             output_mv_test = majority_vote(output_test,idx_FE2classfy);
%             
%             % reshape target test for acc caculation
%             target_test = repmat(idx_FE2classfy,(n_trl-1),1);
%             target_test = target_test(:);
%             for i_seg = 1 : n_seg
%                 ouput_seg = output_mv_test(i_seg,:)';
%                 r.acc(i_seg,i_trl,i_sub,n_t+1,i_emg_pair,i_cf) = ...
%                     sum(target_test==ouput_seg)/(n_FE2classfy*(n_trl-1))*100;
%                 r.output_n_target{i_seg,i_trl,i_sub,n_t+1,i_emg_pair,i_cf} = ...
%                     [ouput_seg,target_test];
%             end
%         end
%     end
% end
% end
% 
% tmp =struct2cell(cell2mat(r_total));
% tmp = tmp(1,:);
% acc_total = cellfun(@(x) permute(mean(mean(x,2),3),[1 4 2 3]),tmp,'UniformOutput',false);
% acc_total = cat(2,acc_total{:});
% 
% % set folder for saving
% name_folder_saving = ['Result_',name_feat_file,'_',name_other_DB,'_',...
%     id_att_compare,'_',id_DBtype,'_n_trans_',num2str(n_transforemd),'_',...
%     cat(2,name_FE2classfy{:})];
% 
% % set saving folder for windows
% path_saving = make_path_n_retrun_the_path(path_DB_process,name_folder_saving);
% 
% %---------------------------------save emg_seg----------------------------%
% % data structure of accuracy: [i_seg,i_trl,i_sub,n_t+1]
% save(fullfile(path_saving,'result'),'acc_total','r_total');
% % plot with emg_pair
% for i_emg_pair = 1 : n_emg_pair
%     tmp = r_total{i_emg_pair};
%     tmp = permute(mean(mean(tmp.acc,2),3),[1 4 2 3]);
%     figure;
%     plot(tmp)
% end
% 
% 
% % plot confusion matrix with specific subejct of a emg pair
% for i_emg_pair = 1 : n_emg_pair
%     tmp = r_total{i_emg_pair};
%     tmp = tmp.output_n_target(15,:,2,:);
%     tmp = cat(1,tmp{:});
%     
%     output_tmp = full(ind2vec(tmp(:,1)'));
%     target_tmp = full(ind2vec(tmp(:,2)'));
%     
%     tmp = countmember(1:max(idx_FE2classfy),idx_FE2classfy)==0;
%     output_tmp(tmp,:) = [];
%     target_tmp(tmp,:) = [];
%     
%     [~,mat_conf,idx_of_samps_with_ith_target,~] = ...
%         confusion(target_tmp,output_tmp);
%     figure(i_emg_pair);
%     plotConfMat(mat_conf, name_FE2classfy)
% %     mat_n_samps = cellfun(@(x) size(x,2),idx_of_samps_with_ith_target);
% %     mat_n_samps(logical(eye(size(mat_n_samps)))) = 0;
% %     fn_sum_of_each_class = sum(mat_n_samps,1);
% end
% 
% 
% for i_emg_pair = 1 : n_emg_pair
%     tmp = r_total{i_emg_pair};
%     tmp = permute(mean(tmp.acc(15,:,:,:),2),[3 4 1 2]);
%     figure;
%     bar(tmp)
%     mean(tmp)
% end
% 
% 
% 
% 
% %-------------------------------------------------------------------------%
% 
% 
% % save at directory of DB\dist
% %             save(fullfile(path_made,['T_',num2str(i_sub),'_',...
% %                 num2str(i_trial),'_',names_feat{i_FeatName},'_5.mat']),'T');
% 
% % function [xt] = dtw_search_n_transf(x1, x2, N_s)
% % % parameters
% % window_width = 3;
% % max_slope_length = 2;
% % speedup_mode = 1;
% % DTW_opt.nDivision_4Resampling = 10;
% % DTW_opt.max_slope_length = 3;
% % DTW_opt.speedup_mode = 1;
% % 
% % [N_f, N]= size(x2); dist = zeros(N,1);
% % for i = 1 : N
% %     dist(i) = fastDTW(x1, x2(:,i),max_slope_length, ...
% %         speedup_mode, window_width );
% % end
% % % Sort
% % [~, sorted_idx] = sort(dist);
% % % xs= x2(:,sorted_idx(1:N_s));
% % xt = zeros(N_f,N_s);
% % for i = 1 : N_s
% %     xt(:,i)= transfromData_accRef_usingDTW(x2(:,sorted_idx(i)), x1, DTW_opt);
% % end
% % end
% 
% 
% 
% 
% 
% 
% 
% %------------------------code analysis parameter--------------------------%
% % name of raw DB
% name_DB_raw = 'DB_raw2';
% 
% % name of process DB to analyze in this code
% name_DB_process = 'DB_processed2';
% 
% % name of folder of DB anlaysis
% name_DB_analy = 'feat_set_n_sub_16_n_seg_30_n_wininc_204_winsize_204';
% 
% % number of transformed data 
% %-------------------------------------------------------------------------%
% 
% 
% %-------------------------add functions-----------------------------------%
% % get tool box
% addpath(genpath(fullfile(path_research,'_matlab_toolbox')));
% addpath(genpath(fullfile(cd,'functions')));
% %-------------------------------------------------------------------------%
% 
% %------------------------experiment infromation---------------------------%
% % load DB processed
% load(fullfile(path_DB_analy,'feat_set'));
% 
% %-------------------------------------------------------------------------%
% 
% %------------------------experiment infromation---------------------------%
% % DB to be analyzed
% [n_seg, n_feat, n_FE, n_trial, n_sub , n_emg_pair] = size(feat); 
% n_sub = size(feat,5); % Database
% idx_sub = 1 : n_sub;
% idx_trl = 1 : n_trial;
% 
% 
% 
% 
% %-------------------------------------------------------------------------%
% 
% %----------------------set saving folder----------------------------------%
% 
% % makeing folder for results 결과 저장 폴더 설정
% % folder_name2make = ['T5_',name_feat_file]; % 폴더 이름
% % path_made = make_path_n_retrun_the_path(fullfile(parentdir,...
% %     'DB','dist'),folder_name2make)
% %-------------------------------------------------------------------------%
% 
% 
% %----------------------memory allocation for results----------------------%
% result.acc = zeros(n_seg,n_trial,n_sub,n_emg_pair,n_cf,n_transforemd+1);
% result.output_n_target = cell(n_seg,n_trial,n_sub,n_emg_pair,n_cf,n_transforemd+1);   
% result.model = cell(n_trial,n_sub,n_emg_pair,n_transforemd+1);  
% %-------------------------------------------------------------------------%
% 
% %------------------------------------main---------------------------------%
% 
% % for i_emg_pair = 1 : n_emg_pair
% for i_emg_pair = 1
% for i_sub = 1 : n_sub
%     for i_trial = 1 : n_trial
%         fprintf('i_sub:%d i_trial:%d\n',i_sub,i_trial);
%         % get similar feature from DB
%         if n_transforemd~=0
%         t = cell(n_seg,n_FE);
%         for n_seg = 1 : n_seg
%             for i_FE = 1 : n_FE
%                 t{n_seg,i_FE} = cell(1,n_ftype);
%                 for i_FeatName = 1 : n_ftype
%                     % get DB with a specific feature
%                     n_feat_interested = length(idx_feat{i_FeatName});
%                     feat_ref = feat(n_seg,idx_feat{i_FeatName} ,i_FE,...
%                         i_trial,i_sub,i_emg_pair)';
%                     feat_DB = feat(:,idx_feat{i_FeatName} ,:,:,:);
% %                     feat_ref = feat(i_seg,:,i_FE,i_trial,i_sub)';
%                     feat_compr = feat_DB(n_seg,:,i_FE,:,:);
%                     feat_compr = reshape(feat_compr,...
%                         [n_feat_interested, n_trial*n_sub]);
%                     % just get 5 similar feat
%                     t{n_seg,i_FE}{i_FeatName} = ...
%                         dtw_search_n_transf(feat_ref, feat_compr, n_transforemd)';
%                 end
%             end
%         end
%         % concatinating feat with types
%         t = cellfun(@(x) cat(2,x{:}),t,'UniformOutput',false);
%         else
%         t = [];
%         end
%         % arrange feat transformed and target
%         % validate with number of transformed DB
%         for n_trans = 0: n_transforemd
%             if n_trans~=0
%             % get feature-transformed with number you want
%             feat_trans = cellfun(@(x) x(1:n_trans,:),t,...
%                 'UniformOutput',false);
%             % get size to have target
%             size_temp = cell2mat(cellfun(@(x) size(x,1),...
%                 feat_trans(:,1),'UniformOutput',false));
%             % feature transformed 
%             feat_trans = cell2mat(feat_trans(:));
%             % target for feature transformed 
%             target_feat_trans = repmat(1:n_FE,sum(size_temp,1),1);
%             target_feat_trans = target_feat_trans(:); 
%             else
%             feat_trans = [];
%             target_feat_trans = [];
%             end
%             % feat for anlaysis
%             feat_ref = reshape(permute(feat(:,:,:,i_trial,i_sub,i_emg_pair),...
%                 [1 3 2]),[n_seg*n_FE,n_feat]);
%             target_feat_ref = repmat(1:n_FE,n_seg,1);
%             target_feat_ref = target_feat_ref(:);
%             % get input and targets for train DB
%             input_train = cat(1,feat_ref,feat_trans);
%             target_train = cat(1,target_feat_ref,target_feat_trans);
%             % get input and targets for test DB
%             % [n_seg, n_feat, n_FE, n_trial, n_sub=1 , n_emg_pair=1] -->
%             % [n_seg, n_trial, n_FE, n_feat, n_sub=1 , n_emg_pair=1] -->
%             % [n_seg*n_trial*n_FE, n_feat]
%             input_test = reshape(permute(feat(:,: ,:,idx_trl~=i_trial,...
%                 i_sub,i_emg_pair),[1 4 3 2]),[n_seg*(n_trial-1)*n_FE,n_feat]);
%             target_test = repmat(1:n_FE,n_seg*(n_trial-1),1);
%             target_test = target_test(:);
%             % train with three classfiers
%             model = cell(n_cf,1);
%             for i_cf = 1 : n_cf
%             % get index of train samples
%             tmp = find(countmember(target_train,idx_FE4CF{i_cf})==1);
%             n_class = length(idx_FE4CF{i_cf});
%             % train
%             model{i_cf} = fitcdiscr(input_train(tmp,:),target_train(tmp));
%             % test
%             % get index of train samples
%             tmp = find(countmember(target_test,idx_FE4CF{i_cf})==1);
%             output_test = predict(model{i_cf},input_test(tmp,:));
%             % reshape ouput_test as <seg, trl, FE>
%             output_test = reshape(output_test,[n_seg,(n_trial-1),n_class]);
%             output_mv_test = majority_vote(output_test,idx_FE4CF{i_cf});
%             % reshape target test for acc caculation
%             target_test4cf = repmat(idx_FE4CF{i_cf},(n_trial-1),1);
%             target_test4cf = target_test4cf(:);
%             for n_seg = 1 : n_seg
%                 % <N_Seg,N_trl,N_Class> -> <N_trl*N_Class,1>
%                 ouput_seg = output_mv_test(n_seg,:)';
%                 tmp = sum(target_test4cf==ouput_seg)/(n_class*(n_trial-1))*100;
% %                 disp(tmp); % dispay of acc
%                 result.acc(n_seg,i_trial,i_sub,i_emg_pair,i_cf,n_trans+1) = tmp;
%                 result.output_n_target{n_seg,i_trial,i_sub,i_emg_pair,i_cf,n_trans+1}...
%                     = [ouput_seg,target_test4cf];
%             end
%             end
%             result.model{i_trial,i_sub,i_emg_pair,n_trans+1} = model;
%         end
%     end
% end
% end
% %-------------------------------------------------------------------------%
% % save results
% % id of classfied expression
% tmp = [];
% for i = 1 : 3
% tmp = [tmp,num2str(idx_FE4CF{i}),' _ '];
% end
% tmp(isspace(tmp)) = []; % remove blanks
% tmp(end) = []; % remove end of '_'
% 
% path4result = make_path_n_retrun_the_path(path_feat,'result');
% save(fullfile(path4result,['result ',tmp]),'result');
% % plot results
% tmp = result.acc;
% % i_seg,i_trial,i_sub,i_emg_pair,i_cf,1 -->
% % mean: i_seg,1,1,i_emg_pair,i_cf,1 -->
% % i_seg,i_cf,i_emg_pair
% tmp = permute(mean(mean(tmp,2),3),[1 5 4 2 3]);
% for i_emg_pair = 1 : n_emg_pair
%     figure(i_emg_pair)
%     plot(tmp(:,:,i_emg_pair))
% end
% % analysis of confustion matrix
% for i_emg_pair = 1
% for i_cf = 1 : 3
% n_class = length(idx_FE4CF{i_cf});
% name_FE_of_class = name_trg(idx_FE4CF{i_cf},1);
% for n_seg = 15
%     % get ouputs and targets
%     tmp = result.output_n_target(n_seg,:,:,i_emg_pair,i_cf);
%     tmp = cell2mat(tmp(:));
%     output_tmp = full(ind2vec(tmp(:,1)'));
%     target_tmp = full(ind2vec(tmp(:,2)'));
%     tmp = countmember(1:max(idx_FE4CF{i_cf}),idx_FE4CF{i_cf})==0;
%     output_tmp(tmp,:) = [];
%     target_tmp(tmp,:) = [];
%     
%     [~,mat_conf,idx_of_samps_with_ith_target,~] = confusion(target_tmp,output_tmp);
%     figure(i_cf);
%     plotConfMat(mat_conf, name_FE_of_class)
%     mat_n_samps = cellfun(@(x) size(x,2),idx_of_samps_with_ith_target);
%     mat_n_samps(logical(eye(size(mat_n_samps)))) = 0;
%     fn_sum_of_each_class = sum(mat_n_samps,1);
%     
% end
% end
% end
% %-------------------------------save results------------------------------%
% save(fullfile(path_save,'results.mat'),'r','db');
% %-------------------------------------------------------------------------%
