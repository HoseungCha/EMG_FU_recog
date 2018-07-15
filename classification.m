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
name.DB_raw = 'DB_raw2';

% name of process DB to analyze in this code
name.DB_process = 'DB_processed2';

% load feature set, which was extracted by feat_extration.m
name.DB_analy = 'feat_set_DB_raw2_n_sub_41_n_seg_60_n_wininc_102_winsize_262';

% decide if validation datbase set
id_DB_val = 'myoexp2'; % myoexp1, myoexp2, both

% decide number of tranfored feat from DB
n_transforemd = 0;

% decide whether to use emg onset fueatre
id_use_emg_onset_feat = 0;

% decide which attibute to be compared when applying train-less algoritm
% 'all' : [:,:,:,:,:], 'Only_Seg' : [i_seg,:,:,:,:], 'Seg_FE' : [i_seg,:,i_FE,:,:]
id_att_compare = 'Seg_FE'; % 'all', 'Only_Seg', 'Seg_FE'
%-------------------------------------------------------------------------%

%-------------set paths in compliance with Cha's code structure-----------%
% path of research, which contains toolbox
path.research = fileparts(fileparts(fileparts(fullfile(cd))));

% path of code, which
path.code = fileparts(fullfile(cd));
path.DB = fullfile(path.code,'DB');
path.DB_raw = fullfile(path.DB,name.DB_raw);
path.DB_process = fullfile(path.DB,name.DB_process);
path.DB_analy = fullfile(path.DB_process,name.DB_analy);
%-------------------------------------------------------------------------%

%-------------------------add functions-----------------------------------%
% get toolbox
addpath(genpath(fullfile(path.research,'_toolbox')));

% add functions
addpath(genpath(fullfile(cd,'functions')));
%-------------------------------------------------------------------------%

%-----------------------------load DB-------------------------------------%
% LOAD MODEL of EMG ONSET DETECTION
load(fullfile(path.DB_process,'model_tree_emg_onset.mat'));
%-------------------------------------------------------------------------%

%-----------------------experiment information----------------------------%
% trigger singals corresponding to each facial expression(emotion)
name.emo = {'angry','clench',...
'lip_corner_up_left','lip_corner_up_right',...
'lip_corner_up_both','fear',...
'happy','kiss','neutral',...
'sad','surprised'};

name.fe = {'eye_brow_down-lip_tighten';'neutral-clench';...
'neutral-lip_corner_up_left';'neutral-lip_corner_up_right';...
'neutral-lip_corner_up_both';'eye_brow_sad-lip_stretch_down';...
'eye_brow_happy-lip_happy';'neutral-kiss';'neutral-neutral';...
'eye_brow_sad-lip_sulky';'eye_brow_up-lip_open'};

% get name list of subjects
[name.subject,~] = read_names_of_file_in_folder(path.DB_raw);

% mapped avartar gesture of each facial part of each emotion
for i = 1 : length(name.fe)
tmp = strsplit(name.fe{i},'-');
name.gesture_c1{1}{i,1} = tmp{1};
name.gesture_c1{2}{i,1} = tmp{2};
end


% name of types of features
name.feat = {'RMS';'WL';'SampEN';'CC'};
%-------------------------------------------------------------------------%

%----------------------set saving folder----------------------------------%
% set folder for saving
name.folder_saving = [id_att_compare,'_',id_DB_val,'_',num2str(n_transforemd)];

% set saving folder for windows
path_saving = make_path_n_retrun_the_path(path.DB_analy,name.folder_saving);
%-------------------------------------------------------------------------%

%---- number related parameters
n_emg_pair = 3;

%------------------------------------main---------------------------------%
% get accrucies and output/target (for confusion matrix) with respect to
% subject, trial, number of segment, FE,
for i_emg_pair = 1 : n_emg_pair
    
% load feature set, from this experiment
load(fullfile(path.DB_process,name.DB_analy,...
    sprintf('feat_seg_pair_%d',i_emg_pair)));

load(fullfile(path.DB_process,name.DB_analy,...
    sprintf('feat_seq_pair_%d',i_emg_pair)));

load(fullfile(path.DB_process,name.DB_analy,...
    sprintf('idx_fe_seq_pair_%d',i_emg_pair)));

%=============numeber of variables
[n_seg, n_feat, n_fe, n_trl, n_sub] = size(feat_seg);
idx_plateau_part = 6: n_seg;
n_plateau = length(idx_plateau_part);
idx_transition_part = 1:5;
n_transition = length(idx_transition_part);
n_emg_pair = 1;
n_sub_compared = n_sub - 1;
n_ftype = length(name.feat);
n_bip_ch = 4;

%=============indices of variables
idx_sub = 1 : n_sub;
idx_trl = 1 : n_trl;
n_trl = 1;

idx_feat = {[1,2,3,4];[5,6,7,8];[9,10,11,12];13:28};

for i_sub = 1 : n_sub
for i_trl = 1 : n_trl
    
%display of subject and trial in progress
fprintf('i_emg_pair:%d i_sub:%d i_trial:%d\n',i_emg_pair,i_sub,i_trl);
if n_transforemd>=1
% memory allocation similarily transformed feature set
feat_t = cell(n_seg,n_fe);
for i_seg = idx_plateau_part
for i_fe = 1 : n_fe

    % memory allocation feature set
    feat_t{i_seg,i_fe} = cell(1,n_ftype);

    % you should get access to DB of other experiment with each
    % features
    for i_FeatName = 1 : n_ftype

        % number of feature of each type
        n_feat_type = length(idx_feat{i_FeatName});

        % feat from this experiment
        xtrain_ref = feat_seg(i_seg,idx_feat{i_FeatName},i_fe,...
            i_trl,i_sub)';

        %---------feat to be compared from this experiment----%
        % [n_seg:30, n_feat:28, n_fe:8, n_trl:20, n_sub:30, n_emg_pair:3]

        % compare ohter subject except its own subject
        idx_sub_DB = find(countmember(idx_sub,i_sub)==0==1);
        n_sub_DB = length(idx_sub_DB);
        switch id_att_compare
            case 'all'
                feat_compare = feat_seg(end-n_seg+1:end,idx_feat{i_FeatName},...
                    :,:,idx_sub_DB);

            case 'Only_Seg'
                feat_compare = feat_seg(i_seg,idx_feat{i_FeatName},...
                    :,:,idx_sub_DB);

            case 'Seg_FE'
                feat_compare = feat_seg(i_seg,idx_feat{i_FeatName},...
                    i_fe,:,idx_sub_DB);
        end

        % permutation giving [n_feat, n_fe, n_trl, n_sub ,n_seg]
        % to bring about formation of [n_feat, others]
        feat_compare = permute(feat_compare,[2 3 4 5 1]);

        %  size(2):FE, size(5):seg
        feat_compare = reshape(feat_compare,...
            [n_feat_type, size(feat_compare,2)*n_trl*n_sub_DB]);

        % get similar features by determined number of
        % transformed DB
        feat_t{i_seg,i_fe}{i_FeatName} = ...
            dtw_search_n_transf(xtrain_ref, feat_compare, n_transforemd)';
        %-----------------------------------------------------%

    end
end
end
feat_t = feat_t(idx_plateau_part,:); 
% arrange feat transformed and target
% concatinating features with types
feat_t = cellfun(@(x) cat(2,x{:}),feat_t,'UniformOutput',false);
end
% validate with number of transformed DB
for n_t = n_transforemd
if n_t >= 1
    % get feature-transformed with number you want
    xtrain_tf = cellfun(@(x) x(1:n_t,:),feat_t,...
        'UniformOutput',false);

    % get size to have target
    size_temp = cell2mat(cellfun(@(x) size(x,1),...
        xtrain_tf(:,1),'UniformOutput',false));

    % feature transformed
    xtrain_tf = cell2mat(xtrain_tf(:));

    % target for feature transformed
    ytrain_tf = repmat(1:n_fe,sum(size_temp,1),1);
    ytrain_tf = ytrain_tf(:);
else
    xtrain_tf = [];
    ytrain_tf = [];
end

% feat for anlaysis
xtrain_ref = reshape(permute(feat_seg(:,:,:,i_trl,i_sub,i_emg_pair),...
    [1 3 2]),[n_seg*n_fe,n_feat]);

% xtrain_ref = reshape(permute(feat_seg(idx_plateau_part,:,:,i_trl,i_sub,i_emg_pair),...
%     [1 3 2]),[n_plateau*n_fe,n_feat]);
% xtrain_transition_part = reshape(permute(feat(idx_transition_part,:,:,i_trl,i_sub,i_emg_pair),...
%     [1 3 2]),[n_transition*n_fe,n_feat]);
% xtrain_mlp_input = reshape(feat(idx_transition_part,1:4,:,i_trl,i_sub,i_emg_pair),...
%    [n_transition*4,n_fe])';
ytrain_ref = repmat(1:n_fe,n_seg,1);
ytrain_ref = ytrain_ref(:);

%=================PREPARE DB FOR TRAIN====================%
xtrain = cat(1,xtrain_ref,xtrain_tf);
ytrain = cat(1,ytrain_ref,ytrain_tf);
%=========================================================%

%=================EMG ONSET FEATURE=======================%
score  = cell(n_bip_ch,1);
for i_ch = 1 : n_bip_ch
    [~,score{i_ch}] = predict(model_tree_emg_onset,...
        xtrain(:,i_ch));
end
score = cat(2,score{:});
if (id_use_emg_onset_feat)
xtrain = [xtrain,score];
end
%=========================================================%

%==================TRAIN EACH EMOTION=====================%
model.emotion = fitcdiscr(...
    xtrain,...
    ytrain);
%=========================================================%


%================= TEST=====================%
% get input and targets for test DB
idx_trl_test = find(idx_trl~=i_trl==1);

c_t = 0;
for i_trl_test = idx_trl_test
    c_t = c_t + 1;
    xtest = feat_test{i_trl_test,i_sub};
    target_inform = idx_fe_seq{i_trl_test,i_sub};
     %====PASS THE TEST FEATURES TO CLASSFIERS=============%
    %----EMG ONSET
    xtest_rms = xtest(:,idx_feat{1});
    
    s_ons = cell(n_bip_ch,1);
    for i_ch = 1 : n_bip_ch
        [~,s_ons{i_ch}] = ...
            predict(model_tree_emg_onset,xtest_rms(:,i_ch));
    end
    %=======EMG ONSET FEATURE ADDITION
    if (id_use_emg_onset_feat)
        xtest = [xtest,cat(2,s_ons{:})];
    end
            
    %----EMOTION CLASSFIER
    [ypred,y_likelihood] = predict(model.emotion,xtest);
    
    
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
        
        conditonal_voting;
        
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
        
%         if i < n_c3
%             continue;
%         end
%         
%         % condition #3
%         tmp_d = ypred(i-n_c3+1:i);
%         tmp = tmp_d(template_c3);
%         if ~range(tmp(1:3))&&tmp(4)==9&&tmp(1)~=9 % check all values are same
%              y_corrected(i) = ypred(i);
%              c3_fail = 0;
%              activated = 1;
%         else
%             c3_fail = 1;
%         end
%         
%         if activated
%             count = count + 1;
%         end
%         if count > 0 && count <=10
%              y_corrected(i) = 9;
%         end
%         if count == 10
%             count = 0;
%             activated = 0;
%         end
    end
    
    
    %- answer data
    len_test = length(xtest);
    ytest = NaN(len_test,1);
    tmp = [target_inform(:,2) ,target_inform(:,2) + 30-1];
    for i_fe = 1 : n_fe
        ytest(linspace(tmp(i_fe,1),tmp(i_fe,2),30)) =  target_inform(i_fe,1);
    end
    
    figure;
    hold on;
%     plot(ypred);
    plot(y_corrected);
    plot(ytest,'r','LineWidth',3);
    set(gca,'YTickLabel',strrep(name.emo,'_',' '))

    %=====================================================%
    
    %- simple majority voting
    mv_size = 10;
    ypred_mv = NaN(length(ypred)-mv_size+1,1);
    count = 0;
    while count < length(ypred)
        count = count + 1;
        if count<15 % get rid of filtering effect
            continue;
        end
%         xtest_ti = [xtest(count - n_transition+1:count,1:4),llhood(count - n_transition+1:count)];
%         net(xtest_ti')
%         abs(tmp)^(alpha*ypred(count))
        
        class_counts = countmember(1:n_fe,ypred(count-mv_size+1:count));
        [max_v,max_idx] = get_max_and_idx(class_counts);
        ypred_mv(count) = max_idx;
    end
    ypred_mv = [NaN(mv_size-1,1);ypred_mv];
    
    figure;
    hold on;
    plot(ypred_mv);
    plot(ytest,'r','LineWidth',3);
    set(gca,'YTickLabel',strrep(name.emo,'_',' '))
    
    a = 1;
end      

end
end
end
end

%-----------------^--------------------------------------------------------%

%-------------------------preprocessing of results------------------------%

%-------------------------------------------------------------------------%

%-------------------------------save results------------------------------%
name.saving = sprintf('DB-%s_ntrans-%d_onset-%d-compmethod-%s',...
id_DB_val,n_transforemd,id_use_emg_onset_feat,id_att_compare);

save(fullfile(path_saving,name.saving),'r');

load(fullfile(path_saving,name.saving));
%-------------------------------------------------------------------------%

%==============================부위별 분류 결과============================%
close all
tmp1 = squeeze(r.output_n_target(1,:,:,1,30,:,:));
tmp1 = tmp1(:);
tmp1 = cat(1,tmp1{:});



for i_part = 1 : n_part
% name.neutral = {'neutral','neutral'};
n_fe_result = length(name.gesture_c1{i_part});
output = strcat(tmp1(:,i_part));
target = strcat(tmp1(:,i_part+2));



% name.fe_eye = {'neutral';'eye_brow_down';'eye_brow_happy';'eye_brow_sad'};
[~,tmp]  = ismember(output,name.gesture{i_part,2});
idx2delete = find(tmp==0);
tmp(idx2delete) =[];

B = unique(tmp);
out = [B,histc(tmp,B)];

output_tmp = full(ind2vec(tmp',n_fe_result));

[~,tmp]  = ismember(target,name.gesture{i_part,2});
tmp(idx2delete) =[];
target_tmp = full(ind2vec(tmp',n_fe_result));

B = unique(tmp);
out = [B,histc(tmp,B)];
% compute confusion
[~,mat_conf,idx_of_samps_with_ith_target,~] = ...
confusion(target_tmp,output_tmp);

figure;
h = plotconfusion(target_tmp,output_tmp);
name.conf = strrep(name.gesture{i_part,2},'_',' ');

h.Children(2).XTickLabel(1:n_fe_result) = name.conf;
h.Children(2).YTickLabel(1:n_fe_result)  = name.conf;

% plotConfMat(mat_conf', name.conf)
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