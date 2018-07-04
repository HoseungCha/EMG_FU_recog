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
for i_ses = 1 : n_trl
%display of subject and trial in progress
fprintf('i_emg_pair:%d i_sub:%d i_trial:%d\n',i_emg_pair,i_sub,i_ses);

% feat for anlaysis
feat_ref = reshape(permute(feat(:,:,:,i_ses,i_sub,i_emg_pair),...
    [1 3 2]),[n_seg*n_fe,n_feat]);
target_feat_ref = repmat(1:n_fe,n_seg,1);
target_feat_ref = target_feat_ref(:);


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