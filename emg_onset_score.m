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

% idx_fe2reject = [3,9,17];
idx_fe2use = [7,15];

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
% feat(:,:,idx_fe2reject,:,:) = [];
feat = feat(:,:,idx_fe2use,:,:);

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
name_fe = name_fe(idx_fe2use);
name_emo = name_emo(idx_fe2use);
name_fe_eb = name_fe_eb(idx_fe2use);
name_fe_lp = name_fe_lp(idx_fe2use);

% name_fe(idx_fe2reject) = [];
% name_emo(idx_fe2reject) = [];
% name_fe_eb(idx_fe2reject) = [];
% name_fe_lp(idx_fe2reject) = [];

name_fe2cfy = {'neutral-neutral';'eye_brow_down-lip_tighten';'eye_brow_happy-lip_happy';...
'eye_brow_sad-lip_sulky';'eye_brow_up-lip_open'};

idx_fe2cfy = find(contains(name_fe,name_fe2cfy)==1);
n_fe_cfy = length(idx_fe2cfy);

% get name list of subjects
[name_subject,~] = read_names_of_file_in_folder(path_DB_raw);

% classifier of each facial part
name_classifier = {'cf_eyebrow','cf_lipshapes'};
n_cf = 2;
n_class = [2,5,5];
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
emg_onset_scores = cell(n_emg_pair,n_sub,n_trl,n_seg,n_fe);
%-------------------------------------------------------------------------%

%------------------------------------main---------------------------------%

% get accrucies and output/target (for confusion matrix) with respect to
% subject, trial, number of segment, FE,
for i_emg_pair = 1 : n_emg_pair
for i_sub = idx_sub
for i_trl = 1 : n_trl
%display of subject and trial in progress
fprintf('i_emg_pair:%d i_sub:%d i_trial:%d\n',i_emg_pair,i_sub,i_trl);


for i_fe = 1 : n_fe
    % save score(likelihood) in circleque
    onset_cq = circlequeue(n_seg,8);
    for i_seg = 1 : n_seg

        % get feature during real-time
        f = feat(i_seg,:,i_fe,i_trl,i_sub,i_emg_pair);

        %====PASS THE TEST FEATURES TO CLASSFIERS=============%
        %----EMG ONSET
        f_onset = f(idx_feat{1});
        s_ons = NaN(n_bip_ch,2);
        for i_ch = 1 : n_bip_ch
            [~,s_ons(i_ch,:)] = ...
                predict(model_tree_emg_onset,f_onset(i_ch));
        end
        onset_cq.add(s_ons(:)');

        %=====================================================%

        if i_seg<15
            onset_cq_mv = mean(onset_cq.getLastN(i_seg),1);
        else
            onset_cq_mv = mean(onset_cq.getLastN(15),1);
        end

        % can we use the socre_matrix as new feature?
        emg_onset_scores{i_emg_pair,i_sub,i_trl,i_seg,i_fe}...
        = onset_cq_mv;
    end
end
end
end
end
%-----------------^--------------------------------------------------------%

%-------------------------preprocessing of results------------------------%

%-------------------------------------------------------------------------%

%-------------------------------save results------------------------------%
% save(fullfile(path_saving,'result'),'r');
%-------------------------------------------------------------------------%

%==============================EMG onset 분류결과==========================%
tmp1 = squeeze(emg_onset_scores(:,2,:,15:30,:));
tmp1 = tmp1(:);
tmp1 = cat(1,tmp1{:});
x = tmp1;
y = repmat(1:n_fe,[length(x)/n_fe,1]);
y = y(:);


model = fitcdiscr(x,y,'discrimType','diagLinear');
cvmdl =  crossval(model);

ypred = kfoldPredict(cvmdl);
output = full(ind2vec(ypred',n_fe));
target = full(ind2vec(y',n_fe));

figure(i_sub);
name_conf = strrep(name_fe,'_',' ');
h = plotconfusion(target,output);
h.Children(2).XTickLabel(1:n_fe) = name_conf;
h.Children(2).YTickLabel(1:n_fe)  = name_conf;
%=========================================================================%

%=====================sub independent tets================================%
output = cell(n_sub,1);
target = cell(n_sub,1);
for i_sub = 1 : n_sub
tmp1 = squeeze(emg_onset_scores(:,i_sub,:,15:30,:));
tmp1 = tmp1(:);
tmp1 = cat(1,tmp1{:});
xtrain = tmp1;
ytrain = repmat(1:n_fe,[length(xtrain)/n_fe,1]);
ytrain = ytrain(:);


model = fitcdiscr(xtrain,ytrain,'discrimType','diagLinear');

tmp1 = squeeze(emg_onset_scores(:,1:n_sub~=i_sub,:,15:30,:));
tmp1 = tmp1(:);
tmp1 = cat(1,tmp1{:});
xtest = tmp1;
ytest = repmat(1:n_fe,[length(xtest)/n_fe,1]);
ytest = ytest(:);

ypred = predict(model,xtest);

target{i_sub} = ytest;
output{i_sub} = ypred;
end
output = full(ind2vec(cell2mat(output)',n_fe));
target = full(ind2vec(cell2mat(target)',n_fe));

figure;
name_conf = strrep(name_fe,'_',' ');
h = plotconfusion(target,output);
h.Children(2).XTickLabel(1:n_fe) = name_conf;
h.Children(2).YTickLabel(1:n_fe)  = name_conf;
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