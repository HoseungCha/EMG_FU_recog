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
name_DB_analy = 'feat_set_DB_raw2_n_sub_41_n_seg_60_n_wininc_102_winsize_409';

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
tmp = load(fullfile(path_DB_process,name_DB_analy,'idx_fe_seq_pair_1'));
tmp_name = fieldnames(tmp);
idx_fe_seq = getfield(tmp,tmp_name{1}); %#ok<GFLD>
%-------------------------------------------------------------------------%

%-----------------------experiment information----------------------------%
% trigger singals corresponding to each facial expression(emotion)
name_emo = {'angry','clench',...
'lip_corner_up_left','lip_corner_up_right',...
'lip_corner_up_both','fear',...
'happy','kiss','neutral',...
'sad','surprised'};

name_fe = {'eye_brow_down-lip_tighten';'neutral-clench';...
'neutral-lip_corner_up_left';'neutral-lip_corner_up_right';...
'neutral-lip_corner_up_both';'eye_brow_sad-lip_stretch_down';...
'eye_brow_happy-lip_happy';'neutral-kiss';'neutral-neutral';...
'eye_brow_sad-lip_sulky';'eye_brow_up-lip_open'};

% get name list of subjects
[name_subject,~] = read_names_of_file_in_folder(path_DB_raw);
n_sub = length(name_subject);
% mapped avartar gesture of each facial part of each emotion
for i = 1 : length(name_fe)
tmp = strsplit(name_fe{i},'-');
name_gesture_c1{1}{i,1} = tmp{1};
name_gesture_c1{2}{i,1} = tmp{2};
end

% name of types of features
name_feat = {'RMS';'WL';'SampEN';'CC'};

n_emg_pair = 3;
n_trl = 20;
idx_trl = 1 : 20;
n_fe = 11;
%-------------------------------------------------------------------------%

%----------------------set saving folder----------------------------------%
% set folder for saving
name_folder_saving = [id_att_compare,'_',id_DB_val,'_',num2str(n_transforemd)];

% set saving folder for windows
path_saving = make_path_n_retrun_the_path(path_DB_analy,name_folder_saving);
%-------------------------------------------------------------------------%

%----------------------memory allocation for results----------------------%
% memory allocatoin for accurucies
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
feat_t = cell(n_seg,n_fe);
for i_seg = idx_plateau_part
    for i_FE = 1 : n_fe

        % memory allocation feature set
        feat_t{i_seg,i_FE} = cell(1,n_ftype);

        % you should get access to DB of other experiment with each
        % features
        for i_FeatName = 1 : n_ftype

            % number of feature of each type
            n_feat_type = length(idx_feat{i_FeatName});

            % feat from this experiment
            xtrain_ref = feat(i_seg,idx_feat{i_FeatName},i_FE,...
                i_trl,i_sub,i_emg_pair)';

            %---------feat to be compared from this experiment----%
            % [n_seg:30, n_feat:28, n_fe:8, n_trl:20, n_sub:30, n_emg_pair:3]

            % compare ohter subject except its own subject
            idx_sub_DB = find(countmember(idx_sub_val,i_sub)==0==1);
            n_sub_DB = length(idx_sub_DB);
            switch id_att_compare
                case 'all'
                    feat_compare = feat(end-n_seg+1:end,idx_feat{i_FeatName},...
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
                dtw_search_n_transf(xtrain_ref, feat_compare, n_transforemd)';
            %-----------------------------------------------------%

        end
    end
end
feat_t(1:n_seg-n_seg,:) = [];
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

% train by regulaized CSP
% one-against-one multiclass problem applied
idx_combi_val = nchoosek(1:n_fe,2);
W = cell(length(idx_combi_val),1);
model = cell(length(idx_combi_val),1);
for i_clf = 1 : length(idx_combi_val)

load(fullfile(path_DB_analy,'DB_win',...
sprintf('xtrain_s_%02d_t_%02d_p_%d',i_sub,i_trl,i_emg_pair)));
tmp = cat(1,xtrain{[idx_combi_val(i_clf,1),idx_combi_val(i_clf,2)]});
xtrain_cur = permute(cat(3,tmp{:}),[2 1 3]);


load(fullfile(path_DB_analy,'DB_win',...
sprintf('xtrain_s_%02d_t_%02d_p_%d',i_sub+1,i_trl,i_emg_pair)));
tmp = cat(1,xtrain{[idx_combi_val(i_clf,1),idx_combi_val(i_clf,2)]});
xtrain_gen = permute(cat(3,tmp{:}),[2 1 3]);

%%%%%%%%%%%%%%%%%%%
betas=[ 0,1e-2,1e-1,2e-1,4e-1,6e-1];
gammas=[0,1e-3,1e-2,1e-1,2e-1];
n_reg = length(betas)*length(gammas);
[numT,numCh,nTrl] = size(xtrain_gen);%Input data, see documentation
genMs=zeros(2,1);
genSs = zeros(numCh,numCh,2);
gnd = repmat(1:2,[40,1]);
gnd = gnd(:);
%%%%%%%%%%%%%%%%%%%
for i=1:2%for each class
    Idxs=find(gnd==i);
    EEG=xtrain_gen(:,:,Idxs);
    genMs(i)=length(Idxs);
    C=zeros(numCh,numCh,genMs(i));%Sample covariance matrix, equation (1) in the paper
    for trial=1:genMs(i)
        tmpC = (EEG(:,:,trial)'*EEG(:,:,trial));        
        C(:,:,trial) = tmpC./trace(tmpC);%normalization
    end 
    genSs(:,:,i)=mean(C,3);
end

% training with all regulaizaion parmaeter
selCh = 6;

% get weight vectors
prjW = get_W_regcsp(xtrain_cur,gnd,betas,gammas,genSs,genMs);
% get features (n_seg, n_feat(n_components), n_reg) by using the vectors
xtrain_feat = get_feat_using_w_regcsp(xtrain_cur,prjW,selCh);

W{i_clf} = prjW;
%==================TRAIN EACH EMOTION by LDA=====================%
gnd = repmat([idx_combi_val(i_clf,1),idx_combi_val(i_clf,2)],40,1);
gnd = gnd(:);
model{i_clf}.emotion = cell(n_reg,1);
for i_reg = 1 : n_reg
    model{i_clf}.emotion{i_reg} = fitcdiscr(xtrain_feat(:,:,i_reg),gnd);
%     disp(kfoldLoss(crossval(model{i_clf}.emotion{i_reg})));
end
%================================================================%
end
%================= TEST=====================%
% get input and targets for test DB
idx_trl_test = find(idx_trl~=i_trl==1);

c_t = 0;
for i_trl_test = idx_trl_test
    c_t = c_t + 1;
    
    load(fullfile(path_DB_analy,'DB_win',...
    sprintf('xtest_s_%02d_t_%02d_p_%d',i_sub,i_trl,i_emg_pair)));

    ypred = NaN(length(xtest),n_reg);
    for i_win = 1 : length(xtest)
        % get features (n_seg, n_feat(n_components), n_reg) by using the vectors
        xtest_feat = get_feat_using_w_regcsp(xtest{i_win}',...
            W{i_clf},selCh);
        
        for i_reg = 1 : n_reg
            tmp = NaN(length(idx_combi_val),1);
            for i_clf = 1 : length(idx_combi_val)
                tmp_xtest = xtest_feat(:,:,i_reg);
                [tmp(i_clf),~] = predict(model{i_clf}.emotion{i_reg},tmp_xtest);
            end
            [~,ypred(i_win,i_reg)] = get_max_and_idx(countmember(1:n_fe,tmp));
            disp([i_win,i_reg,i_clf]);
        end 
    end

    target_inform = idx_fe_seq{i_trl_test,i_sub};
    
    % conditoinal voting
    for i_reg = 1 : n_reg
        y_corrected = conditonal_voting(ypred(:,i_reg));
    

    %- answer data
    len_test = length(xtest);
    ytest = NaN(len_test,1);
    tmp = [target_inform(:,2) ,target_inform(:,2) + 30-1];
    for i_fe = 1 : n_fe
        ytest(linspace(tmp(i_fe,1),tmp(i_fe,2),30)) =  target_inform(i_fe,1);
    end
    
    figure(i_reg);
    hold on;
%     plot(ypred);
    plot(y_corrected);
    plot(ytest,'r','LineWidth',3);
    set(gca,'YTickLabel',strrep(name_emo,'_',' '))
    end
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
    set(gca,'YTickLabel',strrep(name_emo,'_',' '))
    
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