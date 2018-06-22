%--------------------------------------------------------------------------
% feat extracion code for faicial unit recognition using Myo Expression DB
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------
clc; clear all; close all;

%------------------------code analysis parameter--------------------------%
% decide the raw DB to analyse
name_DB_raw = 'DB_raw3';
name_DB_process = 'DB_processed3';

% decide number of segments in 3-sec long EMG data
period_wininc = 0.1; % s

idx_fe2reject = [3,9,17];

%-------------------------------------------------------------------------%

%-------------set paths in compliance with Cha's code structure-----------%
path_research = fileparts(fileparts(fileparts(fullfile(cd))));
path_code = fileparts(fullfile(cd));
path_DB = fullfile(path_code,'DB');
path_DB_raw = fullfile(path_DB,name_DB_raw);
path_DB_process = fullfile(path_DB,name_DB_process);
%-------------------------------------------------------------------------%

%-------------------------add functions-----------------------------------%
addpath(genpath(fullfile(path_research,'_toolbox')));
addpath(genpath(fullfile(cd,'functions')));
%-------------------------------------------------------------------------%

%------------------------experiment infromation---------------------------%
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
name_fe_old = name_fe;
old_index = 1:19;
old_index(idx_fe2reject) = [];
index_pair = [old_index;1:16'];
name_fe(idx_fe2reject) = [];
n_fe = length(name_fe);% Number of facial expression
n_trl = 10; % Number of Trials
%-------------------------------------------------------------------------%

%----------------------------paramters------------------------------------%
% filter parameters
fp.SF2use = 2048;
fp.filter_order = 4; fp.Fn = fp.SF2use/2;
fp.freq_notch = [58 62];
fp.freq_BPF = [20 450];
[fp.nb,fp.na] = butter(fp.filter_order,fp.freq_notch/fp.Fn,'stop');
[fp.bb,fp.ba] = butter(fp.filter_order,fp.freq_BPF/fp.Fn,'bandpass');

% read file path of subjects
[name_sub,path_sub] = read_names_of_file_in_folder(path_DB_raw);
n_sub= length(name_sub);

% experiments or feat extractions parameters
n_feat = 28;
n_emg_pair = 3;
n_ch = 4;
idx_pair_right = [1,2;1,3;2,3]; %% 오른쪽 전극 조합
idx_pair_left = [10,9;10,8;9,8]; %% 왼쪽 전극 조합
period_FE = 3; % 3-sec
n_seg = period_FE/period_wininc; % choose 30 or 60
n_wininc = floor(period_wininc*fp.SF2use); 
n_winsize = floor(period_wininc*fp.SF2use); % win

% subplot 그림 꽉 차게 출력 관련 
id_subplot_make_it_tight = true; subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~id_subplot_make_it_tight,  clear subplot;  end
%-------------------------------------------------------------------------%

%----------------------set saving folder----------------------------------%
name_folder4saving = sprintf(...
'feat_set_%s_n_sub_%d_n_seg_%d_n_wininc_%d_winsize_%d',...
    name_DB_raw,n_sub,n_seg,n_wininc,n_winsize);
path_save = make_path_n_retrun_the_path(fullfile(path_DB_process),...
    name_folder4saving);
%-------------------------------------------------------------------------%


% memory alloation
features = NaN(n_seg,n_feat,n_fe,n_trl,n_sub,n_emg_pair);

for i_emg_pair = 1 : n_emg_pair
for i_sub= 1 : n_sub
    % read BDF
    [~,path_file] = read_names_of_file_in_folder(path_sub{i_sub},'*bdf');
    n_trl_curr = length(path_file);
    % for saving feature Set (processed DB)
    for i_trl = 1 : n_trl
        if n_trl_curr<i_trl
           continue; 
        end    
        fprintf('i_emg_pair-%d i_sub-%d i-trl-%d\n',i_emg_pair,i_sub,i_trl);
        
        % get bdf file
        out = pop_biosig(path_file{i_trl});
        
        % load trigger
        tmp_trg = cell2mat(permute(struct2cell(out.event),[1 3 2]))';
        
        % get latnecy of each trigger
        [lat_trg,idx_seq_fe] = get_trg(tmp_trg);
        
        if length(idx_seq_fe) == 19
        idx_delete = ismember(idx_seq_fe,idx_fe2reject);
        idx_seq_fe(idx_delete) = [];
        lat_trg(idx_delete) = [];
            for i = 1 : 16
                idx_seq_fe(i) = index_pair(2,idx_seq_fe(i)==index_pair(1,:));
            end
        end
       
        % get raw data and bipolar configuration        
        data_bip.RZ= out.data(idx_pair_right(i_emg_pair,1),:) - out.data(idx_pair_right(i_emg_pair,2),:);%Right_Zygomaticus
        data_bip.RF= out.data(4,:) - out.data(5,:); %Right_Frontalis
        data_bip.LF= out.data(6,:) - out.data(7,:); %Left_Corrugator
        data_bip.LZ= out.data(idx_pair_left(i_emg_pair,1),:) - out.data(idx_pair_left(i_emg_pair,2),:); %Left_Zygomaticus
        data_bip = double(cell2mat(struct2cell(data_bip)))';
        clear out;
        % Filtering
        data_filtered = filter(fp.nb, fp.na, data_bip,[],1);
        data_filtered = filter(fp.bb, fp.ba, data_filtered, [],1);
        clear data_bip;
        % for plot
%         figure;plot(filtered_data)
        % Feat extration with windows 
        n_win = floor((length(data_filtered) - n_winsize)/n_wininc)+1;
        temp_feat = zeros(n_win,n_feat); idx_trg_as_window = zeros(n_win,1);
        st = 1;
        en = n_winsize;
        for i = 1: n_win
            idx_trg_as_window(i) = en;
            curr_win = data_filtered(st:en,:);
            temp_rms = sqrt(mean(curr_win.^2));
            temp_CC = featCC(curr_win,n_ch);
            temp_WL = sum(abs(diff(curr_win,2)));
            temp_SampEN = SamplEN(curr_win,2);
%             temp_feat(i,:) = [temp_CC,temp_rms,temp_SampEN,temp_WL];
            temp_feat(i,:) = [temp_rms,temp_WL,temp_SampEN,temp_CC];
            % moving widnow
            st = st + n_wininc;
            en = en + n_wininc;                 
        end
        clear temp_rms temp_CC temp_WL temp_SampEN st en
 
        % cutting trigger 
        idx_trg_start = zeros(n_fe,1);
        for i_emo_orer_in_this_exp = 1 : n_fe
            idx_trg_start(i_emo_orer_in_this_exp,1) = find(idx_trg_as_window >= lat_trg(i_emo_orer_in_this_exp),1);
        end
        
        % To confirm the informaion of trrigers were collected right
        hf =figure(i_sub);
        hf.Position = [1921 41 1920 962];
        subplot(n_trl,1,i_trl);
        tmp_plot = temp_feat(1:end,1:4);
        plot(tmp_plot)
        v_min = min(min(tmp_plot(100:end,:)));
        v_max = max(max(tmp_plot(100:end,:)));
        hold on;
        stem(idx_trg_start,repmat(v_min,[n_fe,1]),'k');
        stem(idx_trg_start,repmat(v_max,[n_fe,1]),'k');
        ylim([v_min v_max]);
        drawnow;
        
       % Get Feature sets(preprocessed DB)
       % [n_seg,n_feat,n_fe,n_trl,n_sub,n_comb]
%         temp = [];
        for i_emo_orer_in_this_exp = 1 : n_fe
            features(:,:,idx_seq_fe(i_emo_orer_in_this_exp),i_trl,i_sub,i_emg_pair) = ...
                        temp_feat(idx_trg_start(i_emo_orer_in_this_exp):...
                        idx_trg_start(i_emo_orer_in_this_exp)+floor((period_FE*fp.SF2use)/n_wininc)-1 ,:);
                    
%             temp = [temp;temp_feat(idx_trg_start(i_emo_orer_in_this_exp):...
%                         idx_trg_start(i_emo_orer_in_this_exp)+floor((period_FE*fp.SF2use)/n_wininc)-1 ,:)];
        end 
    end  
    % plot the DB 
    c = getframe(hf);
    savefig(hf,fullfile(path_save,[name_sub{i_sub},'.fig']));
    imwrite(c.cdata,fullfile(path_save,[name_sub{i_sub},'.png']));
    close(hf);
end

end
% 결과 저장
save(fullfile(path_save,['feat_seg.mat']),'features');



%==========================FUNCTIONS======================================%
function [lat_trg,idx_seq_fe] = get_trg(trg)
Idx_trg_obtained = reshape(trg(:,1),[2,size(trg,1)/2])';
trg = reshape(trg(:,2),[2,size(trg,1)/2])';
lat_trg = trg(:,1);

% get sequnece of facial expression in this trial
[~,idx_in_order] = sortrows(Idx_trg_obtained);    
trg = sortrows([idx_in_order,(1:length(idx_in_order))'],1); 
idx_seq_fe = trg(:,2); 
end
