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
name_DB_raw = 'DB_raw2';
name_DB_process = 'DB_processed2';

% decide number of segments in 3-sec long EMG data
period_wininc = 0.1; % s

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
name_trg = {"화남",1;"어금니깨물기",2;"비웃음(왼쪽)",3;"비웃음(오른쪽)",4;...
    "눈 세게 감기",5;"두려움",6;"행복",7;"키스",8;"무표정",9;"슬픔",10;"놀람",11};

name_FE = name_trg(:,1);
idx_trg = cell2mat(name_trg(:,2));
clear Name_Trg;
n_fe = length(name_FE);% Number of facial expression
n_trl = 20; % Number of Trials
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
win_seg = cell(n_trl,n_sub,n_fe);
win_seq = cell(n_trl,n_sub);
idx_fe_seq = cell(n_trl,n_sub);
for i_emg_pair = 1 : n_emg_pair
for i_sub= 1 : n_sub
    % read BDF
    try
    [~,path_file] = read_names_of_file_in_folder(path_sub{i_sub},'*bdf');
    catch ex
    if strcmp(ex.identifier,'MATLAB:unassignedOutputs')
    [name_file,path_file] = read_names_of_file_in_folder(fullfile(...
        path_sub{i_sub},'emg'),'*bdf');  
    end
    end
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
        
        % check which DB type you are working on
        % total number of trigger 33: Myoexpression1
        % total number of trigger 23: Myoexpression2
        
        switch length(tmp_trg)
            case 33
                [lat_trg,idx_seq_FE] = get_trg_myoexp1(tmp_trg);
            case 23
                [lat_trg,idx_seq_FE] = get_trg_myoexp2(tmp_trg);
            otherwise
                continue; 
        end
        
        
       
        % get raw data and bipolar configuration
%         raw_data = double(OUT.data'); % raw data
%         temp_chan = cell(1,6);
        % get raw data and bipolar configuration     
        idx2use_emg_pair = ...
            [idx_pair_right(i_emg_pair,:),idx_pair_left(i_emg_pair,:)];
        idx2use_ch = sort([idx2use_emg_pair, 4,5,6,7,]);
        data2use = double(out.data(idx2use_ch,:))';
        
        clear out;
        % Filtering
        data_filtered = filter(fp.nb, fp.na, data2use,[],1);
        data_filtered = filter(fp.bb, fp.ba, data_filtered, [],1);
        clear data2use;
        
        n_win = floor((length(data_filtered) - n_winsize)/n_wininc)+1;
        temp_win = cell(n_win,1); idx_trg_as_window = zeros(n_win,1);
        st = 1;
        en = n_winsize;
        for i = 1: n_win
            idx_trg_as_window(i) = en;
            curr_win = data_filtered(st:en,:);
            temp_win{i} = curr_win';
            % moving widnow
            st = st + n_wininc;
            en = en + n_wininc;                 
        end        
        win_seq{i_trl,i_sub}  = temp_win;
 
        % cutting trigger 
        idx_trg_start = zeros(n_fe,1);
        for i_emo_orer_in_this_exp = 1 : n_fe
            idx_trg_start(i_emo_orer_in_this_exp,1) = find(idx_trg_as_window >= lat_trg(i_emo_orer_in_this_exp),1);
        end
        
        idx_fe_seq{i_trl,i_sub} = [idx_seq_FE,idx_trg_start];
        
       % Get Feature sets(preprocessed DB)
       % [n_seg,n_feat,n_fe,n_trl,n_sub,n_comb]
%         temp = [];
        for i_emo_orer_in_this_exp = 1 : n_fe
            win_seg{i_trl,i_sub,idx_seq_FE(i_emo_orer_in_this_exp)} = ...
                        temp_win(idx_trg_start(i_emo_orer_in_this_exp):...
                        idx_trg_start(i_emo_orer_in_this_exp)+floor((period_FE*fp.SF2use)/n_wininc)-1 ,:);
        end 
        
    end  
end
    % 결과 저장
    save(fullfile(path_save,['win_seg_pair_',num2str(i_emg_pair)]),'win_seg');
    save(fullfile(path_save,['win_seq_pair_',num2str(i_emg_pair)]),'win_seq');
end




%==========================FUNCTIONS======================================%
function [lat_trg,idx_seq_FE] = get_trg_myoexp1(trg)
%Trigger latency 및 FE 라벨
if ~isempty(find(trg(:,1)==16385, 1)) || ...
        ~isempty(find(trg(:,1)==16384, 1))
    trg(trg(:,1)==16384,:) = [];
    trg(trg(:,1)==16385,:) = [];
end

idx_seq_FE = trg(2:3:33,1);
lat_trg = trg(2:3:33,2);

% idx2use_fe = zeros(11,1);
% for i_fe = 1 : 11
%     tmp_fe = find(trg_cell(:,1)==i_fe);
%     idx2use_fe(i_fe) = tmp_fe(2);
% end
% [~,idx_seq_FE] = sort(idx2use_fe);
% lat_trg = trg_cell(idx2use_fe,2);
% lat_trg = lat_trg(idx_seq_FE);
end

function [lat_trg,idx_seq_FE] = get_trg_myoexp2(trg)
% get trigger latency when marker DB acquasition has started
lat_trg_onset = trg(1,2);

% check which triger is correspoing to each FE and get latency
tmp_emg_trg = trg(2:end,:);
Idx_trg_obtained = reshape(tmp_emg_trg(:,1),[2,size(tmp_emg_trg,1)/2])';
tmp_emg_trg = reshape(tmp_emg_trg(:,2),[2,size(tmp_emg_trg,1)/2])';
lat_trg = tmp_emg_trg(:,1);

% get sequnece of facial expression in this trial
[~,idx_in_order] = sortrows(Idx_trg_obtained);    
tmp_emg_trg = sortrows([idx_in_order,(1:length(idx_in_order))'],1); 
idx_seq_FE = tmp_emg_trg(:,2); 
end
