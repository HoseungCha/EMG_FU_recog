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
n_comb = 3;
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
features = zeros(n_seg,n_feat,n_fe,n_trl,n_sub,n_comb);

for i_comb = 1 : n_comb
for i_sub= 1 : n_sub
    %subject name
    tmp_name_sub = name_sub{i_sub}(end-2:end); 
    
    % read BDF
    [~,path_file] = read_names_of_file_in_folder(path_sub{i_sub},'*bdf');
   
    % for saving feature Set (processed DB)
    count_i_trl = 0;
    for i_trl = 1:n_trl
%     for i_trl = 1
        count_i_trl = count_i_trl + 1;
        out = pop_biosig(path_file{i_trl});
        
       % load trigger when subject put on a look of facial expressoins
        %Trigger latency 및 FE 라벨
        temp = cell2mat(permute(struct2cell(out.event),[1 3 2]))';
        if ~isempty(find(temp(:,1)==16385, 1)) || ...
                ~isempty(find(temp(:,1)==16384, 1))
            temp(temp(:,1)==16384,:) = [];
            temp(temp(:,1)==16385,:) = [];
        end
        idx2use_fe = zeros(n_fe,1);
        for i_fe = 1 : n_fe
            tmp_fe = find(temp(:,1)==i_fe);
            idx2use_fe(i_fe) = tmp_fe(2);
        end
        [~,idx_seq_FE] = sort(idx2use_fe);
        lat_trg = temp(idx2use_fe,2);
        lat_trg = lat_trg(idx_seq_FE);
        
        clear temp tmp_fe idx2use_fe
        % get raw data and bipolar configuration
%         raw_data = double(OUT.data'); % raw data
%         temp_chan = cell(1,6);
        % get raw data and bipolar configuration        
        data_bip.RZ= out.data(idx_pair_right(i_comb,1),:) - out.data(idx_pair_right(i_comb,2),:);%Right_Zygomaticus
        data_bip.RF= out.data(4,:) - out.data(5,:); %Right_Frontalis
        data_bip.LF= out.data(6,:) - out.data(7,:); %Left_Corrugator
        data_bip.LZ= out.data(idx_pair_left(i_comb,1),:) - out.data(idx_pair_left(i_comb,2),:); %Right_Zygomaticus
        data_bip = double(cell2mat(struct2cell(data_bip)))';
        clear out;
        % Filtering
        data_filtered = filter(fp.nb, fp.na, data_bip,[],1);
        data_filtered = filter(fp.bb, fp.ba, data_filtered, [],1);
        clear data_bip;
        % for plot
%         figure;plot(filtered_data)
        % Feat extration with windows 
        
%         wininc = floor(0.05*SF2use); 
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
        hf.Position = [-2585 -1114 1920 1091];
        subplot(n_trl,1,i_trl);
        plot(temp_feat(1:end,1:4));
        hold on;
        stem(idx_trg_start,repmat(100,[n_fe,1]));
        ylim([1 300]);
        drawnow;
        
       % Get Feature sets(preprocessed DB)
       % [n_seg,n_feat,n_fe,n_trl,n_sub,n_comb]
        temp = [];
        for i_emo_orer_in_this_exp = 1 : n_fe
            features(:,:,idx_seq_FE(i_emo_orer_in_this_exp),count_i_trl,i_sub,i_comb) = ...
                        temp_feat(idx_trg_start(i_emo_orer_in_this_exp):...
                        idx_trg_start(i_emo_orer_in_this_exp)+floor((period_FE*fp.SF2use)/n_wininc)-1 ,:);
                    
            temp = [temp;temp_feat(idx_trg_start(i_emo_orer_in_this_exp):...
                        idx_trg_start(i_emo_orer_in_this_exp)+floor((period_FE*fp.SF2use)/n_wininc)-1 ,:)];
        end 
    end  
    % plot the DB 
    c = getframe(hf);
    imwrite(c.cdata,fullfile(path_save,[tmp_name_sub(1:3),'.jpg']));
    close(hf);
end
end

% 결과 저장
save(fullfile(path_save,'feat_set'),'features');




