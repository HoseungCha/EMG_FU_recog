%--------------------------------------------------------------------------
% feat extracion code for faicial unit recognition 
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------
clc; clear all; close all;
% get tool box
addpath(genpath(fullfile('E:\Hanyang\연구','_matlab_toolbox')));
path_parent=(fileparts(pwd));
addpath(genpath(fullfile(cd,'functions')));
%% 실험 정보
path_rawDB = fullfile(path_parent,'DB','DB_raw');
% trigger singals corresponding to each facial expression(emotion)
Name_Trg = {"무표정",1,1;"눈썹 세게 치켜 올리기",1,2;"눈썹 세게 내려서 인상쓰기",1,3;...
    "양쪽 입고리 세게 올리기",1,4;"코 찡그리기",1,5;"'ㅏ' 모양으로 입모양 짓기",1,6;...
    "'ㅓ' 모양으로 입모양 짓기",1,7;"'ㅗ' 모양으로 입모양 짓기",2,1;...
    "'ㅜ' 모양으로 입모양 짓기",2,2;"'ㅡ' 모양으로 입모양 짓기",2,3;...
    "'l' 모양으로 입모양 짓기",2,4};
Name_FE = Name_Trg(:,1);
Idx_trg = cell2mat(Name_Trg(:,2:3));
clear Name_Trg;
N_FE = length(Name_FE);% Number of facial expression
N_Trl = 20; % Number of Trials

%% filter parameters
fp.SF2use = 2048;
fp.filter_order = 4; fp.Fn = fp.SF2use/2;
fp.Notch_freq = [58 62];
fp.BPF_cutoff_Freq = [20 450];
[fp.nb,fp.na] = butter(fp.filter_order,fp.Notch_freq/fp.Fn,'stop');
[fp.bb,fp.ba] = butter(fp.filter_order,fp.BPF_cutoff_Freq/fp.Fn,'bandpass');

% subplot 그림 꽉 차게 출력 관련 
make_it_tight = true; subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~make_it_tight,  clear subplot;  end

%% read file path of data
[Name_subject,path_subjects] = read_names_of_file_in_folder(path_rawDB);
N_subject = length(Name_subject);
%% decide number of segments in 3-sec long EMG data
N_seg = 30; % choose 30 or 60
%% experiments or feat extractions parameters
N_feat = 28;
N_trl = 20;
N_comb = 3;
N_ch = 4;
idx_pair_right = [1,2;1,3;2,3]; %% 오른쪽 전극 조합
idx_pair_left = [10,9;10,8;9,8]; %% 왼쪽 전극 조합
Time_expression = 3; % 3-sec
Time_window = 0.1;
N_wininc = floor((Time_expression/N_seg)*fp.SF2use); 
N_winsize = floor(Time_window*fp.SF2use); % win
%% set saving folder for windows
Folder2save = sprintf('Feature_set_wininc_time_%.2f_winsize_%d',Time_window,N_winsize);
path2save = make_path_n_retrun_the_path(fullfile(path_parent,'DB',...
    'DB_processed'),Folder2save);
%% 결과 memory alloation
Features = zeros(N_seg,N_feat,N_FE,N_trl,N_subject,N_comb);
% Features(:,:,event_s(i_emo,1),i_trl,i_sub)
for i_comb = 1 : N_comb
for i_sub= 1:N_subject
    
    Name_sub = Name_subject{i_sub}(end-2:end); %subject name
    % read BDF
    [~,path_file] = read_names_of_file_in_folder(path_subjects{i_sub},'*bdf');
   
    % for saving feature Set (processed DB)
    count_i_trl = 0;
    for i_trl = 1:N_Trl
%     for i_trl = 1
        count_i_trl = count_i_trl + 1;
        OUT = pop_biosig(path_file{i_trl});
        
       %% load trigger when subject put on a look of facial expressoins
        %Trigger latency 및 FE 라벨
        temp = cell2mat(permute(struct2cell(OUT.event),[1 3 2]))';
        temp(:,1) = temp(:,1)./128;
        Idx_trg_obtained = reshape(temp(:,1),[2,size(temp,1)/2])';
        temp = reshape(temp(:,2),[2,size(temp,1)/2])';
        lat_trg = temp(:,1);
        [~,idx_in_order] = sortrows(Idx_trg_obtained);
        % [idx_in_order,(1:N_FaExp)'], idices in order corresponding to
        % emotion label
        temp = sortrows([idx_in_order,(1:length(idx_in_order))'],1); 
        Idx_seq_FE = temp(:,2); clear Idx_trg_obtained temp;

        
        %% get raw data and bipolar configuration
%         raw_data = double(OUT.data'); % raw data
%         temp_chan = cell(1,6);
        % get raw data and bipolar configuration        
        data_bip.RZ= OUT.data(idx_pair_right(i_comb,1),:) - OUT.data(idx_pair_right(i_comb,2),:);%Right_Zygomaticus
        data_bip.RF= OUT.data(4,:) - OUT.data(5,:); %Right_Frontalis
        data_bip.LF= OUT.data(6,:) - OUT.data(7,:); %Left_Corrugator
        data_bip.LZ= OUT.data(idx_pair_left(i_comb,1),:) - OUT.data(idx_pair_left(i_comb,2),:); %Right_Zygomaticus
        data_bip = double(cell2mat(struct2cell(data_bip)))';
        clear out;
        %% Filtering
        data_filtered = filter(fp.nb, fp.na, data_bip,[],1);
        data_filtered = filter(fp.bb, fp.ba, data_filtered, [],1);
        clear data_bip;
        % for plot
%         figure;plot(filtered_data)
        %% Feat extration with windows 
        
%         wininc = floor(0.05*SF2use); 
        N_window = floor((length(data_filtered) - N_winsize)/N_wininc)+1;
        temp_feat = zeros(N_window,N_feat); Idx_trg_as_window = zeros(N_window,1);
        st = 1;
        en = N_winsize;
        for i = 1: N_window
            Idx_trg_as_window(i) = en;
            curr_win = data_filtered(st:en,:);
            temp_rms = sqrt(mean(curr_win.^2));
            temp_CC = featCC(curr_win,N_ch);
            temp_WL = sum(abs(diff(curr_win,2)));
            temp_SampEN = SamplEN(curr_win,2);
%             temp_feat(i,:) = [temp_CC,temp_rms,temp_SampEN,temp_WL];
            temp_feat(i,:) = [temp_rms,temp_WL,temp_SampEN,temp_CC];
            % moving widnow
            st = st + N_wininc;
            en = en + N_wininc;                 
        end
        clear temp_rms temp_CC temp_WL temp_SampEN st en
 
        %% cutting trigger 
        Idx_TRG_Start = zeros(N_FE,1);
        for i_emo_orer_in_this_exp = 1 : N_FE
            Idx_TRG_Start(i_emo_orer_in_this_exp,1) = find(Idx_trg_as_window >= lat_trg(i_emo_orer_in_this_exp),1);
        end
        
        %% To confirm the informaion of trrigers were collected right
        hf =figure(i_sub);
        hf.Position = [-2585 -1114 1920 1091];
        subplot(N_Trl,1,i_trl);
        plot(temp_feat(20:end,1:4));
        hold on;
        stem(Idx_TRG_Start,repmat(100,[N_FE,1]));
        ylim([1 300]);
        subplot(N_Trl,2,2*i_trl);
        plot(temp_feat(20:end,1:4));
        hold on;
        stem(Idx_TRG_Start,ones([N_FE,1]));
        drawnow;
        
       %% Get Feature sets(preprocessed DB)
        for i_emo_orer_in_this_exp = 1 : N_FE
            Features(:,:,Idx_seq_FE(i_emo_orer_in_this_exp),count_i_trl,i_sub,i_comb) = ...
                        temp_feat(Idx_TRG_Start(i_emo_orer_in_this_exp):...
                        Idx_TRG_Start(i_emo_orer_in_this_exp)+floor((Time_expression*fp.SF2use)/N_wininc)-1 ,:); 
        end 
    end  
    %% plot the DB 
    c = getframe(hf);
    imwrite(c.cdata,fullfile(path_parent,'DB','DB_inspection',...
        [Name_sub(1:3),'.jpg']));
    close(hf);
end
end

%% 결과 저장
save(fullfile(path2save,sprintf('feat_set_seg_%d',N_seg)),'Features');
set(gca,'FontSize', 12);
% save(fullfile(path_parent,'DB','ProcessedDB',sprintf('feat_set')),...
%     'Features');



