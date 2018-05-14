%--------------------------------------------------------------------------
% Construct machine learning model(SVM) for detecting onset/offset of EMG
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------

%------------------------code analysis parameter--------------------------%
% name of raw DB
name_DB_raw = 'DB_onset';

% name of process DB to analyze in this code
name_DB_process = 'DB_processed2';

% % name of anlaysis DB in the process DB
% name_DB_analy = 'DB_raw2_to_10Hz_cam_winsize_24_wininc_12_emg_winsize_408_wininc_204_delay_0';

%-------------------------------------------------------------------------%

%-------------set paths in compliance with Cha's code structure-----------%

% path of research, which contains toolbox
path_research = fileparts(fileparts(fileparts(fullfile(cd))));

% path of code, which 
path_code = fileparts(fullfile(cd));
path_DB = fullfile(path_code,'DB');
path_DB_raw = fullfile(path_DB,name_DB_raw);
path_DB_process = fullfile(path_DB,name_DB_process);
% path_DB_analy = fullfile(path_DB_process,name_DB_analy);

%-------------------------------------------------------------------------%

%-------------------------add functions-----------------------------------%
addpath(genpath(fullfile(path_research,'_toolbox')));
%-------------------------------------------------------------------------%

%------------------------experiment infromation---------------------------%

%-------------------------------------------------------------------------%


%----------------------------paramters------------------------------------%

%-------------------------------------------------------------------------%

%----------------------set saving folder----------------------------------%

%-------------------------------------------------------------------------%

%----------------------memory allocation for results----------------------%

%-------------------------------------------------------------------------%

%------------------------------------main---------------------------------%
% read file path of data from raw DB
[name_sub,path_sub] = read_names_of_file_in_folder(fullfile(path_DB_raw),'*.mat');

% load labeld DB
DB = cellfun(@load,path_sub);
DB = struct2cell(DB);
cat(1,DB{:});
sum(cellfun(@(x) size(x,1), DB))
%-------------------------------------------------------------------------%

%-------------------------------save results------------------------------%
save(fullfile(path_save,'results.mat'),'r','db');
%-------------------------------------------------------------------------%