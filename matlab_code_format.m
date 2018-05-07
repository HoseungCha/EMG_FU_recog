%--------------------------------------------------------------------------
% explanation of this code
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------

%------------------------code analysis parameter--------------------------%
% name of raw DB
name_DB_raw = 'DB_raw2';

% name of process DB to analyze in this code
name_DB_process = 'DB_processed2';

% name of anlaysis DB in the process DB
name_DB_analy = 'DB_raw2_to_10Hz_cam_winsize_24_wininc_12_emg_winsize_408_wininc_204_delay_0';

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
addpath(genpath(path_research,'_toolbox'));
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

%-------------------------------------------------------------------------%

%-------------------------------save results------------------------------%
save(fullfile(path_save,'results.mat'),'r','db');
%-------------------------------------------------------------------------%