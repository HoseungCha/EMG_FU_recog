%--------------------------------------------------------------------------
% plot all expressions of avartars, which I developed
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------

%==============eye brow control============================%
% One of them should be selected for id_eyebrow 
% neutral
% eye_brow_up
% eye_brow_down
%==========================================================%

%==============zygomaticus control=========================%
% One of them should be selected for id_zygo 
% neutral
% nose_wrinkle
%==========================================================%

%==============lips control================================%
% One of them should be selected for id_lips
% neutral
% lip_corner_up_left
% lip_corner_up_right
% lip_corner_up_both
% clench
% lip_stretch_down
% kiss
% lip_open
% lip_tighten
%==========================================================%

%------------------------code analysis parameter--------------------------%
% name of process DB to analyze in this code
name_DB_process = 'DB_processed2';

name_folder4saving = 'img_avartar_expressions';
%-------------------------------------------------------------------------%

%-------------set paths in compliance with Cha's code structure-----------%
% path of code, which 
path_code = fileparts(fullfile(cd));
path_DB = fullfile(path_code,'DB');
path_DB_process = fullfile(path_DB,name_DB_process);
%-------------------------------------------------------------------------%

%----------------------------paramters------------------------------------%
name_eyebrow_control = {'eye_brow_sad'};
name_nose_control = {'neutral','nose_wrinkle'};
name_lip_control = {'neutral','lip_corner_up_left','lip_corner_up_right',...
    'lip_corner_up_both','clench','lip_stretch_down',...
    'kiss','lip_open','lip_tighten'};
%-------------------------------------------------------------------------%

%------------------------------------main---------------------------------%
for i_brow = 1 : length(name_eyebrow_control)
for i_nose = 1 : length(name_nose_control)
for i_lip = 1 : length(name_lip_control)
    figure
    tmp = sprintf('|eye brow:%s| |zygo:%s| |lip:%s|',...
        name_eyebrow_control{i_brow},...
        name_nose_control{i_nose},...
        name_lip_control{i_lip});
    tmp = strrep(tmp,'_', ' ');
    title(tmp);
    plot_avartar(...
        name_eyebrow_control{i_brow},...
        name_nose_control{i_nose},...
        name_lip_control{i_lip})
    c = getframe(gcf);
    
    % path for saving
    path_DB_save = make_path_n_retrun_the_path(fullfile(path_DB_process),...
    name_folder4saving);

    tmp = sprintf('eye_brow-%s_zygo_%s_lip_%s',...
            name_eyebrow_control{i_brow},...
            name_nose_control{i_nose},...
            name_lip_control{i_lip});
        
    imwrite(c.cdata,fullfile(path_DB_save,[tmp,'.png']));
    close
    
end
end
end
%-------------------------------------------------------------------------%