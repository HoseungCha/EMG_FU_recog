%--------------------------------------------------------------------------
% 입모양 사진을 뿌려주는 코드
%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------
clc; close all; clear all;

% enlarge subplot of matlab
id_make_it_tight = true;
subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
if ~id_make_it_tight,  clear subplot;  end

% path Research (연구)
path_research = fileparts(fileparts(fileparts(cd)));

% add path of my functions
addpath(genpath(fullfile(path_research,'_toolbox')));

% path of Code
path_Code = fileparts(cd);

% path of DB of pics
path_pic = fullfile(path_Code,'DB','DB_pic_lips');


% read subject nams and path of that subejct
[name_sub,path_sub] = read_names_of_file_in_folder(path_pic);

% experiment infromation
names_exp = ["화남";"비웃음";"역겨움";"두려움";"행복";"무표정";"슬픔";"놀람"];
n_FE = length(names_exp);
n_sub = length(name_sub);

% plot
figure;
imds = imageDatastore(path_pic,'IncludeSubfolders',true,...
    'FileExtensions','.png');
montage(imds,'Size', [14 8])



