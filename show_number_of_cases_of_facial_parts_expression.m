%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------
name_e = ["가만히 있기";"눈썹 치켜올리기";"찡그리기"];
name_z = ["가만히 있기";"양쪽 입고리";"코찡그림"];
name_l = ["가만히 있기(다물기)";"꽉다물기";"오른쪽 비웃음";"왼쪽 비웃음";"입벌리고 웃기";"아 모양"];

count = 0;
for i_brow = 1 : 3
    for i_z = 1 : 3
        for i_l = 1 : 6
            count = count +1;
            temp{count,1} = sprintf('%s_%s_%s',name_e{i_brow},name_z{i_z},name_l{i_l})
        end
    end
end
