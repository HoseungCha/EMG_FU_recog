%--------------------------------------------------------------------------
% developed by Ho-Seung Cha, Ph.D Student,
% CONE Lab, Biomedical Engineering Dept. Hanyang University
% under supervison of Prof. Chang-Hwan Im
% All rights are reserved to the author and the laboratory
% contact: hoseungcha@gmail.com
%--------------------------------------------------------------------------
name_e = ["������ �ֱ�";"���� ġ�ѿø���";"���׸���"];
name_z = ["������ �ֱ�";"���� �԰�";"�����׸�"];
name_l = ["������ �ֱ�(�ٹ���)";"�˴ٹ���";"������ �����";"���� �����";"�Թ����� ����";"�� ���"];

count = 0;
for i_brow = 1 : 3
    for i_z = 1 : 3
        for i_l = 1 : 6
            count = count +1;
            temp{count,1} = sprintf('%s_%s_%s',name_e{i_brow},name_z{i_z},name_l{i_l})
        end
    end
end
