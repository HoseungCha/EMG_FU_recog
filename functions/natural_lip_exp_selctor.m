% natural lip expression selector with eye brow expression
function y = natural_lip_exp_selctor(output_eyebrow)
switch output_eyebrow
    %-------neutral 
    case 2
        y = [1,2,3,4,5,6,8,9,10,11];
    case 3
        y = [1,2,3,4,5,6,8,9,10,11];
    case 4
        y = [1,2,3,4,5,6,8,9,10,11];
    case 8
        y = [1,2,3,4,5,6,8,9,10,11];
    case 9
        y = [1,2,3,4,5,6,8,9,10,11];
        
    % surprised
    case 11
        y = [9,11];
        
    % happy
    case 7
        y = [5,7];
        
    % angry
    case 1
        y = [1,2,5,6,9,10,11];
    case 5
        y = [1,2,5,6,9,10,11];
        
    %--------sad
    case 6
        y = [1,2,6,9,10,11];
    case 10
        y = [1,2,6,9,10,11];
end
end