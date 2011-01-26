function [ class1Prj class2Prj ] = fisherdisc( class1, class2 )
% Projects high dimensional data into 1d along direction specified by
% Fisher discriminant
% Expects class1 and class2 data to be in form NxM with N samples and M
% features.
class1 = tnt;
class2 = nontnt;
% Fisher Discriminant
musC1 = mean(class1);
musC2 = mean(class2);

% Compute deviations from mean (x-m)
devC1 = (class1 - repmat(musC1,size(class1,1),1))';
devC2 = (class2 - repmat(musC2,size(class2,1),1))';

% Compute within class variance and weights
sw = (devC1*devC1') + (devC2*devC2');
w = pinv(sw)*(musC1' - musC2');

% Project to 1D
class1Prj = w'*class1';
class2Prj = w'*class2';