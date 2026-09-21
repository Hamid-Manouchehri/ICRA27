clc;        close all;          clear all;

load('data_X.mat');

subID = X(:,1);     shapeID = X(:,2);        bandID = X(:,3);

mE_PerDiff = (X(:,4)-X(:,6))./X(:,4);
dsj_PerDiff = (X(:,5)-X(:,7))./X(:,5);

stem(mE_PerDiff,'o'); hold on; plot(0*mE_PerDiff);
plot(1+subID/10); plot(2+shapeID/10); plot(3+bandID/10)

figure
stem(dsj_PerDiff,'o'); hold on; plot(0*mE_PerDiff);
plot(1+subID/10); plot(2+shapeID/10); plot(3+bandID/10)

for i=1:8
    sub_m(i,1) = mean(mE_PerDiff(subID==i));
    sub_m(i,2) = mean(dsj_PerDiff(subID==i));
end

for i=1:3
    shape_m(i,1) = mean(mE_PerDiff(shapeID==i));
    shape_m(i,2) = mean(dsj_PerDiff(shapeID==i));
end

for i=1:2
    band_m(i,1) = mean(mE_PerDiff(bandID==i));
    band_m(i,2) = mean(dsj_PerDiff(bandID==i));
end
