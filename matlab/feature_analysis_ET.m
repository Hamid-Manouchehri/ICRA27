clc;        clear;       close all;



data_dir = "C:\Users\Hamidjoon\projects\ICRA27\data\processed_data\" + ...
  "features_all_train_W_1s.csv";
data = readtable(data_dir);
save('data','data')


%load('..\data\processed_data\features_all_train_W_1s')
load('data')

Y = data.trial_class_label;
Y = strcmp(Y,'Fine'); % 1 for Fine, 0 for Gross
B = data.bandwidth;
B = strcmp(B,'wide'); % 1 for wide, 0 for narrow
D = data.direction;
D = strcmp(D,'clockwise'); % 1 for clockwise, 0 counter_clockwise

X = zeros(length(Y),5);
X(:,1) = data.robot_path_efficiency;
X(:,2) = data.robot_direction_concentration;
X(:,3) = data.robot_std_curvature;
%X(:,4) = data.robot_std_curvature_dot;
%X(:,4) = data.robot_max_absolute_curvature;
X(:,4) = data.joint_mean_curvature_when_gaze_close;
%X(:,5) = data.joint_mean_curvature_when_gaze_far;
X(:,5) = data.gaze_mean_gaze_lead_along_ee_motion;
buf = isnan(X);
[a b] = find(buf==1);
a = unique(a);

X(a,:)=[]
Y(a,:)=[];
B(a,:)=[];
D(a,:)=[];

corr(X)

%Fine
id_fw = (Y==1).*(B==1);
id_fn = (Y==1).*(B==0);
id_gw = (Y==0).*(B==1);
id_gn = (Y==0).*(B==0);
id_fc = (Y==1).*(D==1);
id_fcc= (Y==1).*(D==0);
id_gc = (Y==0).*(D==1);
id_gcc= (Y==0).*(D==0);



X_f  = X(Y==1,:);
X_fw = X(id_fw==1,:);
X_fn = X(id_fn==1,:);
X_fc = X(id_fc==1,:);
X_fcc= X(id_fcc==1,:);

X_g  = X(Y==0,:);
X_gw = X(id_gw==1,:);
X_gn = X(id_gn==1,:);
X_gc = X(id_gc==1,:);
X_gcc= X(id_gcc==1,:);



for i=1:5
    figure(i)

    r = linspace(min(X(:,i)),max(X(:,i)),300);

    subplot(5,1,1)
    Z = bachat_dis_ET(X_f(:,i),X_g(:,i));
    histogram(X_f(:,i),r,'Normalization','probability');
    hold on
    histogram(X_g(:,i),r,'Normalization','probability');
    title(num2str(Z))

    subplot(5,1,2)
    histogram(X_fn(:,i),r,'Normalization','probability');
    hold on
    histogram(X_gn(:,i),r,'Normalization','probability');

    subplot(5,1,3)
    histogram(X_fw(:,i),r,'Normalization','probability');
    hold on
    histogram(X_gw(:,i),r,'Normalization','probability');

    subplot(5,1,4)
    histogram(X_fc(:,i),r,'Normalization','probability');
    hold on
    histogram(X_gc(:,i),r,'Normalization','probability');

    subplot(5,1,5)
    histogram(X_fcc(:,i),r,'Normalization','probability');
    hold on
    histogram(X_gcc(:,i),r,'Normalization','probability');
end


figure(6)
plot(X_f(:,1),X_f(:,2),'ro')
hold on
plot(X_g(:,1),X_g(:,2),'ks')

Z = pca(X');
Z_f = Z(Y==1,:);
Z_g = Z(Y==0,:);

figure(7)
plot(Z_f(:,1),Z_f(:,2),'ro')
hold on
plot(Z_g(:,1),Z_g(:,2),'ks')


%%
%buf = importdata('..\data\processed_data\features_all_train_W_1s_et.csv');
%gdata = buf.data;
%gName = buf.textdata;
%save('gdata','gdata','gName')


load('gdata')
disp('This is the start')
gdata(a,:)=[];
gdata_f = gdata(Y==1,:);
gdata_g = gdata(Y==0,:);

for i=1:size(gdata,2)
    Zd(i) = bachat_dis_ET(gdata_f(:,i),gdata_g(:,i));
end

figure(8)
plot(Zd)
%}

W = [-0.6552 -0.6548 1.5697 0.3981 -0.6578];  % [Z W] = pca(X')
ZZ = X*W';
%% if ZZ>245 then it is Fine,
r = linspace(min(ZZ),max(ZZ),300);

figure(9)
histogram(ZZ(Y==1,1),r,'Normalization','probability');
hold on
histogram(ZZ(Y==0,1),r,'Normalization','probability');

YY = ZZ>250;
sum(YY==Y)/length(Y)  % accuracy