clc;        clear;       close all;

%data_dir = "C:\Users\Hamidjoon\projects\ICRA27\data\processed_data\" + ...
%  "features_all_train_W_1s.csv"
%data = readtable(data_dir);
%save('data','data')
load('data')

Y = data.trial_class_label;
Y = strcmp(Y,'Fine'); % 1 for Fine, 0 for Gross
B = data.bandwidth;
B = strcmp(B,'wide'); % 1 for wide, 0 for narrow

X = zeros(length(Y),5)
X(:,1) = data.robot_path_efficiency;
X(:,2) = data.robot_direction_concentration;
X(:,3) = data.robot_std_curvature;
%X(:,4) = data.robot_std_curvature_dot;
%X(:,4) = data.robot_max_absolute_curvature;
X(:,4) = data.joint_mean_curvature_when_gaze_close;
X(:,5) = data.joint_mean_curvature_when_gaze_far;
buf = isnan(X);
[a b] = find(buf==1);
a = unique(a);
X(a,:)=[];
Y(a,:)=[];
B(a,:)=[]
corr(X)

%Fine
id_fw = (Y==1).*(B==1);
id_fn = (Y==1).*(B==0);
id_gw = (Y==0).*(B==1);
id_gn = (Y==0).*(B==0);

X_f = X(Y==1,:);    X_fw = X(id_fw==1,:);   X_fn = X(id_fn==1,:);
X_g = X(Y==0,:);    X_gw = X(id_gw==1,:);   X_gn = X(id_gn==1,:);

for i=1:5
    figure(i)
    r = linspace(min(X(:,i)),max(X(:,i)),100);
    subplot(3,1,1)
    histogram(X_f(:,i),r,'Normalization','probability');
    hold on
    histogram(X_g(:,i),r,'Normalization','probability');

    subplot(3,1,2)
    histogram(X_fn(:,i),r,'Normalization','probability');
    hold on
    histogram(X_gn(:,i),r,'Normalization','probability');

    subplot(3,1,3)
    histogram(X_fw(:,i),r,'Normalization','probability');
    hold on
    histogram(X_gw(:,i),r,'Normalization','probability');
end
%{
 figure(6)
plot(X_f(:,1),X_f(:,2),'ro')
hold on
plot(X_g(:,1),X_g(:,2),'ks')

Z = pca(X');
Z_f = Z(Y==1,:);
Z_g = Z(Y==0,:);

figure(7)
plot3(Z_f(:,1),Z_f(:,2),Z_f(:,3),'ro')
hold on
plot3(Z_g(:,1),Z_g(:,2),Z_g(:,3),'ks')
%}