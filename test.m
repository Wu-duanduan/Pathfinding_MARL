clc;clear;close all;
pathMatrix1 = csvread("./MADDPG_data_csv/pathMatrix1.csv");
pathMatrix2 = csvread("./MADDPG_data_csv/pathMatrix2.csv");
pathMatrix3 = csvread("./MADDPG_data_csv/pathMatrix3.csv");
pathMatrix4 = csvread("./MADDPG_data_csv/pathMatrix4.csv");
pathMatrix5 = csvread("./MADDPG_data_csv/pathMatrix5.csv");
pathMatrix6 = csvread("./MADDPG_data_csv/pathMatrix6.csv");
pathMatrix7 = csvread("./MADDPG_data_csv/pathMatrix7.csv");
pathMatrix8 = csvread("./MADDPG_data_csv/pathMatrix8.csv");

obsMatrix1 = csvread("./data_csv/obs_trace1.csv");
obsMatrix2 = csvread("./data_csv/obs_trace2.csv");

start1 = csvread("./data_csv/start1.csv");
goal1 = csvread("./data_csv/goal1.csv");
start2 = csvread("./data_csv/start2.csv");
goal2 = csvread("./data_csv/goal2.csv");
start3 = csvread("./data_csv/start3.csv");
goal3 = csvread("./data_csv/goal3.csv");
start4 = csvread("./data_csv/start4.csv");
goal4 = csvread("./data_csv/goal4.csv");
start5 = csvread("./data_csv/start5.csv");
goal5 = csvread("./data_csv/goal5.csv");
start6 = csvread("./data_csv/start6.csv");
goal6 = csvread("./data_csv/goal6.csv");
start7 = csvread("./data_csv/start7.csv");
goal7 = csvread("./data_csv/goal7.csv");
start8 = csvread("./data_csv/start8.csv");
goal8 = csvread("./data_csv/goal8.csv");

cylinderR = csvread("./data_csv/cylinder_r.csv");           % 动态障碍物的半径
cylinderH = csvread("./data_csv/cylinder_h.csv");
scatter3(start1(1),start1(2),start1(3),60,"cyan",'filled','o','MarkerEdgeColor','k');hold on
scatter3(start2(1),start2(2),start2(3),60,"cyan",'filled','o','MarkerEdgeColor','k');
scatter3(start3(1),start3(2),start3(3),60,"cyan",'filled','o','MarkerEdgeColor','k');
scatter3(start4(1),start4(2),start4(3),60,"cyan",'filled','o','MarkerEdgeColor','k');
scatter3(start5(1),start5(2),start5(3),60,"cyan",'filled','o','MarkerEdgeColor','k');
scatter3(start6(1),start6(2),start6(3),60,"cyan",'filled','o','MarkerEdgeColor','k');
scatter3(start7(1),start7(2),start7(3),60,"cyan",'filled','o','MarkerEdgeColor','k');
scatter3(start8(1),start8(2),start8(3),60,"cyan",'filled','o','MarkerEdgeColor','k');

% scatter3(start1(1),start1(2),start1(3),60,"magenta",'filled','o','MarkerEdgeColor','k');hold on
% scatter3(start2(1),start2(2),start2(3),60,"magenta",'filled','o','MarkerEdgeColor','k');
% scatter3(start3(1),start3(2),start3(3),60,"magenta",'filled','o','MarkerEdgeColor','k');
% scatter3(start4(1),start4(2),start4(3),60,"magenta",'filled','o','MarkerEdgeColor','k');
% scatter3(start5(1),start5(2),start5(3),60,"magenta",'filled','o','MarkerEdgeColor','k');
% scatter3(start6(1),start6(2),start6(3),60,"magenta",'filled','o','MarkerEdgeColor','k');
% scatter3(start7(1),start7(2),start7(3),60,"magenta",'filled','o','MarkerEdgeColor','k');
% scatter3(start8(1),start8(2),start8(3),60,"magenta",'filled','o','MarkerEdgeColor','k');

scatter3(goal1(1),goal1(2),goal1(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k');
scatter3(goal2(1),goal2(2),goal2(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k');
scatter3(goal3(1),goal3(2),goal3(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k');
scatter3(goal4(1),goal4(2),goal4(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k');
scatter3(goal5(1),goal5(2),goal5(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k');
scatter3(goal6(1),goal6(2),goal6(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k');
scatter3(goal7(1),goal7(2),goal7(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k');
scatter3(goal8(1),goal8(2),goal8(3),60,"magenta",'filled',"o",'MarkerEdgeColor','k')

% text(start1(1),start1(2),start1(3),'  Start1 & End3','FontName','Times New Roman','FontWeight','bold');
% text(start2(1),start2(2),start2(3),'  Start2','FontName','Times New Roman','FontWeight','bold');
% text(goal1(1),goal1(2),goal1(3),'  Start3 & End1','FontName','Times New Roman','FontWeight','bold');
% text(goal2(1),goal2(2),goal2(3),'  End2','FontName','Times New Roman','FontWeight','bold');
xlabel('x(m)','FontWeight','bold'); ylabel('y(m)','FontWeight','bold'); zlabel('z(m)','FontWeight','bold');
title('UAV trajectory planning path','FontName','Times New Roman','FontWeight','bold'); axis equal;
set(gca,'fontsize',16,'FontName','Times New Roman','FontWeight','bold');%设置坐标轴字体大小
set(gca,'fontsize',16,'FontName','Times New Roman','FontWeight','bold');%设置坐标轴字体大小
timeStep = 0.1;
[n,~] = size(pathMatrix1);
for i = 1:n-1
    if i <= length(obsMatrix1)
        j = i;
    else
        j = length(obsMatrix1);
    end
    if i <= length(obsMatrix2)
        k = i;
    else
        k = length(obsMatrix2);
    end

    obsCenter1 = [obsMatrix1(j,1),obsMatrix1(j,2),obsMatrix1(j,3)];
    obsCenter2 = [obsMatrix2(k,1),obsMatrix2(k,2),obsMatrix2(k,3)];

    try delete(B1), catch, end
    try delete(B2), catch, end
    try delete(B3), catch, end
    try delete(B4), catch, end
    try delete(B5), catch, end
    try delete(B6), catch, end
    try delete(B7), catch, end
    try delete(B8), catch, end
    try delete(B9), catch, end
    try delete(B10), catch, end

    B1 = drawCylinder(obsCenter1, cylinderR, cylinderH);
    B2 = drawCylinder(obsCenter2, cylinderR, cylinderH);

    B3 = scatter3(pathMatrix1(i,1),pathMatrix1(i,2),pathMatrix1(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
    B4 = scatter3(pathMatrix2(i,1),pathMatrix2(i,2),pathMatrix2(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
    B5 = scatter3(pathMatrix3(i,1),pathMatrix3(i,2),pathMatrix3(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
    B6 = scatter3(pathMatrix4(i,1),pathMatrix4(i,2),pathMatrix4(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
    B7 = scatter3(pathMatrix5(i,1),pathMatrix5(i,2),pathMatrix5(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
    B8 = scatter3(pathMatrix6(i,1),pathMatrix6(i,2),pathMatrix6(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
    B9 = scatter3(pathMatrix7(i,1),pathMatrix7(i,2),pathMatrix7(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
    B10 = scatter3(pathMatrix8(i,1),pathMatrix8(i,2),pathMatrix8(i,3),80,'filled',"^",'MarkerFaceColor','r'...
                  ,'MarkerEdgeColor','k');
   
              
    if i >1
        b1 = plot3([obsMatrix1(j-1,1),obsMatrix1(j,1)],[obsMatrix1(j-1,2),obsMatrix1(j,2)]...
              ,[obsMatrix1(j-1,3),obsMatrix1(j,3)],'LineWidth',2,'color','c');
        b2 = plot3([obsMatrix2(k-1,1),obsMatrix2(k,1)],[obsMatrix2(k-1,2),obsMatrix2(k,2)]...
              ,[obsMatrix2(k-1,3),obsMatrix2(k,3)],'LineWidth',2,'color','c');
    end
    drawnow;

    b3 = plot3([pathMatrix1(i,1),pathMatrix1(i+1,1)],[pathMatrix1(i,2),pathMatrix1(i+1,2)],[pathMatrix1(i,3),pathMatrix1(i+1,3)],'LineWidth',2,'Color','g');
    b4 = plot3([pathMatrix2(i,1),pathMatrix2(i+1,1)],[pathMatrix2(i,2),pathMatrix2(i+1,2)],[pathMatrix2(i,3),pathMatrix2(i+1,3)],'LineWidth',2,'Color','g');
    b5 = plot3([pathMatrix3(i,1),pathMatrix3(i+1,1)],[pathMatrix3(i,2),pathMatrix3(i+1,2)],[pathMatrix3(i,3),pathMatrix3(i+1,3)],'LineWidth',2,'Color','g');
    b6 = plot3([pathMatrix4(i,1),pathMatrix4(i+1,1)],[pathMatrix4(i,2),pathMatrix4(i+1,2)],[pathMatrix4(i,3),pathMatrix4(i+1,3)],'LineWidth',2,'Color','g');
    b7 = plot3([pathMatrix5(i,1),pathMatrix5(i+1,1)],[pathMatrix5(i,2),pathMatrix5(i+1,2)],[pathMatrix5(i,3),pathMatrix5(i+1,3)],'LineWidth',2,'Color','g');
    b8 = plot3([pathMatrix6(i,1),pathMatrix6(i+1,1)],[pathMatrix6(i,2),pathMatrix6(i+1,2)],[pathMatrix6(i,3),pathMatrix6(i+1,3)],'LineWidth',2,'Color','g');
    b9 = plot3([pathMatrix7(i,1),pathMatrix7(i+1,1)],[pathMatrix7(i,2),pathMatrix7(i+1,2)],[pathMatrix7(i,3),pathMatrix7(i+1,3)],'LineWidth',2,'Color','g');
    b10 = plot3([pathMatrix8(i,1),pathMatrix8(i+1,1)],[pathMatrix8(i,2),pathMatrix8(i+1,2)],[pathMatrix8(i,3),pathMatrix8(i+1,3)],'LineWidth',2,'Color','g');
    
%     if i == 2
%         legend([b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,B3,B4,B5],["Obstacle trajectory1","Obstacle trajectory2","UAV planning path1","UAV planning path2","UAV planning path3","UAV1","UAV2","UAV3"],'FontName','Times New Roman','FontWeight','bold','AutoUpdate','off','Location','best')
%     end
end
%% 计算GS,LS,L
% pathLength = 0;
% for i=1:length(pathMatrix(:,1))-1, pathLength = pathLength + distanceCost(pathMatrix(i,1:3),pathMatrix(i+1,1:3)); end
% fprintf("航路长度为:%f\n GS:%f °\n LS:%f °",pathLength, calGs(pathMatrix)/pi*180, calLs(pathMatrix)/pi*180);
%% 函数
% 球绘制函数
function bar = drawSphere(pos, r)
[x,y,z] = sphere(60);
bar = surfc(r*x+pos(1), r*y+pos(2), r*z+pos(3));
hold on;
end
function bar = drawCylinder(pos, r, h)
[x,y,z] = cylinder(r,40);
z(2,:) = h;
bar = surfc(x + pos(1),y + pos(2),z,'FaceColor','interp');hold on;

% theta = linspace(0,2*pi,40);
% X = r * cos(theta) + pos(1);
% Y = r * sin(theta) + pos(2);
% Z = ones(size(X)) * h;
% fill3(X,Y,Z,[0 0.5 1]); % 顶盖
% fill3(X,Y,zeros(size(X)),[0 0.5 1]); % 底盖
end
% 欧式距离求解函数
function h=distanceCost(a,b)
h = sqrt(sum((a-b).^2, 2));
end