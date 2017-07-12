function [] = analysis(index)
maxloc= load(strcat('maxloc_'+string(index)));
theta= load(strcat('theta_'+string(index)));
theta=reshape(theta, size(theta,1)*size(theta,2),1);
theta = (mapminmax(theta')'+1)/2;
figure(1);
imshow(reshape(theta, int32(length(theta)^0.5), int32(length(theta)^0.5)));
distribution = zeros(max(maxloc)-min(maxloc)+1);
minval = min(maxloc)-1;
distance = zeros(size(maxloc,1),1);
for i=1:size(maxloc,1)
    distribution(int32(maxloc(i,1))-minval(1), int32(maxloc(i,2))-minval(2)) = ...
    distribution(int32(maxloc(i,1))-minval(1), int32(maxloc(i,2))-minval(2))+1;
    distance(i) = (maxloc(i,1)*maxloc(i,1)+maxloc(i,2)*maxloc(i,2))^0.5;
end

disp('average distance = ');
disp(mean(distance));
disp('distance<5 propotion = ');
disp(sum(distance<5)/length(distance));
figure(2);
bar3(distribution);