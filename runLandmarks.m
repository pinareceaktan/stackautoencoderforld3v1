
box_color = {'yellow'};

figure
imshow(face)
hold on
plot(groundTruth(:,1),groundTruth(:,2),'r.','MarkerSize',20)
for i = 1: 194
position = [groundTruth(i,1) groundTruth(i,2)];
RGB = insertText(face,position,num2str(i),'FontSize',18,'BoxColor',box_color,'BoxOpacity',0.4,'TextColor','white');
plot(groundTruth(i,1),groundTruth(i,2),'y.','MarkerSize',10)

disp(num2str(i))
pause(1.5);
end