plot(groundTruth(:,1),groundTruth(:,2),'r.','MarkerSize',20)
for i = 1: 194
plot(groundTruth(i,1),groundTruth(i,2),'b.','MarkerSize',20)
disp(num2str(i))
pause(1.5);
end