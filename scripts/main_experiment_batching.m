clc
clear all
lw=2
figure(1)
scheme=['rs-';'bh-';'k<-';'ro-';'b>-';'k*-';'r+-';'bs-']
hold off
Data = importdata('../results/paper_experiment_batching.txt'); 
Data=Data.data 
SP=unique(Data(:,6));
SIZES=unique(Data(:,3));
Y=[];
for stp=1:size(SP)
    y=Data(...
    (  Data(:,6)==SP(stp)) & (Data(:,8)==1)   ...
    ,4);
    Y=[Y y];
end
Y=diag(max(Y').^(-1))*Y
figure(2)
 Y=Y.^(-1);
hold off
for i=1:length(SP)
   semilogx(SIZES,Y(:,i),scheme(i,:),'LineWidth',lw,'MarkerSize',10)
   hold on
end
ylim([1, max(max(Y))+0.1])
hold off
Z=Y;
Y=[];
for stp=1:size(SP)
      y=Data(...
    (  Data(:,6)==SP(stp)) & (Data(:,8)~=1)   ...
    ,4);
    Y=[Y y];
end
Y=diag(max(Y').^(-1))*Y
set(gca,'FontSize',25)
 xlabel('p')
 ylabel('Speedup (12 CORES)')
 xlim([min(SIZES), max(SIZES)])
 grid on
 legend('BAT1 = NAI','BAT4','BAT16','BAT64','BAT256 = SFA', 'Orientation','vertical',...
    'Location','NorthWest')
 print('-depsc','-tiff','-r800','batching_12_cores.eps') 
 figure(1)
 hold off
Y=Y.^(-1);
for i=1:length(SP)
   semilogx(SIZES,Y(:,i),scheme(i,:),'LineWidth',lw,'MarkerSize',10)
   hold on
end
ylim([1, max(max(Y))+1])
XX=[Z';Y']
set(gca,'FontSize',25)
legend('BAT1 = NAI','BAT4','BAT16','BAT64','BAT256 = SFA', 'Orientation','vertical',...
    'Location','NorthWest')
xlim([min(SIZES), max(SIZES)])
ylim([1, max(max(Y))+1])
xlabel('p')
ylabel('Speedup (1 CORE)')
grid on
print('-depsc','-tiff','-r800','batching_1_core.eps') 