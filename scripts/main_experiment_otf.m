clc
clear all
scheme=['rh-';'bs-';'ko-';'r>-';'bs-']
Data = importdata('../results/paper_experiment_otf.txt'); 
Data = Data.data
n=unique(Data(:,3));
sizes=Data(:,3);
loops = Data(:,10);
batchsize=Data(:,6);
batches=unique(batchsize)
onthefly=Data(:,7);
duration=Data(:,4)
plotdata=[];
durations=[];
for i =1:length(n)
   siz=n(i);
   for j=1:5
       switch j
           case 1,
               bs=64
               onth=0
           case 2,
               bs=64
               onth=1
            case 3
               bs=1024
               onth=0   
       end
       ID = sizes==siz  & onthefly==onth & batchsize==bs
       loops(ID)*bs/1024
       plotdata(i,j)=  loops(ID)*bs/1024; 
       durations (i,j)= duration(ID); 
   end
end
figure(1)
set(gca,'FontSize',25)
hold off
for i=[3 2 1]
    i
    semilogx(n,plotdata(:,i),scheme(i,:),'MarkerSize',10,'LineWidth',2)
hold on
end
xlim([min(n),max(n)]) 
legend('BAT1024 = SFA','BAT64','OTF64','Location','NorthWest')
ylabel('Average number of iter/SP')
xlabel('p')
grid on
print('-depsc','-tiff','-r400','otf_average.eps')  
figure(2)
set(gca,'FontSize',25)
hold off
for i= [3 2 1]
semilogx(n,(durations(:,i)./durations(:,3)).^(-1),scheme(i,:),'MarkerSize',10,'LineWidth',2)
hold on
end
xlim([min(n),max(n)])
ylabel('Speedup')
xlabel('p')
legend( 'BAT1024 = SFA','BAT64','OTF64',   'Location','NorthWest')
grid on
print('-depsc','-tiff','-r400','otf_speedup.eps')  