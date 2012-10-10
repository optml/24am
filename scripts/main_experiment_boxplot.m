clc 
clear all
data=csvread('../results/paper_experiment_boxplots.txt');
data=data(1:end,:);
vals=data(:,2:1001);
cons=data(1:end,1);
pp=[]
valsAll=[];
idx=1;
for i = length(cons):-1:1
    pp=[pp; ones(1000,1)*cons(i)];
    vals(i,:)=vals(i,:)/max(vals(i,:));
    valsAll=[valsAll; vals(i,:)'];
end
%%
figure(1) 
set(gca,'FontSize',14)
boxplot(valsAll(:),pp(:),'notch','on')
%xticklabel_rotate(pp,90)
xlabel('Target sparsity level s')
ylabel('Explained variance / Best explained variance')
set(findobj(gca,'Type','text'),'FontSize',14)
print('-depsc','-tiff','-r800','boxplot.eps') 

 
 
 