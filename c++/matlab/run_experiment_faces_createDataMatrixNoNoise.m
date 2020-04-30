
clc
clear all

orl_folder = '../datasets/orl_faces/s'
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
A=[];
B=[];

a=20;
b=30;

 

por=1;
for folder=1:10
   for im=1:10
      
     file =[orl_folder sprintf('%d',folder) '/' sprintf('%d',im) '.pgm'];
     photo = imread(file);
      xx = (double(photo) + 1)/255;
      A=[A; xx(:)'];
      
      zz=xx;
      [m,n]=size(zz)
       
      
      B=[B; zz(:)'];
      por=por+1;
      
     %imshow(xx/255)
   end
end 

save('orl_faces2.mat')