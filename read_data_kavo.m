clc;
clear all;
close all;

%Vision Nerve Canal Data Statusfile Version 3.0 
fname='ncData.bin';
fid=fopen(fname);
a=fread(fid,'int');
fclose(fid);

%%
nseg=3;
%npoint=10320;
b=a(20:end);


npoint=length(b)/nseg;
b=reshape(b,[nseg,npoint]);
b=b(:,1:end-1);
[m, n] = max(b(3,:));
b(:,n) = [];
% [m, n] = min(b(3,:));
% b(:,n) = [];
% b=reshape(b,[npoint,nseg]);


% 
% nskip=1;
% figure(2);plot3(b(1,1:nskip:end),b(2,1:nskip:end),b(3,1:nskip:end),'b.');
figure(3);plot3(b(1,1:end),b(2,1:end),b(3,1:end),'.');
% figure(4);subplot(1,2,1);plot(b);subplot(1,2,2);plot(b');


