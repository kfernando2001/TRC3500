clc;clear all;close all
syms c m
min=2.611/2;
max=3.246/2;
eqn1= max*m+c==3.0;
eqn2= min*m+c==0.5;
sol=solve([eqn1,eqn2],[m,c]);
msol=sol.m;
csol=sol.c;
fprintf("Rf should be %.2f higher than Rg\n",msol-1)
temp=abs(csol)/((msol-1)*3.3);
R1R2ratio=(1-temp)/temp; 
fprintf("R1 should be %.2f higher than R2",R1R2ratio)
