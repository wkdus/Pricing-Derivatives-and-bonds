# -*- coding: utf-8 -*-
"""
시뮬레이션

"""
import numpy as np
import matplotlib.pyplot as plt

#변수들 대입
S=100
K=100
T=30/365
r=0
v=0.01
kappa=2
theta=0.01
Lambda=0
sigma=0.1
rho=0
call =1.1345
N=100

#Heston 함수 만들기
def Heston_MCS(Se,K,T,r,vol,kappa,theta,Lambda,sigma,rho,N,MCS):
    payoffs=np.zeros(MCS)
    dt=T/N#T를 N등분하고
    for j in range(MCS):#시뮬레이션 횟수
        #다시 초기화
        S=Se
        v=vol
        for i in range(N):#dt가 N회 움직인 뒤
            ev=np.random.randn()
            e=np.random.randn()
            es=rho*ev+np.sqrt(1-rho**2)*e
            S=np.exp(np.log(S)+(r-0.5*max(v,0))*dt+np.sqrt(max(v,0)*dt)*es)
            v=v+(kappa+Lambda)*((kappa*theta/(kappa+Lambda))-max(v,0))*dt+sigma*np.sqrt(max(v,0))*np.sqrt(dt)*ev
        payoffs[j]=max(S-K,0)
    err=np.std(payoffs)/np.sqrt(MCS)
    price=np.mean(payoffs)*np.exp(-r*T)
    return price, err

#실제 계산
M=np.ones(10,dtype=int)
c1=np.zeros(10)
c2=np.zeros(10)
for i in range(10):
    M[i]=1000*(i+1)
    price,err=Heston_MCS(S,K,T,r,v,kappa,theta,Lambda,sigma,rho,N,M[i])
    c1[i]=price+1.96*err
    c2[i]=price-1.96*err

#그래프그리기
fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(M,np.ones(10)*call)
ax.plot(M,c1)
ax.plot(M,c2)
plt.show()