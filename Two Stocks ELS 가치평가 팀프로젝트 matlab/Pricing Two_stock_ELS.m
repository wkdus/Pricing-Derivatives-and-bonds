function two_stock_ELS
clc;
clear all;
%%% Parameters %%%
A = xlsread('HSCEI'); % HCSEI excel file
B = xlsread('EUROSTOXXX50'); % Eurostoxxx50 excel file

A_log = log(A(1:end-1,1) ./ A(2:end,1)); % log return of HCEI
B_log = log(B(1:end-1,1) ./ B(2:end,1)); % log return of Eurostoxxx50

sig1 = std(A_log)*sqrt(252); sig2 = std(B_log)*sqrt(252); rho = 0.473;
r = 0.0154; % 3���� CD �ݸ�

T = 3;
c = [0.1860 0.1550 0.1240 0.0930 0.0620 0.0310]; % �����ȯ coupon (�ð� ����)
K = [0.80 0.85 0.85 0.90 0.90 0.90]; % �����ȯ barrier (�ð� ����)
KI = 0.45; % knock-in barrier
d = 0.1860; % knock-in ���� �ʾ��� ���� coupon

%%%%%%%%%%%%%%%%%%%% Price by using FDM %%%%%%%%%%%%%%%%%%%%%
[mesh, x, y] = FDM(T,c,K,KI,d,sig1,sig2,rho,r);
% mesh = ���� ���������� ELS ������ mesh grid
% x, y = �� �����ڻ��� ���� linspace (percentage�� normalize �� ��)

price = mesh(find(x==100,1), find(y==100,1));

%%%%%%%%%%%%%%%%%%%%%% Greeks by using FDM %%%%%%%%%%%%%%%%%%%
% delta (by central difference scheme)
delta_x = (mesh(find(x==100,1)+1, find(y==100,1)) - mesh(find(x==100,1)-1, find(y==100,1))) / (2*h);
delta_y = (mesh(find(x==100,1), find(y==100,1)+1) - mesh(find(x==100,1), find(y==100,1)-1)) / (2*h);

% gamma
gamma_x = (mesh(find(x==100,1)+1, find(y==100,1)) - 2*mesh(find(x==100,1), find(y==100,1)) + mesh(find(x==100,1)-1, find(y==100,1))) / h^2;
gamma_y = (mesh(find(x==100,1), find(y==100,1)+1) - 2*mesh(find(x==100,1), find(y==100,1)) + mesh(find(x==100,1), find(y==100,1)-1)) / h^2;

% vega (�� �ڻ��� �������� 1%�� ���� ���� price ��ȭ������ ���)
[mesh_vega_x, x, y] = FDM(T,c,K,KI,d,sig1+0.01,sig2,rho,r);
[mesh_vega_y, x, y] = FDM(T,c,K,KI,d,sig1,sig2+0.01,rho,r);
vega_x = mesh_vega_x(find(x==100,1), find(y==100,1)) - mesh(find(x==100,1), find(y==100,1));
vega_y = mesh_vega_y(find(x==100,1), find(y==100,1)) - mesh(find(x==100,1), find(y==100,1));

% rho (risk-free rate�� 1% ���� ���� price ��ȭ������ ���)
[mesh_rho, x, y] = FDM(T,c,K,KI,d,sig1,sig2,rho,r+0.01);
rho_greek = mesh_rho(find(x==100,1), find(y==100,1)) - mesh(find(x==100,1), find(y==100,1));

%%%%%%%%%%%%%%%%%%%% Price by using Simulation %%%%%%%%%%%%%%%%%%%%%
M = 10000; % simulation Ƚ��
simul = simulation(T,c,K,KI,d,sig1,sig2,rho,r,M);

fprintf('\n');
fprintf('ELS price');fprintf('\n');
fprintf('FDM           ');fprintf(' %8.4f',price);fprintf('\n');
fprintf('Simulation    ');fprintf(' %8.4f',simul);fprintf('\n');
fprintf('\n');
fprintf('Greeks');fprintf('\n');
fprintf('Delta of x    ');fprintf(' %8.4f',delta_x);fprintf('\n');
fprintf('Delta of y    ');fprintf(' %8.4f',delta_y);fprintf('\n');
fprintf('Gamma of x    ');fprintf(' %8.4f',gamma_x);fprintf('\n');
fprintf('Gamma of y    ');fprintf(' %8.4f',gamma_y);fprintf('\n');
fprintf('Vega of x     ');fprintf(' %8.4f',vega_x);fprintf('\n');
fprintf('Vega of y     ');fprintf(' %8.4f',vega_y);fprintf('\n');
fprintf('Rho           ');fprintf(' %8.4f',rho_greek);fprintf('\n');

function [mesh, x, y] = FDM(T,c,K,KI,d,sig1,sig2,rho,r)
%%% FDM�� ���� ���� ����
Smax = 300; Smin = 0; % mesh grid max, min�� (�� �ڻ� ���� ����)
K0 = 100; % �ڻ��� ���� ����. (�Ѵ� �����ϰ� 100���� ����)
Nt = 100;
Nx = 300; Ny = Nx;
dt = T/Nt;
h = (Smax - Smin)/Nx; % space step for x
k = (Smax - Smin)/Ny; % space step for y

x = Smin-h:h:Smax; y = x; % domain for x,y,    1 x (Nx+1)  &  1 x (Ny+2) matrix
u = zeros(Nx+2,Ny+2); u0 = u; % path ���� knock-in ������ �������� �ʾ������� mesh grid, (Nx+2, Ny+2) matrix
v = zeros(Nx+2,Ny+2); v0 = v; % path ���� knock-in ������ �������� ���� mesh grid, (Nx+2, Ny+2) matrix
weight = 0.15; % simulation�� ���� knock-in barrier���� ���� Ƚ���� weight�� ���
fx = zeros(1,Nx); fy = zeros(1,Ny); % 1 x Nx  &  1 x Ny matrix

early = [round(Nt/6) round(2*Nt/6) round(3*Nt/6) round(4*Nt/6) round(5*Nt/6) Nt+2];
iter = 1; % �� �����ȯ�� ���� iteration ����

%%% make diagonals of f&g (for using tridiagonal solver)
% f
diag_x1 = 1/dt + (sig1*x(2:end-1)/h).^2 + 0.5*r;
diag_x2 = -0.5*(sig1*x(2:end-1)/h).^2 + 0.5*r*x(2:end-1)/h; % ����� �ٲ��� ����...
diag_x3 = -0.5*(sig1*x(2:end-1)/h).^2 - 0.5*r*x(2:end-1)/h;

diag_x1(1) = 2*diag_x2(1) + diag_x1(1); diag_x1(end) = diag_x1(end) + 2*diag_x3(end);
diag_x2(end) = diag_x2(end) - diag_x3(end);
diag_x3(1) = diag_x3(1) - diag_x2(1);
% g
diag_y1 = 1/dt + (sig2*y(2:end-1)/k).^2 + 0.5*r;
diag_y2 = -0.5*(sig2*y(2:end-1)/k).^2 + 0.5*r*y(2:end-1)/k;
diag_y3 = -0.5*(sig2*y(2:end-1)/k).^2 - 0.5*r*y(2:end-1)/k;

diag_y1(1) = 2*diag_y2(1) + diag_y1(1); diag_y1(end) = diag_y1(end) + 2*diag_y3(end);
diag_y2(end) = diag_y2(end) - diag_y3(end);
diag_y3(1) = diag_y3(1) - diag_y2(1);

%%% boundary condition at maturity
for i = 2:Nx+1
    for j = 2:Ny+1
        if x(i)<KI*K0 || y(j)<KI*K0
            u0(i,j) = exp(r*T)*min(x(i), y(j));
            v0(i,j) = exp(r*T)*min(x(i), y(j));
        elseif x(i)<=K(1)*K0 || y(j)<=K(1)*K0
            u0(i,j) = exp(r*T)*K0*(1+d);
            v0(i,j) = exp(r*T)*min(x(i), y(j));
        else
            u0(i,j) = exp(r*T)*K0*(1+c(1));
            v0(i,j) = exp(r*T)*K0*(1+c(1));
        end
    end
end
u = u0; v = v0; % mesh update

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Time loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:Nt
    % �����ȯ���� �������� ��
    if k == early(iter)
        x_line = find(x>K(iter+1)*K0, 1); % �����ȯ �踮�� �ѱ� �����ϴ� x ����
        y_line = find(y>K(iter+1)*K0, 1); % �����ȯ �踮�� �ѱ� �����ϴ� y ����
        u0(x_line:end-1, y_line:end-1) = exp(r*(T-early(iter)*3/Nt))*K0*(1+c(iter+1)); % �����ȯ �Ǵ� mesh payoff (linear boundary condition�� ���� grid ���� �ڸ��� ���� ä�� ����)
        v0(x_line:end-1, y_line:end-1) = exp(r*(T-early(iter)*3/Nt))*K0*(1+c(iter+1));
        
        iter = iter+1;
    end
    u = u0; v = v0; % mesh update
    
    %%%%%%%%% Iteration for u (knock-in barrier�� ġ�� ���� ���� ���)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% x - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear boundary condition
    u(1,2:end-1) = 2*u(2,2:end-1) - u(3,2:end-1); % u(t+1) - u(t) = u(t) - u(t-1) �� �̿��ؼ� ù row���� ��
    u(end,2:end-1) = 2*u(end-1,2:end-1) - u(end-2,2:end-1); % ���� �Ȱ��� ������ row���� ��
    u(:,1) = 2*u(:,2) - u(:,3); % �Ȱ��� ������� ù column ���� ��
    u(:,end) = 2*u(:,end-1) - u(:, end-2); % �Ȱ��� ������ column ���� ��
    
    % linear boundary condition�� ������ grid�� ä�� ����
    for j = 2:Ny+1
        for i = 2:Nx+1
            fx(i-1) = 1/2*rho*sig1*sig2*x(i)*y(j) * (u(i+1,j+1) - u(i+1,j-1) - u(i-1,j+1) + u(i-1,j-1))/(4*h*k) + u(i,j)/dt;
        end
        u0(2:end-1,j) = tridiag(diag_x1, diag_x2, diag_x3, fx);
    end
    u = u0; % mesh update
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% y - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear boundary condition
    u(1,2:end-1) = 2*u(2,2:end-1) - u(3,2:end-1); % u(t+1) - u(t) = u(t) - u(t-1) �� �̿��ؼ� ù row���� ��
    u(end,2:end-1) = 2*u(end-1,2:end-1) - u(end-2,2:end-1); % ���� �Ȱ��� ������ row���� ��
    u(:,1) = 2*u(:,2) - u(:,3); % �Ȱ��� ������� ù column ���� ��
    u(:,end) = 2*u(:,end-1) - u(:, end-2); % �Ȱ��� ������ column ���� ��
    
    for i = 2:Nx+1
        for j = 2:Ny+1
            fy(j-1) = 1/2*rho*sig1*sig2*x(i)*y(j) * (u(i+1,j+1) - u(i+1,j-1) - u(i-1,j+1) + u(i-1,j-1))/(4*h*k) + u(i,j)/dt;
        end
        u0(i,2:end-1) = tridiag(diag_y1, diag_y2, diag_y3, fy);
    end
    
    %%%%%%%%% Iteration for v (knock-in barrier�� ġ�� ���� ���)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% x - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear boundary condition
    v(1,2:end-1) = 2*v(2,2:end-1) - v(3,2:end-1); % v(t+1) - v(t) = v(t) - v(t-1) �� �̿��ؼ� ù row���� ��
    v(end,2:end-1) = 2*v(end-1,2:end-1) - v(Nx,2:end-1); % ���� �Ȱ��� ������ row���� ��
    v(:,1) = 2*v(:,2) - v(:,3); % �Ȱ��� ������� ù column ���� ��
    v(:,end) = 2*v(:,end-1) - v(:, end-2); % �Ȱ��� ������ column ���� ��
    
    % linear boundary condition�� ������ grid�� ä�� ����
    for j = 2:Ny+1
        for i = 2:Nx+1
            fx(i-1) = 1/2*rho*sig1*sig2*x(i)*y(j) * (v(i+1,j+1) - v(i+1,j-1) - v(i-1,j+1) + v(i-1,j-1))/(4*h*k) + v(i,j)/dt;
        end
        v0(2:end-1,j) = tridiag(diag_x1, diag_x2, diag_x3, fx);
    end
    v = v0; % mesh update
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% y - direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear boundary condition
    v(1,2:end-1) = 2*v(2,2:end-1) - v(3,2:end-1); % v(t+1) - v(t) = v(t) - v(t-1) �� �̿��ؼ� ù row���� ��
    v(end,2:end-1) = 2*v(end-1,2:end-1) - v(Nx,2:end-1); % ���� �Ȱ��� ������ row���� ��
    v(:,1) = 2*v(:,2) - v(:,3); % �Ȱ��� ������� ù column ���� ��
    v(:,end) = 2*v(:,end-1) - v(:, end-2); % �Ȱ��� ������ column ���� ��
    
    for i = 2:Nx+1
        for j = 2:Ny+1
            fy(j-1) = 1/2*rho*sig1*sig2*x(i)*y(j) * (v(i+1,j+1) - v(i+1,j-1) - v(i-1,j+1) + v(i-1,j-1))/(4*h*k) + v(i,j)/dt;
        end
        v0(i,2:end-1) = tridiag(diag_y1, diag_y2, diag_y3, fy);
    end
    
    % �����ȯ �踮�� ������Ʈ
    x_line = find(x>=KI, 1); % knock-in ���� (x ����)
    y_line = find(y>=KI, 1); % knock-in ���� (y ����)
    % knock-in ������ mesh���� update
    %(u�� knock-in�� ġ�� �ʾҴٰ� �����߱� ������ knock-in�� �ƴٰ� ������ v�� mesh�� �ٲ���)
    u0(:,2:y_line) = v0(:,2:y_line);
    u0(2:x_line,y_line+1:end-1) = v0(2:x_line, y_line+1:end-1);
end

% discount
u0 = exp(-r*T)*u0;
v0 = exp(-r*T)*v0;

%%% knock-in�� ���� ���� �� �������� weighted average sum
mesh = (1-weight)*u0 + weight*v0;
end

function simul = simulation(T,c,K,KI,d,sig1,sig2,rho,r,M)
N = 6; % �����ȯ Ȯ�� �Ⱓ (6����)
dt = 0.5/N;
S1 = zeros(6*N,1); % HSCEI price : 9482.01 at 2016/11/03 (currency : HKD)
S2 = zeros(6*N,1); % Eurostoxx50 price : 2973.49 at 2016/11/03 (currency : EUR)
S1(1) = 100;  % ù�� ���� 100���� ǥ��ȭ
S2(1) = 100; 
f = zeros(M,1);
sum = 0;
num = 0;
for i = 1:M
    min1 = S1(1);
    min2 = S2(1);
    for j = 1:4*N
        x1 = normrnd(0,1);
        x2 = normrnd(0,1);
        e1 = x2;
        e2 = rho*x1 + x2*sqrt(1-rho^2);
        S1(j+1) = S1(j) * exp( (r-sig1^2/2)*dt + sig1*sqrt(dt)*e1 );
        S2(j+1) = S2(j) * exp( (r-sig2^2/2)*dt + sig2*sqrt(dt)*e2 );
        min1 = min(S1(j+1), min1);
        min2 = min(S2(j+1), min2);
    end
    
    if S1(N) >= K(end) && S2(N) >= K(end)
        f(i) = 100*(1+c(end))*exp(-r*0.5);
    elseif S1(2*N) >= K(end-1) && S2(2*N) >= K(end-1)
        f(i) = 100*(1+c(end-1))*exp(-r*1);    
    elseif S1(3*N) >= K(end-2) && S2(3*N) >= K(end-2)
        f(i) = 100*(1+c(end-2))*exp(-r*1.5);
    elseif S1(4*N) >= K(end-3) && S2(4*N) >= K(end-3)
        f(i) = 100*(1+c(end-3))*exp(-r*2);
    elseif S1(5*N) >= K(end-4) && S2(5*N) >= K(end-4)
        f(i) = 100*(1+c(end-4))*exp(-r*2.5);
    elseif S1(6*N) >= K(end-5) && S2(6*N) >= K(end-5) 
        f(i) = 100*(1+c(end-5))*exp(-r*3);
    elseif min1> KI && min2 > KI
        f(i) = 100*(1+d)*exp(-r*3);
    else
        f(i) = min(S1(6*N), S2(6*N)) * exp(-r*3);
        num = num+1;
    end
end
simul = mean(f);
end

function y = tridiag( a, b, c, f )

%  Solve the  n x n  tridiagonal system for y:
%
%  [ a(1)  c(1)                                  ] [  y(1)  ]   [  f(1)  ]
%  [ b(2)  a(2)  c(2)                            ] [  y(2)  ]   [  f(2)  ]
%  [       b(3)  a(3)  c(3)                      ] [        ]   [        ]
%  [            ...   ...   ...                  ] [  ...   ] = [  ...   ]
%  [                    ...    ...    ...        ] [        ]   [        ]
%  [                        b(n-1) a(n-1) c(n-1) ] [ y(n-1) ]   [ f(n-1) ]
%  [                                 b(n)  a(n)  ] [  y(n)  ]   [  f(n)  ]
%
%  f must be a vector (row or column) of length n
%  a, b, c must be vectors of length n (note that b(1) and c(n) are not used)

% some additional information is at the end of the file

n = length(f);
v = zeros(n,1);   
y = v;
w = a(1);
y(1) = f(1)/w;
for i=2:n
    v(i-1) = c(i-1)/w;
    w = a(i) - b(i)*v(i-1);
    y(i) = ( f(i) - b(i)*y(i-1) )/w;
end
for j=n-1:-1:1
   y(j) = y(j) - v(j)*y(j+1);
end


%  This is an implementation of the Thomas algorithm.  It does not overwrite a, b, c, f but 
%  it does introduce a working n-vector (v).

%%%%%  Example
% n = 5; a = 4*ones(n,1); b = ones(n,1); c = 3*ones(n,1);
% f = rand(n,1);
% y = tridiag(a,b,c,f);
%%%%%  check solution
% A = diag(a,0) + diag(ones(n-1,1),-1) + diag(3*ones(n-1,1),1)
% A*y - f

%%%%% Conditions that will guarantee the matrix equation can be solved using this algorithm: 
%%%%%  1. matrix strictly diagonally dominant
%%%%%  2. matrix diagonally dominant, c_i not zero for all i, and abs(b_n) < abs(a_n)

%  It has been tested on MATLAB, version R2010b and version R2012a
%  version: 1.0
%  March 9, 2013
end

end