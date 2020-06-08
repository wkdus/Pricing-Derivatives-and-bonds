# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:28:50 2016

@author: Nam
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
from datetime import datetime as dt


def main(): 
#==============================================================================
# bond1 today[채권A] X
        clean = False
        p1 = 103.75
        y1 = 0.01411
        c1 = 0.0225
        m1 = 2
        n1 = 11
        LCD1 = 'Sep/30/2015'
        NCD1= 'Mar/31/2016'
        settle1 = 'Mar/08/2016'
##==============================================================================

##==============================================================================
## Bond2 today[채권B] Y
        clean = False
        p2 = 101.5
        y2 = 0.01369
        c2 = 0.0175
        m2 = 2
        n2 = 10
        LCD2 = 'Dec/31/2015'
        NCD2 = 'Jun/30/2016'
        settle2 = 'Mar/08/2016'
##==============================================================================

##==============================================================================
## Bond3 today[채권C] Z
        clean = False
        p3 = 400.4375
        y3 =  0.0135
        c3 = 0.01375
        m3 = 2
        n3 = 10
        LCD3 = 'Sep/30/2015'
        NCD3 = 'Mar/30/2016'
        settle3 = 'Mar/08/2016'
#==============================================================================

#==============================================================================
# bond1 future day[채권A] D
        clean = False
        p4 = 104.3125
        y4 = 0.1411
        c4 = 0.0225
        m4 = 2
        n4 = 11
        LCD4 = 'Dec/31/2015'
        NCD4 = 'Jun/30/2016'
        settle4 = 'Mar/08/2016'
#==============================================================================
#===================
#==============================================================================
# bond2 future day[채권B]
        clean = False
        p5 = 101.719
        y5 = 0.07209
        c5 = 0.01875
        m5 = 2
        n5 = 4
        LCD5 = 'Aug/31/2015'
        NCD5 = 'Feb/29/2016'
        settle5 = 'Mar/07/2016'

#==============================================================================

#==============================================================================
# Bond3 future day[채권C]
        clean = False
        p6 = 100.672
        y6 =  0.012338
        c6 = 0.01375
        m6 = 2
        n6 = 10
        LCD6 = 'Jan/31/2015'
        NCD6 = 'Jul/31/2016'
        settle6 = 'Mar/07/2016'
#==============================================================================

        print('\ncompute a yield - bond1 today[채권A]') 
        AI1 = get_AI(LCD1, NCD1, settle1, c1, n1)
        y1 = get_yield(p1 + AI1, LCD1, NCD1, settle1, c1, m1, n1)
        full_price1 = p1 + AI1
        print('AI: ' , AI1)
        print('yield (computed): {}'.format(y1*100))
        print('yield (database): n/a')
        print('clean price (database): n/a')
        print('full price (computed): {}'.format(full_price1))

        print('\ncompute a yield - bond2 today[채권B]') 
        AI2 = get_AI(LCD2, NCD2, settle2, c2, n2)
        y2 = get_yield(p2 + AI2, LCD2, NCD2, settle1, c2, m2, n2)
        full_price2 = p2 + AI2
        print('AI: ' , AI2)
        print('yield (computed): {}'.format(y2*100))
        print('yield (database): n/a')
        print('clean price (database): n/a')
        print('full price (computed): {}'.format(full_price2))

        print('\ncompute a yield - bond2 today[채권C]') 
        AI3 = get_AI(LCD3, NCD3, settle3, c3, n3)
        y3 = get_yield(p3 + AI3, LCD3, NCD3, settle3, c3, m3, n3)
        full_price3 = p3 + AI3
        print('AI: ' , AI3)
        print('yield (computed): {}'.format(y3*100))
        print('yield (database): n/a')
        print('clean price (database): n/a')
        print('full price (computed): {}'.format(full_price3))

        print('\ncompute a yield - bond1 future day[채권A]') 
        AI4 = get_AI(LCD4, NCD4, settle4, c4, n4)
        y6 = get_yield(p4 + AI4, LCD4, NCD4, settle4, c4, m4, n4)
        full_price4 = p4 + AI4
        print('AI: ' , AI4)
        print('yield (computed): {}'.format(y4*100))
        print('yield (database): n/a')
        print('clean price (database): n/a')
        print('full price (computed): {}'.format(full_price4))
        
        print('\ncompute a yield - bond2 future day[채권B]') 
        AI5 = get_AI(LCD5, NCD5, settle5, c5, n5)
        y5 = get_yield(p5 + AI5, LCD5, NCD5, settle5, c5, m5, n5)
        full_price5 = p5 + AI5
        print('AI: ' , AI5)
        print('yield (computed): {}'.format(y5*100))
        print('yield (database): n/a')
        print('clean price (database): n/a')
        print('full price (computed): {}'.format(full_price5))
        
        print('\ncompute a yield - bond3 future day[채권C]') 
        AI6 = get_AI(LCD6, NCD6, settle6, c6, n6)
        y6 = get_yield(p6 + AI6, LCD6, NCD6, settle6, c6, m6, n6)
        full_price6 = p6 + AI6
        print('AI: ' , AI6)
        print('yield (computed): {}'.format(y6*100))
        print('yield (database): n/a')
        print('clean price (database): n/a')
        print('full price (computed): {}'.format(full_price6))

def get_yield(p, LCD, NCD, settle, c, m, n):
    # computes the yield, given price and so on
    # p: price of a bond (dirty)
    # LCD: last coupon date
    # NCD: next coupon date
    # settle: settlement date
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining    
    equation = lambda y: get_bond_price(y, LCD, NCD, settle, c, m, n) - p # y를 제외한 나머지 모수들을 고정시키고 y의 optimal value를 구함
    return optim.root(equation, 0.1).x[0]

def get_bond_price(y, LCD, NCD, settle, c, m, n):
    # for non-coupon date
    # computes the bond price (dirty)
    # y: yield
    # LCD: last coupon date
    # NCD: next coupon date
    # settle: settlement date
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining 
    
    v = (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(settle,'%b/%d/%Y'))\
        / (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(LCD,'%b/%d/%Y'))
    p = 0
    for i in range(1,n+1):
        disc_coup = (400*c/m)/((1+y/m)**(i-1))
        p += disc_coup
    p += 400/((1+y/m)**(n-1))
    return p/((1+y/m)**v)


def get_AI(LCD, NCD, settle, c, m):
    # computes the accrued interest
    # LCD: last coupon date
    # NCD: next coupon date
    # settle: settlement date
    # c: coupon rate (annualized)
    # m: number of compounding a year

    v = (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(settle,'%b/%d/%Y')) \
        / (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(LCD,'%b/%d/%Y'))
    return (1-v)*(400*c/m)  # accrued interest


main()
plt.show()


## time to maturity calculation in years
#a = (dt.strptime('Jan/31/2021','%b/%d/%Y') - dt.strptime('Feb/05/2016','%b/%d/%Y')) # %b used
#a = (dt.strptime('11/15/2045','%m/%d/%Y') - dt.strptime('02/05/2016','%m/%d/%Y')) # %m used
#a
#b = 1822/365
#b

##
####==============================================================================
#### Bond2 today[채권B]
#        clean = False
#        p = 101.734
#        y = 0.07229
#        c = 0.01875
#        m = 2
#        n = 4
#        LCD = 'Aug/31/2015'
#        NCD = 'Feb/29/2016'
#        settle = 'Feb/24/2016'
###==============================================================================

###==============================================================================
### Bond3 today[채권C]
#        clean = False
#        p = 100.641
#        y =  0.012408
#        c = 0.01375
#        m = 2
#        n = 10
#        LCD = 'Jan/31/2015'
#        NCD = 'Jul/31/2016'
#        settle = 'Feb/24/2016'
##==============================================================================
#
#
##==============================================================================
## bond1 future day[채권A]
#        clean = False
#        p = 101.555
#        y = 0.010759
#        c = 0.015
#        m = 2
#        n = 8
#        LCD = 'Nov/30/2015'
#        NCD = 'May/31/2016'
#        settle = 'Feb/24/2016'
##==============================================================================
##
##==============================================================================
## bond2 future day[채권B]
#        clean = False
#        p = 101.719
#        y = 0.07209
#        c = 0.01875
#        m = 2
#        n = 4
#        LCD = 'Aug/31/2015'
#        NCD = 'Feb/29/2016'
#        settle = 'Feb/24/2016'

##==============================================================================

##==============================================================================
## Bond3 future day[채권C]
#        clean = False
#        p = 100.672
#        y =  0.012338
#        c = 0.01375
#        m = 2
#        n = 10
#        LCD = 'Jan/31/2015'
#        NCD = 'Jul/31/2016'
#        settle = 'Feb/24/2016'
###==============================================================================
