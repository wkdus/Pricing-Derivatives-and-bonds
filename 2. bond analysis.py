# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:44:51 2016
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
from datetime import datetime as dt





def main():
    # Duration and convexity

    command = 3
    # 1: duration exercise
    # 2: immunization exercise
    # 3: convexity exercise

    # duration exercise
    if command==1:

##==============================================================================
## bond1 today[채권A]
#        clean = False
#        p = 101.85923009747516
#        y = 0.011613030843071948
#        c = 0.015
#        m = 2
#        n = 8
#        LCD = 'Nov/30/2015'
#        NCD= 'May/31/2016'
#        settle = 'Feb/24/2016'
###==============================================================================
#
###==============================================================================
### Bond2 today[채권B]
#        clean = False
#        p = 102.18987225274725
#        y = 0.010219016899818951
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
#        p = 100.73878336380257
#        y = 0.013192551139330468
#        c = 0.01375
#        m = 2
#        n = 10
#        LCD = 'Jan/31/2015'
#        NCD = 'Jul/31/2016'
#        settle = 'Feb/24/2016'
##==============================================================================

#==============================================================================
## bond1 future day[채권A]
#        clean4 = False
#        p4 = 101.64618852459017
#        y4 = 0.010758999999999999
#        c4 = 0.015
#        m4 = 2
#        n4 = 8
#        LCD4 = 'Nov/30/2015'
#        NCD4 = 'May/31/2016'
#        settle4 = 'Feb/27/2016'
##==============================================================================
#
##==============================================================================
## bond2 future day[채권B]
#        clean5 = False
#        p5 = 102.1825989010989
#        y5 = 0.01032454837883132
#        c5 = 0.01875
#        m5 = 2
#        n5 = 4
#        LCD5 = 'Aug/31/2015'
#        NCD5 = 'Feb/29/2016'
#        settle5 = 'Feb/27/2016'
#
##==============================================================================
#
##==============================================================================
## Bond3 future day[채권C]
#        clean = False
#        p6 = 100.77053747714808
#        y6 = 0.013129995793435005
#        c6 = 0.01375
#        m6 = 2
#        n6 = 10
#        LCD6 = 'Jan/31/2015'
#        NCD6 = 'Jul/31/2016'
#        settle6 = 'Feb/27/2016'
##==============================================================================
#    
        print('======================')
        print('     duration')
        print('======================')
        
#        print()
#        print('forward difference')
#        h = 0.0001
#        p0 = get_price(y,c,m,n)
#        p1 = get_price(0.06+h,0.08,2,2)
#        print('6.00%:', p0)
#        print('6.01%:', p1)
#        print('DV01 : {:.20f}'.format(-(p1-p0)/h/100,20))
#        h = 0.01
#        p0 = get_price(y,c,m,n)
#        p1 = get_price(y+h,c,m,n)
#        print('6.00%: {:.10f}'.format(p0))
#        print('6%+h: {:.10f}'.format(p1))
#        print('h =', h)
#        print('DV01 : {:.20f}'.format(-(p1-p0)/h/100))
#
#        print()
#        print('central difference')
#        h = 0.0001
#        p0 = get_price(y-h/2,c,m,n)
#        p1 = get_price(y+h/2,c,m,n)
#        print('6%-h/2:', p0)
#        print('6%+h/2:', p1)
#        print('h =',  h)
#        print('DV01 : {:.20f}'.format(-(p1-p0)/h/100,20))
#
#        print()
#        print('analytic derivative')
#        print('DV01 : {:.20f}'.format(get_DV01(y,c,m,n),20))
#
#        print()
#        print('example of durations')
#        print('Macaulay duration:', get_Mac_dur(y, clean, LCD, NCD, settle, c, m, n), 'years')
#        print('Modified duration:', get_mod_dur(y, clean, LCD, NCD, settle, c, m, n), '% change in price per 1% change in yield')
#        print('DV01             :', get_mod_dur(y, clean, LCD, NCD, settle, c, m, n)*get_price(y,c,m,n)/100, '$ change per 1% point change in yield')

#        print()
#        print('duration of a portfolio consisting of: 1 of bond1 and 2 of bond2')
#        P1 = get_price(0.03933,0.04,2,4)
#        ModD1 = get_mod_dur(0.03933,0.04,2,4)
#        DV011 = get_DV01(0.03933,0.04,2,4)
#        P2 = get_price(0.04056,0.04125,2,10)
#        ModD2 = get_mod_dur(0.04056,0.04125,2,10)
#        DV012 = get_DV01(0.04056,0.04125,2,10)
#        # portfolio consisting of par value 100 of bond1 and par value 200 of bond2
#        P = 1*P1 + 2*P2
#        DV01 = 1*DV011 + 2*DV012
#        ModD = P1/P*ModD1 + P2/P*ModD2
#        print('asset         P          DV01         ModD')
#        print('bond1   {:9.6f}   {:9.6f}     {:9.6f}'.format(P1, DV011, ModD1))
#        print('bond2   {:9.6f}   {:9.6f}     {:9.6f}'.format(P2, DV012, ModD2))
#        print('portf   {:9.6f}   {:9.6f}     {:9.6f}'.format(P,  DV01,  ModD))

        print()
        print('duration on a non-coupon date')
        non_m = 2
        non_y = get_yield(p, False, LCD, NCD, settle, c, m, n)
        non_h = 0.005
        non_P0 = get_bond_price(non_y  , False, LCD, NCD, settle, c, m, n)  # full price at y
        non_Ph = get_bond_price(non_y+non_h, False, LCD, NCD, settle, c, m, n)  # full price at y+h
        non_DV01 = (non_P0-non_Ph)/non_h/100
        non_ModD = non_DV01*100/non_P0
        non_MacD = non_ModD*(1+non_y/non_m)
        print('Macaulay duration {:.4f} years'.format(non_MacD))
        print('Modified duration: {:.4f} % change per 1%p yield change'.format(non_ModD))
        print('DV01             : {:.4f} $ change per 1%p yield change'.format(non_DV01))


    # immunization exercise
    if command==2:   

#
#==============================================================================
# bond1 today[채권A]
        clean = False
        p1 = 101.85923009747516
        y1 = 0.011613030843071948
        c1 = 0.015
        m1 = 2
        n1 = 8
        LCD1 = 'Nov/30/2015'
        NCD1 = 'May/31/2016'
        settle1 = 'Feb/24/2016'
##==============================================================================

##==============================================================================
## Bond2 today[채권B]
        clean2 = False
        p2 = 102.18987225274725
        y2 = 0.010219016899818951
        c2 = 0.01875
        m2 = 2
        n2 = 4
        LCD2 = 'Aug/31/2015'
        NCD2 = 'Feb/29/2016'
        settle2 = 'Feb/24/2016'
##==============================================================================

##==============================================================================
## Bond3 today[채권C]
        clean = False
        p3 = 100.73878336380257
        y3 = 0.013192551139330468
        c3 = 0.01375
        m3 = 2
        n3 = 10
        LCD3 = 'Jan/31/2015'
        NCD3 = 'Jul/31/2016'
        settle3 = 'Feb/24/2016'
#==============================================================================
###==============================================================================
# bond1 future day[채권A]
        clean = False
        p1_f = 101.64618852459017
        y1_f = 0.010758999999999999
        c1_f = 0.015
        m1_f = 2
        n1_f = 8
        LCD1_f = 'Nov/30/2015'
        NCD1_f = 'May/31/2016'
        settle1_f = 'Feb/27/2016'
#==============================================================================

#==============================================================================
# bond2 future day[채권B]
        clean = False
        p2_f = 102.1825989010989
        y2_f = 0.01032454837883132
        c2_f = 0.01875
        m2_f = 2
        n2_f = 4
        LCD2_f = 'Aug/31/2015'
        NCD2_f = 'Feb/29/2016'
        settle2_f = 'Feb/27/2016'

#==============================================================================

#==============================================================================
# Bond3 future day[채권C]
        clean = False
        p3_f = 100.77053747714808
        y3_f = 0.013129995793435005
        c3_f = 0.01375
        m3_f = 2
        n3_f = 10
        LCD3_f = 'Jan/31/2015'
        NCD3_f = 'Jul/31/2016'
        settle3_f = 'Feb/27/2016'
#==============================================================================
    
        # construct an immunization portfolio against a 4 year debt, using 2 and 5 year bonds
        print('======================')
        print('    Immunization')
        print('======================')
        par_liab_num = 4  # $1,000 due 4 years later

        # compute portfolio weights
        non_y1 = get_yield(p1, clean, LCD1, NCD1, settle1, c1, m1, n1) # input dirty price p1 --> dirty yield  
        non_y2 = get_yield(p2, clean, LCD2, NCD2, settle2, c2, m2, n2) # input dirty price p2 --> dirty yield  
        non_y3 = get_yield(p3, clean, LCD3, NCD3, settle3, c3, m3, n3) # input dirty price p3 --> dirty yield  
        (P4, ModD4, DV014, cnv4) = get_all(p2,non_y2, clean, LCD2, NCD2, settle2, c2, m2, n2)  # liability / dirty price
        (P2, ModD2, DV012, cnv2) = get_all(p1,non_y1, clean, LCD1, NCD1, settle1, c1, m1, n1)  # asset / dirty price
        (P5, ModD5, DV015, cnv5) = get_all(p3,non_y3, clean, LCD3, NCD3, settle3, c3, m3, n3)  # asset / dirty price
        w2 = (ModD4-ModD5) / (ModD2-ModD5)  # weight of 2 year bond
        w5 = 1-w2                           # weight of 5 year bond
        x2 = (P4*par_liab_num*w2)/P2  # number of 2 year bonds to buy
        x5 = (P4*par_liab_num*w5)/P5  # number of 5 year bonds to buy

        print( 'securities             P         ModD          DV01       convexity')
        print('bond1(liabil~)   {:9.6f}   {:9.6f}     {:9.6f}      {:9.6f}'.format(P4, ModD4, DV014, cnv4)) # acquired from get_all()
        print('bond2(asset)     {:9.6f}   {:9.6f}     {:9.6f}      {:9.6f}'.format(P2, ModD2, DV012, cnv2)) # acquired from get_all()
        print('bond3(asset)     {:9.6f}   {:9.6f}     {:9.6f}      {:9.6f}'.format(P5, ModD5, DV015, cnv5)) # acquired from get_all()
        print()
        print('numbers of bonds to buy (bond2 and bond3):     {:9.6f}   {:9.6f}'.format(x2, x5))
        print()
        print( 'securities            P         ModD          DV01')
        print('liabil~          {:9.6f}   {:9.6f}     {:9.6f}'.format(P4*par_liab_num, ModD4, DV014*par_liab_num))
        print('asset            {:9.6f}   {:9.6f}     {:9.6f}'.format(P2*x2+P5*x5, ModD2*w2+ModD5*w5, DV012*x2+DV015*x5))

        print()
        print('effect of duration matching')
        h = 0.005  # change of yield
       
        P4_ydec = get_bond_price(non_y2-h,clean, LCD2, NCD2, settle2, c2, m2, n2)*par_liab_num
        P4_ynoc = get_bond_price(non_y2  ,clean, LCD2, NCD2, settle2, c2, m2, n2)*par_liab_num
        P4_yinc = get_bond_price(non_y2+h,clean, LCD2, NCD2, settle2, c2, m2, n2)*par_liab_num
        P_ydec = x2*get_bond_price(non_y1-h,clean, LCD1, NCD1, settle1, c1, m1, n1) + x5*get_bond_price(non_y3-h,clean, LCD3, NCD3, settle3, c3, m3, n3)
        P_ynoc = x2*get_bond_price(non_y1  ,clean, LCD1, NCD1, settle1, c1, m1, n1) + x5*get_bond_price(non_y3  ,clean, LCD3, NCD3, settle3, c3, m3, n3)
        P_yinc = x2*get_bond_price(non_y1+h,clean, LCD1, NCD1, settle1, c1, m1, n1) + x5*get_bond_price(non_y3+h,clean, LCD3, NCD3, settle3, c3, m3, n3)
        print('yield change     liability      asset       asset-liab')
        print('yield -{}%:    {:9.6f}   {:9.6f}     {:9.6f}'.format(h*100, P4_ydec, P_ydec, P_ydec-P4_ydec))
        print('yield +0.0%:    {:9.6f}   {:9.6f}     {:9.6f}'.format(      P4_ynoc, P_ynoc, P_ynoc-P4_ynoc))
        print('yield +{}%:    {:9.6f}   {:9.6f}     {:9.6f}'.format(h*100, P4_yinc, P_yinc, P_yinc-P4_yinc))

        print()
        print('adjust portfolio when yield changes: yield = +0.5%')
        (P4_new, ModD4_new) = get_all(p1,non_y1+h,clean, LCD1, NCD1, settle1, c1, m1, n1)[:2]
        (P2_new, ModD2_new) = get_all(p2,non_y2+h,clean, LCD2, NCD2, settle2, c2, m2, n2)[0:2]
        (P5_new, ModD5_new) = get_all(p3,non_y3+h,clean, LCD3, NCD3, settle3, c3, m3, n3)[0:2]
        w2_new = (ModD4_new-ModD5_new) / (ModD2_new-ModD5_new)
        w5_new = 1-w2_new
        x2_new = P4_new*w2_new/P2_new*par_liab_num  # number of 2 year bonds to buy
        x5_new = P4_new*w5_new/P5_new*par_liab_num  # number of 5 year bonds to buy
        print('numbers of bonds to buy (bond2 and bond5):  {:9.6f}   {:9.6f}'.format(x2_new, x5_new))


        print('======================================================================')
        print('   evaluate the actual effectiveness of the immunization strategy')
        print('======================================================================')

        non_y1_f = get_yield(p1_f, clean, LCD1_f, NCD1_f, settle1_f, c1_f, m1_f, n1_f) # input dirty price p1 --> dirty yield  
        non_y2_f = get_yield(p2_f, clean, LCD2_f, NCD2_f, settle2_f, c2_f, m2_f, n2_f) # input dirty price p2 --> dirty yield  
        non_y3_f = get_yield(p3_f, clean, LCD3_f, NCD3_f, settle3_f, c3_f, m3_f, n3_f) # input dirty price p3 --> dirty yield  


        P4_ydec_f = get_bond_price(non_y2_f-h,clean, LCD2_f, NCD2_f, settle2_f, c2_f, m2_f, n2_f)*par_liab_num # future liability
        P4_ynoc_f = get_bond_price(non_y2_f  ,clean, LCD2_f, NCD2_f, settle2_f, c2_f, m2_f, n2_f)*par_liab_num
        P4_yinc_f = get_bond_price(non_y2_f+h,clean, LCD2_f, NCD2_f, settle2_f, c2_f, m2_f, n2_f)*par_liab_num
        P_ydec_f = x2*get_bond_price(non_y1_f-h,clean, LCD1_f, NCD1_f, settle1_f, c1_f, m1_f, n1_f) + x5*get_bond_price(non_y3_f-h,clean, LCD3_f, NCD3_f, settle3_f, c3_f, m3_f, n3_f) # future asset
        P_ynoc_f = x2*get_bond_price(non_y1_f  ,clean, LCD1_f, NCD1_f, settle1_f, c1_f, m1_f, n1_f) + x5*get_bond_price(non_y3_f  ,clean, LCD3_f, NCD3_f, settle3_f, c3_f, m3_f, n3_f)
        P_yinc_f = x2*get_bond_price(non_y1_f+h,clean, LCD1_f, NCD1_f, settle1_f, c1_f, m1_f, n1_f) + x5*get_bond_price(non_y3_f+h,clean, LCD3_f, NCD3_f, settle3_f, c3_f, m3_f, n3_f)
       
        print('yield change     liability      asset       asset-liab')
        print('yield -{}%:    {:9.6f}   {:9.6f}     {:9.6f}'.format(h*100, P4_ydec_f, P_ydec_f, P_ydec_f-P4_ydec_f))
        print('yield +0.0%:    {:9.6f}   {:9.6f}     {:9.6f}'.format(      P4_ynoc_f, P_ynoc_f, P_ynoc_f-P4_ynoc_f))
        print('yield +{}%:    {:9.6f}   {:9.6f}     {:9.6f}'.format(h*100, P4_yinc_f, P_yinc_f, P_yinc_f-P4_yinc_f))



        # figure
        dy_N = 101
        dy_range = np.linspace(-0.09,0.2,dy_N)
        p_liab = np.empty_like(dy_range)
        p_asset = np.empty_like(dy_range)
        for i in range(dy_N):
            dy = dy_range[i]
            p_liab[i] = get_price(y1+dy,c1,m1,n1)*par_liab_num
            p_asset[i] = get_price(y2+dy,c2,m2,n2)*x2 + get_price(y3+dy,c3,m3,n3)*x5

        plt.plot(dy_range,p_liab, color='green')
        plt.plot(dy_range,p_asset, LineStyle='--', Color='blue', LineWidth=2)
        plt.scatter(0,P4*par_liab_num, color='red', s=100)
        plt.legend(['liability','asset','current'], scatterpoints=1)
        plt.xlabel('yield change (0 represents current yield)')
        plt.ylabel('present value')


    # convexity exercise
    if command==3:
        print('======================')
        print('      convexity')
        print('======================')

#==============================================================================
# bond1 today[채권A]
        clean = False
        p1 = 101.85923009747516
        y1 = 0.011613030843071948
        c1 = 0.015
        m1 = 2
        n1 = 8
        LCD1 = 'Nov/30/2015'
        NCD1 = 'May/31/2016'
        settle1 = 'Feb/24/2016'
##==============================================================================

##==============================================================================
## Bond2 today[채권B]
        clean2 = False
        p2 = 102.18987225274725
        y2 = 0.010219016899818951
        c2 = 0.01875
        m2 = 2
        n2 = 4
        LCD2 = 'Aug/31/2015'
        NCD2 = 'Feb/29/2016'
        settle2 = 'Feb/24/2016'
##==============================================================================

##==============================================================================
## Bond3 today[채권C]
        clean = False
        p3 = 100.73878336380257
        y3 = 0.013192551139330468
        c3 = 0.01375
        m3 = 2
        n3 = 10
        LCD3 = 'Jan/31/2015'
        NCD3 = 'Jul/31/2016'
        settle3 = 'Feb/24/2016'
#===============================================================================
 
        # asset
        par_liab_num = 4  # $1,000 due 4 years later
        h=0.001
        
        non_y1 = get_yield(p1, clean, LCD1, NCD1, settle1, c1, m1, n1) # input dirty price p1 --> dirty yield  
        non_y2 = get_yield(p2, clean, LCD2, NCD2, settle2, c2, m2, n2) # input dirty price p2 --> dirty yield  
        non_y3 = get_yield(p3, clean, LCD3, NCD3, settle3, c3, m3, n3) # input dirty price p3 --> dirty yield  
        (P4, ModD4, DV014, cnv4) = get_all(p2,non_y2, clean, LCD2, NCD2, settle2, c2, m2, n2)  # liability / dirty price
        (P2, ModD2, DV012, cnv2) = get_all(p1,non_y1, clean, LCD1, NCD1, settle1, c1, m1, n1)  # asset / dirty price
        (P5, ModD5, DV015, cnv5) = get_all(p3,non_y3, clean, LCD3, NCD3, settle3, c3, m3, n3)  # asset / dirty price
        w2 = (ModD4-ModD5) / (ModD2-ModD5)  # weight of 2 year bond
        w5 = 1-w2                           # weight of 5 year bond
        x2 = (P4*par_liab_num*w2)/P2  # number of 2 year bonds to buy
        x5 = (P4*par_liab_num*w5)/P5  # number of 5 year bonds to buy
        
        P_ydec = x2*get_bond_price(non_y1-h,clean, LCD1, NCD1, settle1, c1, m1, n1) + x5*get_bond_price(non_y3-h,clean, LCD3, NCD3, settle3, c3, m3, n3)
        P_ynoc = x2*get_bond_price(non_y1  ,clean, LCD1, NCD1, settle1, c1, m1, n1) + x5*get_bond_price(non_y3  ,clean, LCD3, NCD3, settle3, c3, m3, n3)
        P_yinc = x2*get_bond_price(non_y1+h,clean, LCD1, NCD1, settle1, c1, m1, n1) + x5*get_bond_price(non_y3+h,clean, LCD3, NCD3, settle3, c3, m3, n3)
        cnv_assets = ((P_yinc+P_ydec-2*P_ynoc)/h**2)/P_ynoc       
                
        print('immunization portfolio on today - convexity: {:.20f}'.format(cnv_assets))
        
        # liability
        h=0.001
        P4_ydec = get_bond_price(non_y2-h,clean, LCD2, NCD2, settle2, c2, m2, n2)*par_liab_num
        P4_ynoc = get_bond_price(non_y2  ,clean, LCD2, NCD2, settle2, c2, m2, n2)*par_liab_num
        P4_yinc = get_bond_price(non_y2+h,clean, LCD2, NCD2, settle2, c2, m2, n2)*par_liab_num
        cnv_liability = ((P4_ydec+P4_yinc-2*P4_ynoc)/h**2)/P4_ynoc
        print('issue A on today - convexity: {:.20f}'.format(cnv_liability))


def get_all(p,y, clean, LCD, NCD, settle, c, m, n):
    non_y = get_yield(p, False, LCD, NCD, settle, c, m, n) # false로 바꿨음!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    non_h = 0.005
    non_P0 = get_bond_price(non_y  , False, LCD, NCD, settle, c, m, n)  # full price at y
    non_Ph = get_bond_price(non_y+non_h, False, LCD, NCD, settle, c, m, n)  # full price at y+h
    non_DV01 = (non_P0-non_Ph)/non_h/100
    non_ModD = non_DV01*100/non_P0
    non_MacD = non_ModD*(1+non_y/m)

#    ModD = get_mod_dur(non_y, clean, LCD, NCD, settle, c, m, n)
#    DV01 = get_DV01(non_y, c, m, n)
    cnv = get_convexity(non_y, clean, LCD, NCD, settle, c, m, n)
    return p, non_ModD, non_DV01, cnv


def get_price(y, c, m, n):
    # computes the bond price
    # p: price of a bond
    # y: yield
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining

    p = 0
    for i in range(1,n+1):  # 1,2,...,n
        disc_coup = (100*c/m)/((1+y/m)**i)
        p += disc_coup
    p += 100/((1+y/m)**n)
    return p


def get_price_diff(y, c, m, n):
    # computes the derivative of the bond price
    # dp: derivative
    # y: yield
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining

    dp = 0
    for i in range(1,n+1):
        d_disc_coup = -(i/m)*(100*c/m)/((1+y/m)**(i+1))
        dp += d_disc_coup
    dp -= (n/m)*100/((1+y/m)**(n+1))
    return dp


def get_Mac_dur(y, clean, LCD, NCD, settle, c, m, n):
    # computes the Macaulay duration
    # d: duration
    # y: yield
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining

    v = (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(settle,'%b/%d/%Y')) \
        / (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(LCD,'%b/%d/%Y'))
    d = 0
    for i in range(1,n+1):
        disc_coup = (100*c/m)/((1+y/m)**(i+v-2))
        d += ((i+v-1)/m)*disc_coup
    d += ((n+v-1)/m)*100/((1+y/m)**(n+v-2))
    d /= get_bond_price(y, clean, LCD, NCD, settle, c, m, n)  # duplicate computations but let's not worry this time
    return d


def get_mod_dur(y, clean, LCD, NCD, settle, c, m, n):
    # computes the modified duration
    return get_Mac_dur(y, clean, LCD, NCD, settle, c, m, n)/(1+y/m)


def get_DV01(y, c, m, n):
    # computes the analytical DV01
    return -get_price_diff(y, c, m, n)/100


def get_convexity(y, clean, LCD, NCD, settle, c, m, n):
    # computes the convexity
    # d: duration
    # y: yield
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining

    v = (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(settle,'%b/%d/%Y')) \
        / (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(LCD,'%b/%d/%Y'))
    cnv = 0
    for i in range(1,n+1):
        disc_coup = (100*c/m)/((1+y/m)**(i+v-3))
        cnv += ((i+v-1)*(i+v-2)/m**2)*disc_coup
    cnv += ((n+v-1)*(n+v-2)/m**2)*100/((1+y/m)**(n+v-3))
    cnv /= get_bond_price(y, clean, LCD, NCD, settle, c, m, n)  # duplicate computations but let's not worry this time
    return cnv


def get_yield(p, clean, LCD, NCD, settle, c, m, n):
    # computes the yield
    # p: price of a bond (dirty or clean, depending on "clean")
    # clean: if True, p is clean. Otherwise, p is dirty
    # LCD: last coupon date
    # NCD: next coupon date
    # settle: settlement date
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining
    equation = lambda y: get_bond_price(y, clean, LCD, NCD, settle, c, m, n) - p
    return optim.root(equation, 0.1).x[0]


def get_bond_price(y, clean, LCD, NCD, settle, c, m, n):
    # computes the bond price
    # p: price of a bond (dirty or clean, depending on "clean")
    # clean: if True, p is clean. Otherwise, p is dirty
    # y: yield
    # LCD: last coupon date
    # NCD: next coupon date
    # settle: settlement date
    # c: coupon rate (annualized)
    # m: number of compounding a year
    # n: number of coupons remaining

    v = (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(settle,'%b/%d/%Y')) \
        / (dt.strptime(NCD,'%b/%d/%Y') - dt.strptime(LCD,'%b/%d/%Y'))
    p = 0
    for i in range(1,n+1):
        disc_coup = (100*c/m)/((1+y/m)**(i-1))
        p += disc_coup
    p += 100/((1+y/m)**(n-1))
    p /= (1+y/m)**v
    if clean:  # compute the clean price
        AI = (1-v)*(100*c/m)  # accrued interest
        p -= AI
    return p

main()
plt.show()

# command 1 part에 이용
##==============================================================================
## bond1 today[채권A]
#        clean = False
#        p = 100.03125
#        y = 0.00883
#        c = 0.0125
#        m = 2
#        n = 5
#        LCD = 'Dec/15/2015'
#        NCD= 'Jun/15/2016'
#        settle = 'Feb/05/2016'
####==============================================================================
##
####==============================================================================
#### Bond2 today[채권B]
#        clean = False
#        p = 100.828125
#        y = 0.01491
#        c = 0.01625
#        m = 2
#        n = 12
#        LCD = 'Feb/15/2016'
#        NCD = 'Aug/15/2016'
#        settle = 'Feb/05/2016'
###==============================================================================

###==============================================================================
### Bond3 today[채권C]
#        clean = False
#        p = 103.859375
#        y =  0.01773
#        c = 0.0225
#        m = 2
#        n = 9
#        LCD = 'Nov/15/2015'
#        NCD = 'May/15/2016'
#        settle = 'Feb/05/2016'
##==============================================================================

#command = 2에 이용
##==============================================================================
## bond1 today[채권A]
#        clean = False
#        p = 100.03125
#        y = 0.00883
#        c = 0.0125
#        m = 2
#        n = 5
#        LCD = 'Dec/15/2015'
#        NCD= 'Jun/15/2016'
#        settle = 'Feb/05/2016'
####==============================================================================
##
####==============================================================================
#### Bond2 today[채권B]
#        clean = False
#        p = 100.828125
#        y = 0.01491
#        c = 0.01625
#        m = 2
#        n = 12
#        LCD = 'Feb/15/2016'
#        NCD = 'Aug/15/2016'
#        settle = 'Feb/05/2016'
###==============================================================================

###==============================================================================
### Bond3 today[채권C]
#        clean = False
#        p = 103.859375
#        y =  0.01773
#        c = 0.0225
#        m = 2
#        n = 9
#        LCD = 'Nov/15/2015'
#        NCD = 'May/15/2016'
#        settle = 'Feb/05/2016'
##==============================================================================
##==============================================================================
## bond1 future day[채권A]
#        clean = False
#        p = 100.15625
#        y = 0.00831
#        c = 0.0125
#        m = 2
#        n = 5
#        LCD = 'Dec/15/2015'
#        NCD = 'Jun/15/2016'
#        settle = 'Feb/25/2016'
##==============================================================================
##
##==============================================================================
## bond2 future day[채권B]
#        clean = False
#        p = 101.390625
#        y = 0.01399
#        c = 0.01625
#        m = 2
#        n = 12
#        LCD = 'Feb/15/2016'
#        NCD = 'Aug/15/2016'
#        settle = 'Feb/25/2016'
##==============================================================================

##==============================================================================
## Bond3 future day[채권C]
#        clean = False
#        p = 104.78125
#        y =  0.01685
#        c = 0.0225
#        m = 2
#        n = 9
#        LCD = 'Nov/15/2015'
#        NCD = 'May/15/2016'
#        settle = 'Feb/25/2016'
###==============================================================================
