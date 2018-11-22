
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import splev
from scipy.interpolate import splrep

from scipy import integrate

#from matplotlib.backends.backend_pdf import PdfPages

import time
import sys


def find_eps(t, a, adot, addot, H):
    startt = time.time()
    #finds the start time of inflation and the stop time of inflation by detecting \epsilon crossings. find the number of efolds. returns inflation start and stop times, number of efolds, and the index corresponding to the inflation stop time.

    #t: an array of time (M_{pl}^{-1})
    #a: array of a
    #adot: array of adot
    #addot: array of addot
    #H: array of H

    #first construct the slowroll parameter /epsilon

    eps = -(1.0/(adot/a)**(2.0))*(addot/a-(adot/a)**(2.0))


    # # plt.plot(t, eps, 'g-')
    # plt.plot(t, a, 'g-')
    # #plt.plot(t[0:indend], eps[0:indend], 'r-')
    # #plt.axvline(tinf_end, linestyle='--', color='blue')
    # plt.show()

    

    t_wind = 10.0  #this is a window below which you dont allow eps=1 (t measured in M_{pl}^{-1})
    
    nswitch = 1  #later if nswitch is still 1, you will calculate Nef

    if np.min(eps)<0.1:

        if eps[0]>=1.0:
            feps_red_start = UnivariateSpline(t,(eps-0.1),s=0)
            zs_eps_st = feps_red_start.roots()

            if len(zs_eps_st)==1.0:
                tstart = zs_eps_st[0]
                print("only one 0.1 crossing?? -- no inflation?? -- check")
            elif len(zs_eps_st)>1.0:
                tstart = zs_eps_st[0]
            else:
                #print("check : (phi_0, phi_dot_0, dphi_k1_0, dphi_k2_0, dphi_k3_0, dphi_k4_0, dphi_k5_0, dphi_k6_0, dphi_k7_0, dphi_k8_0, dphi_k9_0, dphi_k10_0, dphi_k11_0, dphi_k12_0, dphi_dot_k1_0, dphi_dot_k2_0, dphi_dot_k3_0, dphi_dot_k4_0, dphi_dot_k5_0, dphi_dot_k6_0, dphi_dot_k7_0, dphi_dot_k8_0, dphi_dot_k9_0, dphi_dot_k10_0, dphi_dot_k11_0, dphi_dot_k12_0): {0}".format((phi[0], phidot[0], dphik1[0], dphik2[0], dphik3[0], dphik4[0], dphik5[0], dphik6[0], dphik7[0], dphik8[0], dphik9[0], dphik10[0], dphik11[0], dphik12[0], dphidotk1[0], dphidotk2[0], dphidotk3[0], dphidotk4[0], dphidotk5[0], dphidotk6[0], dphidotk7[0], dphidotk8[0], dphidotk9[0], dphidotk10[0], dphidotk11[0], dphidotk12[0])))
                sys.exit("something went wrong finding start time")

            feps_red_stop = UnivariateSpline(t,(eps-1.0),s=0)
            zs_eps_sp = feps_red_stop.roots()

            if len(zs_eps_sp)>1:
                clicker=0
                i=0
                while clicker<1:
                    if (feps_red_stop(zs_eps_sp[i])-0.0)>(feps_red_stop(zs_eps_sp[i]-1.0)-0.0) and (feps_red_stop(zs_eps_sp[i]+1.0)-0.0)>(feps_red_stop(zs_eps_sp[i])-0.0) and zs_eps_sp[i]>tstart and zs_eps_sp[i]>t_wind:
                        tend = zs_eps_sp[i]
                        clicker=1
                    else:
                        i = i+1

            else:
                print("len(zs_eps_sp): {0}".format(len(zs_eps_sp)))
                #print("check : (phi_0, phi_dot_0, dphi_k1_0, dphi_k2_0, dphi_k3_0, dphi_k4_0, dphi_k5_0, dphi_k6_0, dphi_k7_0, dphi_k8_0, dphi_k9_0, dphi_k10_0, dphi_k11_0, dphi_k12_0, dphi_dot_k1_0, dphi_dot_k2_0, dphi_dot_k3_0, dphi_dot_k4_0, dphi_dot_k5_0, dphi_dot_k6_0, dphi_dot_k7_0, dphi_dot_k8_0, dphi_dot_k9_0, dphi_dot_k10_0, dphi_dot_k11_0, dphi_dot_k12_0): {0}".format((phi[0], phidot[0], dphik1[0], dphik2[0], dphik3[0], dphik4[0], dphik5[0], dphik6[0], dphik7[0], dphik8[0], dphik9[0], dphik10[0], dphik11[0], dphik12[0], dphidotk1[0], dphidotk2[0], dphidotk3[0], dphidotk4[0], dphidotk5[0], dphidotk6[0], dphidotk7[0], dphidotk8[0], dphidotk9[0], dphidotk10[0], dphidotk11[0], dphidotk12[0])))
                sys.exit("either didnt find a zero crossing or missed the downward crossing -- should have inflated some")

        elif eps[0]<1.0 and eps[0]>0.1:
            feps_red_start = UnivariateSpline(t,(eps-0.1),s=0)
            zs_eps_st = feps_red_start.roots()

            if len(zs_eps_st)==1.0:
                tstart = zs_eps_st[0]
                print("only one 0.1 crossing? no inflation?? -- check")
            elif len(zs_eps_st)>1:
                tstart = zs_eps_st[0]
            else:
                #print("check : (phi_0, phi_dot_0, dphi_k1_0, dphi_k2_0, dphi_k3_0, dphi_k4_0, dphi_k5_0, dphi_k6_0, dphi_k7_0, dphi_k8_0, dphi_k9_0, dphi_k10_0, dphi_k11_0, dphi_k12_0, dphi_dot_k1_0, dphi_dot_k2_0, dphi_dot_k3_0, dphi_dot_k4_0, dphi_dot_k5_0, dphi_dot_k6_0, dphi_dot_k7_0, dphi_dot_k8_0, dphi_dot_k9_0, dphi_dot_k10_0, dphi_dot_k11_0, dphi_dot_k12_0): {0}".format((phi[0], phidot[0], dphik1[0], dphik2[0], dphik3[0], dphik4[0], dphik5[0], dphik6[0], dphik7[0], dphik8[0], dphik9[0], dphik10[0], dphik11[0], dphik12[0], dphidotk1[0], dphidotk2[0], dphidotk3[0], dphidotk4[0], dphidotk5[0], dphidotk6[0], dphidotk7[0], dphidotk8[0], dphidotk9[0], dphidotk10[0], dphidotk11[0], dphidotk12[0])))
                sys.exit("something went wrong finding start time -- didnt find 0.1 crossing")

            
            feps_red_stop = UnivariateSpline(t,(eps-1.0),s=0)
            zs_eps_sp = feps_red_stop.roots()

            if len(zs_eps_sp)>=1.0:
                clicker=0
                i=0
                while clicker<1:
                    if (feps_red_stop(zs_eps_sp[i])-0.0)>(feps_red_stop(zs_eps_sp[i]-1.0)-0.0) and (feps_red_stop(zs_eps_sp[i]+1.0)-0.0)>(feps_red_stop(zs_eps_sp[i])-0.0) and zs_eps_sp[i]>tstart and zs_eps_sp[i]>t_wind:
                        tend = zs_eps_sp[i]
                        clicker=1
                    else:
                        i=i+1

            else:
                #print("check : (phi_0, phi_dot_0, dphi_k1_0, dphi_k2_0, dphi_k3_0, dphi_k4_0, dphi_k5_0, dphi_k6_0, dphi_k7_0, dphi_k8_0, dphi_k9_0, dphi_k10_0, dphi_k11_0, dphi_k12_0, dphi_dot_k1_0, dphi_dot_k2_0, dphi_dot_k3_0, dphi_dot_k4_0, dphi_dot_k5_0, dphi_dot_k6_0, dphi_dot_k7_0, dphi_dot_k8_0, dphi_dot_k9_0, dphi_dot_k10_0, dphi_dot_k11_0, dphi_dot_k12_0): {0}".format((phi[0], phidot[0], dphik1[0], dphik2[0], dphik3[0], dphik4[0], dphik5[0], dphik6[0], dphik7[0], dphik8[0], dphik9[0], dphik10[0], dphik11[0], dphik12[0], dphidotk1[0], dphidotk2[0], dphidotk3[0], dphidotk4[0], dphidotk5[0], dphidotk6[0], dphidotk7[0], dphidotk8[0], dphidotk9[0], dphidotk10[0], dphidotk11[0], dphidotk12[0])))
                sys.exit("didnt find a 1 crossing? or missed a downward crossing? should have inflated some")


        else:
            tstart = 0.0

            feps_red_stop = UnivariateSpline(t,(eps-1.0),s=0)
            zs_eps_sp = feps_red_stop.roots()

            if len(zs_eps_sp)>=1.0:
                clicker=0
                i=0
                while clicker<1:
                    if (feps_red_stop(zs_eps_sp[i])-0.0)>(feps_red_stop(zs_eps_sp[i]-1.0)-0.0) and (feps_red_stop(zs_eps_sp[i]+1.0)-0.0)>(feps_red_stop(zs_eps_sp[i])-0.0) and zs_eps_sp[i]>t_wind:
                        tend = zs_eps_sp[i]
                        clicker=1
                    else:
                        i=i+1

            else:
                #print("check : (phi_0, phi_dot_0, dphi_k1_0, dphi_k2_0, dphi_k3_0, dphi_k4_0, dphi_k5_0, dphi_k6_0, dphi_k7_0, dphi_k8_0, dphi_k9_0, dphi_k10_0, dphi_k11_0, dphi_k12_0, dphi_dot_k1_0, dphi_dot_k2_0, dphi_dot_k3_0, dphi_dot_k4_0, dphi_dot_k5_0, dphi_dot_k6_0, dphi_dot_k7_0, dphi_dot_k8_0, dphi_dot_k9_0, dphi_dot_k10_0, dphi_dot_k11_0, dphi_dot_k12_0): {0}".format((phi[0], phidot[0], dphik1[0], dphik2[0], dphik3[0], dphik4[0], dphik5[0], dphik6[0], dphik7[0], dphik8[0], dphik9[0], dphik10[0], dphik11[0], dphik12[0], dphidotk1[0], dphidotk2[0], dphidotk3[0], dphidotk4[0], dphidotk5[0], dphidotk6[0], dphidotk7[0], dphidotk8[0], dphidotk9[0], dphidotk10[0], dphidotk11[0], dphidotk12[0])))
                sys.exit("didnt find 1 crossing? should have inflated some")

    else:
        nswitch=0
        tstart=0.0
        tend=t[-1]
        
            
    #now you should have tstart, tend, nswitch
    #print("tstart: {0}".format(tstart))
    #print("tend: {0}".format(tend))
    #print("nswitch: {0}".format(nswitch))
    ###hardcoded for now: tstart
    tstart = 0.0

    #define interoplation function fH (to integrate up Nef)
    fH = interp1d(t,H)

    #if nswitch=1 integrate up Nef, if it is =0 then Nef=0 
    #(note that we take the 0th element of quad for Nef because quad outputs an object of the form (Nef, error)
    if nswitch==1:
        Nef = integrate.quad(fH,tstart,tend)[0]
    else:
        Nef=0.0

    print("Nef: {0}".format(Nef))
    #find the element along the array t at which inflation ends so we can truncate the arrays there
    indend = np.abs(t-tend).argmin()
    print("tend: {0}".format(t[indend]))

    endd = time.time()

    print("efolds finished in: {0}".format(endd - startt))

    return (Nef, tstart, tend, eps, indend)




