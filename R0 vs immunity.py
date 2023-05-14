"""
Created on Nov 4th 2020
@author: Madi Stoddard
"""

import pylab as pp
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.colors as colors
from matplotlib.patches import Patch

class Parameterize_ODE():
    def __init__(self, S0, I10, I20, Ir10, Ir20, R10, R20, R120):
        #Target / binding partner concentration
        self.S0 = S0
        self.I10 = I10
        self.I20 = I20
        self.Ir10 = Ir10
        self.Ir20 = Ir20
        self.R10 = R10
        self.R20 = R20
        self.R120 = R120
        
    def sirodes(self, t, C, p): #renamed from peptideodes
        #Parameters
        mu, N, CI1v2, CI2v1, R01, R02, rho, _, gamma, tau = p 
    
        #Compartments
        S, I1, I2, Ir1, Ir2, R1, R2, R12= C 
        
        
        #Beta rate
        beta1 = gamma * R01 
        beta2 = gamma * R02 
        betaR1 = beta1 * (1 - CI2v1)
        betaR2 = beta2 * (1 - CI1v2)
        
        
        if scenario == 1 or scenario == 2:
            #system of equations - scen 1 / 2
            dSdt = mu*N - beta1*S*I1/N - beta1*S*Ir1/N - beta2*S*I2/N - beta2*S*Ir2/N -mu*S + rho*R1 + rho*R2
            dI1dt = beta1*S*I1/N + beta1*S*Ir1/N - gamma*I1 - mu*I1
            dI2dt = beta2*S*I2/N + beta2*S*Ir2/N - gamma*I2 - mu*I2
            dIr1dt = betaR1*R2*Ir1/N + betaR1*R2*I1/N - gamma*Ir1 - mu*Ir1 
            dIr2dt = betaR2*R1*Ir2/N + betaR2*R1*I2/N - gamma*Ir2 - mu*Ir2
            dR1dt = gamma*I1 - betaR2*R1*Ir2/N - betaR2*R1*I2/N - rho*R1 + rho*R12 - mu*R1
            dR2dt = gamma*I2 - betaR1*R2*Ir1/N - betaR1*R2*I1/N - rho*R2 + rho*R12 - mu*R2  
            dR12dt = gamma*Ir1 + gamma*Ir2 - mu*R12 - rho*R12*2
            #totCases = beta1*S*I1/N +beta1*S*Ir1/N + beta2*S*I2/N + beta2*S*Ir2/N + betaR1*R2*Ir1/N + betaR1*R2*I1/N + betaR2*R1*Ir2/N + betaR2*R1*I2/N
        
        if scenario == 3:
            #scen 3: rho2 = tau*rho1
            dSdt = mu*N - beta1*S*I1/N - beta1*S*Ir1/N - beta2*S*I2/N - beta2*S*Ir2/N -mu*S + rho*R1 + tau*rho*R2
            dI1dt = beta1*S*I1/N + beta1*S*Ir1/N - gamma*I1 - mu*I1
            dI2dt = beta2*S*I2/N + beta2*S*Ir2/N - gamma*I2 - mu*I2
            dIr1dt = betaR1*R2*Ir1/N + betaR1*R2*I1/N - gamma*Ir1 - mu*Ir1 
            dIr2dt = betaR2*R1*Ir2/N + betaR2*R1*I2/N - gamma*Ir2 - mu*Ir2
            dR1dt = gamma*I1 - betaR2*R1*Ir2/N - betaR2*R1*I2/N - rho*R1 + tau*rho*R12 - mu*R1
            dR2dt = gamma*I2 - betaR1*R2*Ir1/N - betaR1*R2*I1/N - tau*rho*R2 + rho*R12 - mu*R2  
            dR12dt = gamma*Ir1 + gamma*Ir2 - mu*R12 - rho*R12 - tau*rho*R12    
         
        return[dSdt, dI1dt, dI2dt, dIr1dt, dIr2dt, dR1dt, dR2dt, dR12dt] #return new set of equations
        
        

    def solvesirodes(self, t, p):
        
        #Solver for ODEs

        t0 = t[0]
        tf = t[-1]
        
        y0 = [self.S0, self.I10, self.I20, self.Ir10, self.Ir20, self.R10, self.R20, self.R120]
        solarray = solve_ivp(self.sirodes, (t0, tf), y0, t_eval = t, method = 'BDF', args = (p,))


        return solarray


#Parameter estimates
mu = .00002466 #births per person / day
N = 330000000  #population
CI1v2 =  1. # cross immunity 1 against 2 
CI2v1 = 1. # cross immunity  2 against 1
gamma = 1/10 #1/days   
R01 = 8.2 #interested in range of R0 from 1 to 10 for each strain people (arbitrary)
R02 = 1.5 #people  (arbitrary)
rho1 = 1/547.501 #1/18 1/months changed to 1/547.501 1/days
rho2 = 1/547.501 
tau = 1# loss of immunity constant 

#initial conditions, two strain simulation
#1. Symmetric adaptive evasion  (cross immunity 2->1 = 1->2 and both < 1)
#2. Asymmetric adaptive evasion (cross immunity 2-> 1 != 1->2 and 1->2 < 1)

I10 = 5266123.67822816
I20 = 1.
Ir10 = 0
Ir20 = 0
R10 = 284480055.3885778
R20 = 0
R120 = 0
S0 = N - I10 - I20 - R10

p0 = [mu, N, CI1v2, CI2v1, R01, R02, rho1, rho2, gamma, tau]
scenario = 1

def multisensitivity(paramx, minx, maxx, paramy, miny, maxy, p0, scenario, outcome = 'long-term binary', contour = True):
    #2D sweep wrapper plotting heatmaps
    # paramx: heatmap x-axis sweep parameter
    # minx: lower bound of sweep for paramx
    # maxx: upper bound of sweep for paramx
    # paramy: heatmap y-axis sweep parameter
    # p: list of parameters
    # bound = True if bound drug measured for Cmax/AUC
    # outcome: 'AUCss', 'Cmax', 'TI'
    # contour = True if contour plot desired
    
    # Create a linearly spaced time vector
    p = p0.copy()
    dt = 1
    xt = np.arange(0, 3650, dt)

    #Number of x and y-axis points
    points = 25
    #minx = np.log10(minx)      
    #maxx = np.log10(maxx)
    #miny = np.log10(miny)
    #maxy = np.log10(maxy)
    prangey = np.linspace(miny, maxy, num=points)
    prangex = np.linspace(minx, maxx, num=points)
    
    Outcomearray = np.zeros((points,points))
    if outcome != 'binary':
        Outcomearray2 = np.zeros((points,points))
        Outcomearray3 = np.zeros((points,points))
    i2 = 0
    
    for valx in prangex:
        #X-axis sweep
        p[paramx] = valx
        print(i2)
        i1 = 0
        for valy in prangey:
            #y-axis sweep
            p[paramy] = valy
            
            if scenario == 1:
                #set cross-immunity parameters to be equal
                p[3] = p[2]
                
            #run model
            po = Parameterize_ODE(S0, I10, I20, Ir10, Ir20, R10, R20, R120) 
            infected = po.solvesirodes(xt, p).y 
            infectedStrain1 = infected[1,:] + infected[3,:]
            infectedStrain2 = infected[2,:] + infected[4,:]
            totalinfected = infectedStrain1 + infectedStrain2
            
            gamma = p[8]
            N = p[1]

            #outcome metrics
            if outcome == 'binary': #LONG TERM simplified graph
                if infectedStrain1[-1] < 3300 and infectedStrain2[-1] > 3300:  # S2 dominates
                    result = 1
                if infectedStrain2[-1] < 3300 and infectedStrain1[-1] > 3300: #s1 dominates
                    result = 0
                if infectedStrain2[-1] > 3300 and infectedStrain1[-1] > 3300: #coexistence
                    result = .5
                #Z array for heatmap  
                Outcomearray[i1,i2] = result

            if outcome == 'SS':
                infected1_people = np.trapz(infectedStrain1[3285: 3650], xt[3285: 3650])
                result1 = infected1_people * gamma / N
            
                infected2_people = np.trapz(infectedStrain2[3285: 3650], xt[3285: 3650])
                result2 = infected2_people * gamma / N

                infected_people = np.trapz(totalinfected[3285: 3650], xt[3285: 3650])
                result3 = infected_people * gamma / N
                
                #Z array for heatmap  
                Outcomearray[i1,i2] = result1 
                Outcomearray2[i1,i2] = result2  
                Outcomearray3[i1,i2] = result3
        
            if outcome == 'short term':
                infected1_people = np.trapz(infectedStrain1[0: 180], xt[0: 180])
                result1 = infected1_people * gamma / N
              
                infected2_people = np.trapz(infectedStrain2[0: 180], xt[0: 180])
                result2 = infected2_people * gamma / N
               
                infected_people = np.trapz(totalinfected[0: 180], xt[0: 180])
                result3 = infected_people * gamma / N
                
                 #Z array for heatmap  
                Outcomearray[i1,i2] = result1 
                Outcomearray2[i1,i2] = result2  
                Outcomearray3[i1,i2] = result3
            

            i1+=1 #y axis counter
        i2 += 1 #x axis counter
         
    if paramy == 9:
        prangey = 1/(prangey*rho1) #duration of immunity conversion

    #Heatmap plotting
    if outcome == 'binary':
        cmap1 = colors.ListedColormap(['yellow', 'green', 'blue'])
    else:
        cmap1 = pp.cm.get_cmap("plasma")
    #Parameter names list
    p0 = [mu, N, CI1v2, CI2v1, R01, R02, rho1, rho2, gamma, tau]
    params_list = ["Birth rate", "Total population", "Original cross-immunity against Invader", "Invader cross-immunity against Original", 
                   "Original strain R0", "Invader strain R0", "Duration of immunity (days)", "Invader sterilizing immunity (days)", "Recovery rate (1/days)" , "Invader duration of immunity (days)"] 
   
    if outcome == 'binary':
        fig, ax = pp.subplots()
        pp.pcolor(prangex, prangey, Outcomearray, cmap = cmap1, shading = 'auto')#, norm = colors.LogNorm())
        legend_handles = [Patch(color='yellow', label='Invader strain extinct'),
                          Patch(color='green', label='Co-circulation'),
                          Patch(color='blue', label='Original strain extinct')]  
        pp.legend(handles=legend_handles, bbox_to_anchor=(1.45, 1.), fontsize=10, handlelength=.8)
        pp.ylabel(params_list[paramy])
        pp.xlabel(params_list[paramx])
        pp.xlim((prangex[0], prangex[-1]))
        pp.ylim((prangey[0], prangey[-1]))
        if scenario == 1:
            pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig2.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
        elif scenario == 2:
            pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig4.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
        else:
            pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig6.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
    
    else:
        vmin = np.min([np.min(Outcomearray), np.min(Outcomearray2), np.min(Outcomearray3)])
        vmax = np.max([np.max(Outcomearray), np.max(Outcomearray2), np.max(Outcomearray3)])
        outcomes = [Outcomearray, Outcomearray2, Outcomearray3]
        for i in [0, 1, 2]:
            fig, ax = pp.subplots()
            pp.pcolor(prangex, prangey, outcomes[i], cmap = cmap1, shading = 'auto', vmin = vmin, vmax = vmax)
            
            #contour lines
            minval = np.min(outcomes[i])
            maxval = np.max(outcomes[i])
            print(minval, maxval)
            if maxval - minval < 0.1:
                CS = ax.contour(prangex, prangey, outcomes[i], colors = 'k')
                ax.clabel(CS, CS.levels, fmt = '%.2f', fontsize = 12)
            elif maxval < 2: 
                levels = np.linspace(0.1, round(maxval,1), int(round(maxval,1)/0.1)+1)
                CS = ax.contour(prangex, prangey, outcomes[i], levels = levels, colors = 'k')
                ax.clabel(CS, levels, fmt = '%.1f', fontsize = 12)
            else:
                levels = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
                CS = ax.contour(prangex, prangey, outcomes[i], levels = levels, colors = 'k')
                ax.clabel(CS, levels, fmt = '%.1f', fontsize = 12)
            pp.colorbar()
            pp.ylabel(params_list[paramy])
            pp.xlabel(params_list[paramx])
            pp.xlim((prangex[0], prangex[-1]))
            pp.ylim((prangey[0], prangey[-1]))
    
            if outcome == 'SS':
                titles = ["SS yearly fraction of population infected with Original strain",
                          "SS yearly fraction of population infected with Invader strain",
                          "SS yearly fraction of population infected"]
                pp.title(titles[i])
                if scenario == 1:
                    if i == 0:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig3D.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    elif i == 1:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig3E.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    else:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig3F.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                elif scenario == 2:
                    if i == 0:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig5D.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    elif i == 1:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig5E.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    else:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig5F.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                else:
                    if i == 0:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/FigS1D.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    elif i == 1:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/FigS1E.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    else:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/FigS1F.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
            if outcome == 'short term':
                titles = ["Fraction of population infected with Original strain in 6 mo",
                          "Fraction of population infected with Invader strain in 6 mo",
                          "Fraction of population infected in 6 mo"]
                pp.title(titles[i])
                if scenario == 1:
                    if i == 0:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig3A.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    elif i == 1:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig3B.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    else:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig3C.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                elif scenario == 2:
                    if i == 0:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig5A.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    elif i == 1:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig5B.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    else:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/Fig5C.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                else:
                    if i == 0:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/FigS1A.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    elif i == 1:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/FigS1B.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)
                    else:
                        pp.savefig("C:/Users/amper/Desktop/work/R0 vs immunity/R0 vs immunity figures/FigS1C.tiff", dpi=600, facecolor='w', edgecolor='w',pad_inches=0.)


PO = Parameterize_ODE(S0, I10, I20, Ir10, Ir20, R10, R20, R120)

t = np.arange(0, 3650, .01)
psol = PO.solvesirodes(t, p0).y
#QC figure to test steady-state initial conditions
#pp.plot(t, psol[0,:])
pp.plot(t, psol[1,:]) 
pp.plot(t, psol[2,:])
pp.plot(t, psol[3,:])
pp.plot(t, psol[4,:])
#pp.plot(t, psol[5,:])
#pp.plot(t, psol[6,:])
#pp.plot(t, psol[7,:])

plt.xlabel("Time (days)")
pp.ylabel("People")
pp.legend(["I1", "I2", "Ir1", "Ir2"])
plt.show()


#p0 = mu, N, CI1v2, CI2v1, R01, R02, rho, gamma, tau, alpha, t_half

#Scenario 1: Symmetric immune evasion
#x-axis: R0 S2
#y-axis: cross-immunity S1 vs S2
scenario = 1
paramx = 5
minR0 = 0
maxR0 = 20
paramy = 2
minCI = 0
maxCI = 1
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'binary')
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'SS')
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'short term')

#Scenario 2: Asymmetric immune evasion
#x-axis: R0 S2
#y-axis: S1 cross immunity
scenario = 2
paramx = 5
minR0 = 0
maxR0 = 20
paramy = 2
minCI = 0
maxCI = 1
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'binary')
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'SS')
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'short term')

#Scenario 3: Loss of immunogenicity
#x-axis: R0 S2
#y-axis: duration of S2 immunity
scenario = 3
paramx = 5
minR0 = 0
maxR0 = 20
paramy = 9
maxCI = 1
minCI = 5
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'binary')
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'SS')
multisensitivity(paramx, minR0, maxR0, paramy, minCI, maxCI, p0, scenario, outcome = 'short term')
