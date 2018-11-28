
class runs:
    
    def __init__(self,  a, phi, phidot,  H, rho, drho2, hpotential, Nef, infl):
        
        #these arrays (whose entries can be arrays) hold the parameter values and solutions from each run
        
        #
        self.a = a
        self.phi = phi
        self.phidot = phidot
        #
        self.H = H
        self.rho = rho
        self.drho2 = drho2
        self.hpotential = hpotential
        #
        self.Nef = Nef
        self.infl = infl
