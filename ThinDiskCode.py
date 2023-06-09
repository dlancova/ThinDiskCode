import numpy as np

#help:
def mysqrt(x): return np.sqrt(x)

## IMPORTANT R IN KERR

# Event Horizon radius in Kerr metric
def horizon(spin):
    return 1 + np.sqrt(1-spin**2)

def horizon_minus(spin):
    return 1 - np.sqrt(1-spin**2)

# Innermost Stable Circular orbit in Kerr metric
def isco(spin):
    Z1 = 1.0 + np.power(1.0 - spin * spin,1./3.) * (np.power(1.0+spin,1./3.) + np.power(1.0 - spin,1./3.))
    Z2 = np.sqrt(3.0 * spin * spin + Z1 * Z1)

    return 3.0 + Z2 - np.sqrt( (3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2) )

def isco_minus(spin):
    Z1 = 1.0 + np.power(1.0 - spin * spin,1./3.) * (np.power(1.0+spin,1./3.) + np.power(1.0 - spin,1./3.))
    Z2 = np.sqrt(3.0 * spin * spin + Z1 * Z1)

    return 3.0 + Z2 + np.sqrt( (3-Z1) * (3.0 + Z1 + 2.0 * Z2) )

# Keplerian omega at given r 
def kepnu(r,a):
    delta = r**2 - 2.*r + a**2
    return (a**2 - 2.*a*np.sqrt(r) + r**2)/(np.sqrt(delta)*(a + r*np.sqrt(r)))

# Efficienty of a NT disk 
def eta(spin):
    return (1 - np.sqrt(1-2/(3*isco(spin))))


# important radii in kerr metric

def ergosphere(spin,theta):
    return (1+np.sqrt(1-(spin**2)*(np.cos(theta))**2))

def photon(spin):
    return 2*(1+np.cos(2/3 * np.arccos(-spin)))


def photon_minus(spin):
    return 2*(1+np.cos(2/3 * np.arccos(spin)))

def mb(spin):
    return (2-spin+ 2* np.sqrt((1-spin)))


def mb_minus(spin):
    return (2 + spin+ 2* np.sqrt((1+spin)))


## CONSTANTS AND UNIT CONVERSION

# Sollar mass in grams
Msun = 1.988435e33 
# c in cm/s
c = 2.998e10
# Gravitational constant in cgs
G = 6.674e-8

# no efficiency? (TO DO!)
Ledd_f = lambda m: 1.26e38 * m

M_f = lambda m: m*Msun
Mdot_f = lambda mdot,m: mdot*(c**2)/Ledd_f(m)

# outer edge of the disk, used for some integrations (make it big (inf))
rout = 1000

def MdotEdd(M,spin):
    return  Ledd_f(M*Msun)/(eta(spin) * np.square(c))

### NT coefficients ###
# Q with fix from Page&Thorne 76 (exp is + instead of -)
# L from Page&Thorne 76, as a separate function
class calculate_kerr_coefficients():
    def __init__ (self,a,r,rin=0):
        
        self.A = 1.+((2.*((a**2)*(r**-3.)))+((a**2)*(r**-2.)))
        self.B = 1.+(a*(mysqrt((r**-3.))))
        self.C = 1.+((2.*(a*(mysqrt((r**-3.)))))+(-3./r))
        self.D = 1.+(((a**2)*(r**-2.))+(-2./r))
        aux0=(3.*((a**4.)*(r**-4.)))+((-4.*((a**2)*(r**-3.)))+(4.*((a**2)*(r**-2.))))
        self.E = 1.+aux0
        self.F = 1.+((-2.*(a*(mysqrt((r**-3.)))))+((a**2)*(r**-2.)))
        self.G = 1.+((a*(mysqrt((r**-3.))))+(-2./r))
        self.H = 1.+((-4.*(a*(mysqrt((r**-3.)))))+(3.*((a**2)*(r**-2.))))
        
        r1 = (mysqrt(2.))*(np.cos(0.333333*(np.arccos(a)) - np.pi/3))
        r2 = (mysqrt(2.))*(np.cos(0.333333*(np.arccos(a)) + np.pi/3))
        r3 = - (mysqrt(2.))*(np.cos(0.333333*(np.arccos(a))))
        
        if (rin == 0):
            Risco = isco(a)
        else:
            Risco = rin
        
        aux0 = np.log(((((2.**-0.5)*(mysqrt(r)))-r1)/(((2.**-0.5)*(mysqrt(Risco)))-r1)))
        aux1 = np.log(((((2.**-0.5)*(mysqrt(r)))-r2)/(((2.**-0.5)*(mysqrt(Risco)))-r2)))
        aux2 = np.log(((((2.**-0.5)*(mysqrt(r)))-r3)/(((2.**-0.5)*(mysqrt(Risco)))-r3)))
        aux3 = ((((-1.5*((((r3-a)**2))*aux2))/(r3-r2))/(r3-r1))/r3)+(-1.5*((2.**-0.5)*(a*(np.log(((mysqrt(r))*(Risco**-0.5)))))))
        aux4 = ((((-1.5*((((r1-a)**2))*aux0))/(r1-r3))/(r1-r2))/r1)+(((((-1.5*((((r2-a)**2))*aux1))/(r2-r3))/(r2-r1))/r2)+aux3)
        aux5 = (((2.**-0.5)*(mysqrt(r)))+aux4)-((2.**-0.5)*(mysqrt(Risco)))
        self.Q = self.B/np.sqrt(self.C) * np.sqrt(2/r) * aux5


       


# Inner
class inner_region:
    def __init__(self,r,a,m,mdot,alpha,rin=0):
        
        coefs = calculate_kerr_coefficients(a,r,rin)

        foo = 1/coefs.B*np.sqrt(coefs.C)*1/coefs.H*coefs.Q
        self.H = 5.5e4 * m * mdot * foo

        foo = coefs.B * np.sqrt(coefs.C)* 1/coefs.D * coefs.H * 1/coefs.Q
        self.Sigma = 35.3 * 1/alpha * 1/mdot * np.power(r,3/2) * foo
        
        foo = np.power(coefs.B,2) * 1/coefs.D * np.power(coefs.H,2)*np.power(coefs.Q,-2)
        self.rho = 3.2e-4 * 1/alpha * 1/m * np.power(mdot,-2) * np.power(r,3/2) * foo
        
        foo = 1/coefs.B * np.power(coefs.C,-1/2)*np.sqrt(coefs.D)*1/coefs.H * coefs.Q
        self.vr = -4.3e9 * alpha * np.square(mdot) * np.power(r,-5/2)        
        
        foo = np.power(coefs.D,-1/4)*np.power(coefs.H,1/4)
        self.T = 6.35e7 * np.power(alpha,-1/4)*np.power(m,-1/4)*np.power(r,-3/8) * foo

        foo = np.square(coefs.B)*np.sqrt(coefs.C)*np.power(coefs.D,-17/16)*np.power(coefs.H,25/16)*np.power(coefs.Q,-2)
        self.tau = 1.12e-3 * np.power(alpha,-17/16) *np.power(m,-1/16) * np.power(mdot,-2)*np.power(r,93/32) * foo
        #self.F = 7e26*(1/m*mdot)*1/np.power(r,3)*np.power(coefs.B,-1)*np.sqrt(1/coefs.C)*coefs.L

# Middle region

class middle_region():
    def __init__(self,r,a,m,mdot,alpha,rin=0):
        coefs = calculate_kerr_coefficients(a,r,rin)
        
        foo = np.power(coefs.B,-1/5)*np.sqrt(coefs.C)*np.power(coefs.D,-1/10)*np.power(coefs.H,-1/2)*np.power(coefs.Q,1/5)
        self.H = 1.3e2*np.power(alpha,-1/10)*np.power(m,9/10)*np.power(mdot,1/5) * np.power(r,21/20) * foo

        foo = np.power(coefs.B,-3/5)*np.sqrt(coefs.C)*np.power(coefs.D,-4/5)*np.power(coefs.Q,3/5)
        self.Sigma = 6.52e4 * np.power(alpha,-4./5.) * np.power(m,1./5.) * np.power(mdot,3/5) * np.power(r,-3/5) * foo

        foo = np.power(coefs.B,-2/5) * np.power(coefs.D,-7/10) * np.power(coefs.H,1/2) * np.power(coefs.Q,2/5)
        self.rho = 2.51e1 *  np.power(alpha,-7/10) * np.power(m,-7/10) * np.power(mdot,2/5) * np.power(r,-33/20) * foo
        
        foo = np.power(coefs.B,3/5) * np.power(coefs.C,-1/2) * np.power(coefs.D,-1/5) *  np.power(coefs.Q,-3/5)
        self.vr = - 2.24e6 * np.power(alpha,4/5) * np.power(m,-1/5) * np.power(mdot,2/5) * np.power(r,-2/5) * foo
        
        foo = np.power(coefs.B,-2/5) * np.power(coefs.D,-1/5) *  np.power(coefs.Q,2/5)
        self.T = 4.11e8 * np.power(alpha,-1/5) * np.power(m,-1/5) * np.power(mdot,2/5) * np.power(r,-9/10) * foo
        
        foo = np.power(coefs.B,-1/10) * np.power(coefs.C,1/2) * np.power(coefs.D,-4/5) * np.power(coefs.H,1/4) * np.power(coefs.Q,1/10)
        self.tau = 2.16e1 * np.power(alpha,-4/5) * np.power(m,1/5) * np.power(mdot,1/10) *np.power(r,3/20) * foo    

        
        return

# Outer region
class outer_region:
    def __init__(self,r,a,m,mdot,alpha,rin=0):
 
        coefs = calculate_kerr_coefficients(a,r,rin)
        
        foo = np.power(coefs.B,-3/20)*np.sqrt(coefs.C)*np.power(coefs.D,-1/10)*np.power(coefs.H,-19/40)*np.power(coefs.Q,3/20)
        self.H = 6.88e2*np.power(alpha,-1/10)*np.power(m,9/10)*np.power(mdot,3/20) * np.power(r,9/8) * foo

        foo = np.power(coefs.B,-7/10)*np.sqrt(coefs.C)*np.power(coefs.D,-4/5)*np.power(coefs.H,-1/20)*np.power(coefs.Q,7/10)
        self.Sigma = 2.35e5 * np.power(alpha,-4./5.) * np.power(m,1./5.) * np.power(mdot,7./10.) * np.power(r,-3/4) * foo

        foo = np.power(coefs.B,-11/20) * np.power(coefs.D,-7/10) * np.power(coefs.H,17/40) * np.power(coefs.Q,11/20)
        self.rho = 1.72e2 *  np.power(alpha,-7/10) * np.power(m,-7/10) * np.power(mdot,11/20) * np.power(r,-15/8) * foo
        
        foo = np.power(coefs.B,7/10) * np.power(coefs.C,-1/2) * np.power(coefs.D,-1/5) * np.power(coefs.H,1/20) * np.power(coefs.Q,-7/10)
        self.vr = - 6.72e5 * np.power(alpha,4/5) * np.power(m,-1/5) * np.power(mdot,3/10) * np.power(r,-1/4) * foo
        
        foo = np.power(coefs.B,-3/10) * np.power(coefs.D,-1/5) * np.power(coefs.H,1/20) * np.power(coefs.Q,3/10)
        self.T = 1.16e8 * np.power(alpha,-1/5) * np.power(m,-1/5) * np.power(mdot,3/10) * np.power(r,-3/4) * foo
        
        foo = np.power(coefs.B,-1/5) * np.power(coefs.C,1/2) * np.power(coefs.D,-4/5) * np.power(coefs.H,1/5) * np.power(coefs.Q,1/5)
        self.tau = 79. * np.power(alpha,-4/5) * np.power(m,1/5) * np.power(mdot,1/5) * foo    

        return

## Flux

def flux(r,m,a,mdot,rin=0):
    coefs = calculate_kerr_coefficients(a,r,rin)
    Mdot = Mdot_f(mdot,m)
    M = M_f(m)
    fr = coefs.Q/(coefs.B*np.sqrt(coefs.C))
    foo = 3*G*M*Mdot/(8*np.pi*np.power(r,3))
    return foo*fr




        