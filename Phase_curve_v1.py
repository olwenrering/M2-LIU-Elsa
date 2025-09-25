#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Louis-Julien Cartigny
# January 2025
# Phase curves

import numpy as np
import matplotlib.pyplot as plt
from Orbital_motion import compute_true_anomaly
from Transits import eclipse, eclipse_impact_parameter
from TRAPPIST1_parameters import *
from Solar_System_constants import *
from astropy.modeling.models import BlackBody
from astropy import units as u
#from Phase_curve_TTV import phase_TTV

def phase_angle(omega, nu, i):
    """
    Determines the phase angle of a planet from its orbital parameters (in rad).

    :param omega: the argument of pericentre (in rad)
    :type omega: float

    :param nu: the true anomaly (in rad)
    :type nu: float

    :param i: the inclination (in rad)
    :type i: float

    :return: alpha
    :rtype: float
    """

    alpha = np.arccos(np.sin(omega+nu)*np.sin(i))
    return alpha

def phase_function(alpha):
    """
    Determines the phase function of a Lambert sphere.

    :param alpha: the phase angle (in rad)
    :type alpha: float

    :return: g
    :rtype: float
    """

    g = (np.sin(alpha)+(np.pi-alpha)*np.cos(alpha))/np.pi
    return g

def phase_planet(t,P,t0=0):
    """
    Determines the phase of a planet at a given time.

    :param t: the time (in days)
    :type t: float

    :param P: the orbital period (in days)
    :type P: float

    :param t0: the reference time (in days)
    :type t0: float

    :return: phase
    :rtype: float
    """

    phase = np.sin(((t+t0)/P)*2*np.pi)/2+0.5 # equation 15
    return phase

def star_planet_separation(a,e,nu):
    """
    Determines the distance between a planet and its star using its orbital parameters.

    :param a: the semimajor axis (in m)
    :type a: float

    :param e: the eccentricity
    :type e: float

    :param nu: the true anomaly (in rad)
    :type nu: float

    :return: r
    :rtype: float
    """

    r = (a*(1-e**2))/(1+e*np.cos(nu))
    return r

def flux_star(L,d):
    """
    Determines the flux received from a star (in W/m^2) at a distance d.

    :param L: the star luminosity (in W)
    :type L: float

    :param d: the distance (in m)
    :type d: float

    :return: F
    :rtype: float
    """

    F = L/(4*np.pi*d**2)
    return F

def flux_planet(F_star):
    """
    Determines the flux reemitted by a planet (in W/m^2) from the one it receives from its star considering the planet is a black body.

    :param F_star: the flux received by the planet from its star (in W/m^2)
    :type F_star: float

    :return: F_planet
    :rtype: float
    """

    F_planet = F_star*2/3
    return F_planet

def surface_sphere(R):
    """
    Determines the surface of a sphere of radius R.

    :param R: the radius (in m)
    :type R: float

    :return: S
    :rtype: float
    """

    S = 4*np.pi*R**2
    return S

def luminosity_planet_dayside(F_planet,R_planet):
    """
    Determines the luminosity of the dayside of a planet from the flux it reemits and its radius.

    :param F_planet: the flux reemitted by the planet's dayside (in W/m^2)
    :type F_planet: float

    :param R_planet: the planet radius (in m)
    :type R_planet: float

    :return: L_planet
    :rtype: float
    """

    L_planet = F_planet * surface_sphere(R_planet)/2
    return L_planet

def luminosity_bb(T, lambda_1, lambda_2):

    bb = BlackBody(temperature=T*u.K)
    wav = np.arange(lambda_1, lambda_2) * u.micron
    flux = bb(wav)
    return np.trapezoid(flux)

def phase_curve(L_star, L_planet, R_star, R_planet, phase_planet, eclipse):
    """
    Determines the phase curve of a planet from its luminosity, its star's luminosity and its phase function expressed as the ratio between the planet and star's luminosities in ppm.

    :param L_star: the star luminosity (in W)
    :type L_star: float

    :param L_planet: the planet luminosity (in W)
    :type L_planet: float

    :param R_star: the star radius (in m)
    :type R_star: float

    :param R_planet: the planet radius (in m)
    :type R_planet: float

    :param phase_planet: the phase function of the planet
    :type phase_planet: float

    :param eclipse: True if the planet is in eclipse, False otherwise
    :type eclipse: bool

    :return: curve
    :rtype: float
    """
    
    print((L_planet/L_star)*(R_planet/R_star)**2)
    print((L_planet/L_star)*(R_star/R_planet)**2)
    curve = (L_planet/L_star)*phase_planet*(R_planet/R_star)**2 * (-1*eclipse+1) + 1  # *10**6 to have in ppm
    return curve


def main():
    
    t_start = 0.2
    t_end = 0.7 # simulation duration in days
    nb_points = 10000 # number of points in the time array

    t = np.linspace(t_start,t_end,nb_points) # time array in days


    # For LHS3844 b

    e = 0.0
    P = 0.46292964#*24*3600
    omega = 0.0 # bcs e = 0
    i = 88.5*np.pi/180
    a = 0.00622*149597870700 # a in m
    R_star = 0.1886240*R_Sun
    R = 1.303*R_Earth 
    L_star = (10**(-2.5833))*L_Sun

    
    nu = compute_true_anomaly(0,e,P,t)
    alpha = phase_angle(omega,nu,i)
    #phase = phase_function(alpha)
    # t0_b = omega_b/(2*np.pi)*P_b
    # phase_b = phase_planet(t,P_b,t0_b)
    phase = phase_TTV(P, t_start, t_end, [P], nb_points)[0]
    b = eclipse_impact_parameter(a,i,e,R_star,omega)
    eclipsee = eclipse(P,a,R_star,R,i,np.arccos(phase)/(2*np.pi),e,omega,b)

    r = star_planet_separation(a,e,nu)

    flux_starr = flux_star(L_star,r)
    flux = flux_planet(flux_starr)
    L = luminosity_planet_dayside(flux,R)

    phase_curvee = phase_curve(L_star,L,R_star,R,phase,eclipsee)
    # np.savetxt("Phase_curve_v1_output/phase_curve_b.txt",np.concatenate((t.reshape(nb_points,1),phase_curve_b.reshape(nb_points,1)),axis=1))
    
    # Total signal

    #phase_curve_total = phase_curve_b + phase_curve_c + phase_curve_d + phase_curve_e + phase_curve_f + phase_curve_g + phase_curve_h
    # np.savetxt("Phase_curve_v1_output/phase_curve_total.txt",np.concatenate((t.reshape(nb_points,1),phase_curve_total.reshape(nb_points,1)),axis=1))



    # Plot

    # plt.figure()
    # plt.plot(t,phase_b,label="b")
    # plt.plot(t,phase_c,label="c")
    # plt.xlabel("Time (days)")
    # plt.ylabel("Phase")
    # plt.title("Phase of planets of TRAPPIST-1")
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.figure(figsize=(16,9))
    plt.plot(t,phase_curvee)
    plt.xlabel("Time (days)")
    plt.ylabel("$F_{planet}/F_{star}$ (ppm)")
    plt.title("Phase curve of LHS3844b")
    plt.legend()
    plt.grid()
    #plt.savefig("Phase_curve_v1_plots/Phase_curves_TRAPPIST1_bolometric.png", bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    main()