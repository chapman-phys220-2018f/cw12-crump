#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Conner Carnahan
# Student ID: 1614309
# Email: Carna104@mail.chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: Midterm
###

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.jit
def rdot(m,r,nu,F,omega,t):
    temp = -nu*r[1]+r[0]-r[0]**3+F*np.cos(omega*t)
    return np.array([r[1],np.divide(temp,m)])

@nb.jit
def rungekutta4th(t0,tf,x0,y0, F, m = 1.0, nu = 0.25, omega = 1.0, N = 1000):
    
    npt = np.linspace(float(t0),float(tf),N)
    
    r = np.zeros((N,2))
    
    r[0,0] = x0
    r[0,1] = y0
    
    count = 1
    
    rk1 = np.zeros(2)
    rk2 = np.zeros(2)
    rk3 = np.zeros(2)
    rk4 = np.zeros(2)
    rk = np.zeros(2)
    
    dt = float((tf-t0)/N)
    
    while count < npt.size:
        rk1 = r[count-1,:]+dt*rdot(m,r[count-1,:],nu,F,omega,npt[count-1])
        rk2 = dt*rdot(m,r[count-1,:]+np.divide(rk1,2),nu,F,omega,npt[count-1])
        rk3 = dt*rdot(m,r[count-1,:]+np.divide(rk2,2),nu,F,omega,npt[count-1])
        rk4 = dt*rdot(m,r[count-1,:]+rk3,nu,F,omega,npt[count-1])
        r[count,:] = r[count-1,:] + np.divide(rk1+2*rk2+2*rk2+rk4,6)
        print(str(r[count,:]))
        count += 1
    
    plotboi(npt,r,"Plot for driven sombrero potential equation for $\nu$ = {},$x_0$ = {},$y_0$ = {},$m$ = {},$\omega$ = {},$F$ = {}".format(nu,x0,y0,m,omega,F))
    
def plotboi(t,r,titl):
    fig = plt.figure(figsize = (12,8))
    a = plt.axes()
    
    a.plot(t, r[:,0], label = "$x(t)$")
    a.plot(t, r[:,1], label = "$v(t)$")
    
    plt.title(titl)
    a.legend()
    plt.show()