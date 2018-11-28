#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Conner Carnahan
# Student ID: 1614309
# Email: carna104@mail.chapman.edu
#FULL NAME :NATANAEL ALPAY
#ID        :002285534
#email:alpay100@mail.chapman.edu

# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: HW 10
###


import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.jit
def rdot(m,r,nu,F,omega,t):
    """rdot(m = float, r = 2D array of floats, nu = float, F = float, omega = float, t = float):
    This is a helper function the calculates the derivative of a vector r according to the sombrero potential for the parametrizations given in the input"""
    temp = -nu*r[1]+r[0]-r[0]**3+F*np.cos(omega*t)
    return np.array([r[1],temp/m])

@nb.jit
def rungekutta4th(t0,tf,n,x0,y0, F, m = 1.0, nu = 0.25, omega = 1.0, dt = 0.001, plotall = True):
    """rungekutta4th(float t0, float tf,int n,float x0,float y0,float F,float m = 1.0, float nu = 0.25, float omega = 1.0, float dt = 0.001, boolean plotall =  
    True):
    This is the calculation and plotting method for the sombrero plot, all floats above are used as parameters for the diff eq and three plots are displayed at the 
    end of computation
    n is an int which will give how many repeats of the period t0-tf will be computed/ displayed
    plotall is a value that sets whether all plots should be created, or only the poincare section (since n = 1000 will be annoying as hell)"""
    periodlength = int((tf-t0)/dt)
    npt = np.linspace(float(t0),float(n*tf),n*periodlength)
    
    r = np.zeros((npt.size,2))
    index = np.arange(n)
    poin = np.zeros((n,2))
    
    r[0,0] = x0
    r[0,1] = y0
    
    count = 1
    
    rk1 = np.zeros(2)
    rk2 = np.zeros(2)
    rk3 = np.zeros(2)
    rk4 = np.zeros(2)
    rk = np.zeros(2)
    
    while count < npt.size:
        rk1 = dt*rdot(m,r[count-1,:],nu,F,omega,npt[count-1])
        rk2 = dt*rdot(m,r[count-1,:]+np.divide(rk1,2),nu,F,omega,npt[count-1])
        rk3 = dt*rdot(m,r[count-1,:]+np.divide(rk2,2),nu,F,omega,npt[count-1])
        rk4 = dt*rdot(m,r[count-1,:]+rk3,nu,F,omega,npt[count-1])
        rk = rk1+2*rk2+2*rk3+rk4
        r[count,:] = r[count-1,:] + np.divide(rk,6)
        count += 1
    
    for i in index:
        poin[i,:] = r[i*periodlength,:]
    
    if plotall:
        plotit(npt,r,"Plot for driven sombrero potential equation for nu = {},x_0 = {},y_0 = {},m = {},omega = {},F = {}".format(nu,x0,y0,m,omega,F))
        plot(r,"Plot for driven sombrero potential equation for nu = {},x_0 = {},y_0 = {},m = {},omega = {},F = {}".format(nu,x0,y0,m,omega,F))
    poinplot(poin,"plot for driven sombrero potential equation for nu = {},x_0 = {},y_0 = {},m = {},omega = {},F = {}".format(nu,x0,y0,m,omega,F))

    
@nb.jit
def plot(r,titl):
    """plot(2D array r,string titl):
    this is a helper plotting method that will generate the phase portrait for an array r, and will give it the title titl"""
    fig = plt.figure(figsize = (16,9))
    a = plt.axes()
    
    plt.scatter(r[:,0],r[:,1], s=1, c=(0,0,0), alpha = 0.5)
    a.set(xlabel = "x(t)", ylabel = "y(t)")
    
    plt.title(titl)
    plt.show()
    
@nb.jit
def plotit(t,r,titl):
    """plot(2D array r,string titl):
    this is a helper plotting method that will generate the time plot for an array r, and will give it the title titl"""
    fig = plt.figure(figsize = (16,9))
    a = plt.axes()
 
    a.plot(t, r[:,0], label = "$x(t)$")
    a.plot(t, r[:,1], label = "$v(t)$")
    
    plt.title(titl)
    a.legend()
    plt.show()
    
def poinplot(r,titl):
    """plot(2D array r,string titl):
    this is a helper plotting method that will generate the Poincare Section for an array r, and will give it the title titl"""
    fig = plt.figure(figsize = (12,8))
    a = plt.axes()
    
    plt.scatter(r[:,0],r[:,1], s=1, c=(0,0,0), alpha = 1)
    a.set(xlabel = "x(t)", ylabel = "y(t)")
    
    plt.title("Poincare section " + titl)
    plt.show()
