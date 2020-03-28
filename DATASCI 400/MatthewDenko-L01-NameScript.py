#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 17:34:45 2018

@author: matt.denko
"""

#### Creating Function that returns my name

def myname(name = "Your Name"):
    name = "Hi my name is " + name
    return(name)
    
myname("Matthew Denko")


#### Print Date and Time with Import & Print Statement

import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))


### Source for datetime https://www.w3resource.com/python-exercises/python-basic-exercise-3.php