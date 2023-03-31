#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:55:13 2022

Author: williampennington-fitzgerald
Moayedi Lab
"""

import pandas as pd
import os
from pandas import read_hdf
from csv import writer as csvwriter
from numpy import percentile, max as npmax, min as npmin, array as nparray
from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, VideoWriter, line, circle, putText, destroyAllWindows, FONT_HERSHEY_SIMPLEX, LINE_AA
from os import path as ospath
import matplotlib.pyplot as plt
from scipy import stats
from math import atan, pi, radians, degrees
import matplotlib.pyplot as plt
import numpy as np 
import math
from math import sqrt, sin, cos
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from statistics import mean, median, stdev

outfile = '' #file location where you want your outputs to be saved.
vid = '' #video clip you want to analyze, for printing lines on.
h5 = '' #h5 file from DeepLabCut output

heading = 'DLC_resnet50_VFTrack5.1Oct15shuffle1_656500' #do not change, this specifies the DeepLabCut model used

con_const = .8  # How confident we want Deeplabcut to be of its point placement. 

#Savitsky-Golay smoothing function parameteres. User may need to change this based on video frame-rate parameters. 
bandwidth = 31
order = 9

#defining structures
bps = ['l.ary.tip', 'r.ary.tip', 'l.ary.low', 'r.ary.low', 'l.ary.mid', 'r.ary.mid', 'l.ary.midhigh', 'r.ary.midhigh', 'l.ary.midlow', 
       'r.ary.midlow','l.epi', 'r.epi', 'l.line.low', 'r.line.low', 'l.line.mid', 'r.line.mid']


def read_data(path): #reads data from h5 file
    data = read_hdf(path) 
    sorted_data = []
    for part in bps:
        plist = []
        for i in range(len(data[heading][part]['x'])):
            if data[heading][part]['likelihood'][i] >= con_const:
                plist.append((data[heading][part]['x'][i], data[heading][part]['y'][i],
                              data[heading][part]['likelihood'][i]))
             
            else:
                plist.append(None)
        sorted_data.append(plist)
    return sorted_data

data = read_data(h5)  

def my_reg_line (p_list): #creates a regression line given a list of points
    pfx = []
    pfy = []
    for item in p_list:
        if item is not None: 
            pfx.append(item[0])
            pfy.append(item[1])
    if len(pfx) > 1:
        pf = stats.linregress(pfx, pfy)
        slope = pf[0]
        yint = pf[1]
        return slope, yint
    else:
        return None
    
def intersection (line1,line2): #finds the intersection of two lines
    if line1 is not None and line2 is not None:
        xpoint = (line2[1]-line1[1])/(line1[0]-line2[0])
        ypoint = (line1[0]*xpoint)+line1[1]
        return xpoint, ypoint
    else:
        return None
    
def make_line (point1, point2): #creates a line given two points
    if point1 is not None and point2 is not None:
        slope = (point2[1]-point1[1])/(point2[0]-point1[0])
        yint = point1[1]-((slope)*(point1[0]))
        return slope, yint
    else:
        return None

def calc_angle (line1, line2): #calculates the angle between two lines
    if line1 is not None and line2 is not None:
        tan = abs((line1[0] - line2[0]) / (1 + line1[0] * line2[0]))
        return math.atan(tan)*57.2958
    else:
        return None
    
cap = VideoCapture(vid) 
frames = 60

#assigning points to anatomical structures
leftary = [data[0],data[2],data[4],data[6],data[8]]
rightary = [data[1],data[3],data[5],data[7],data[9]]
leftvf = [data[10], data[12],data[14]]
rightvf = [data[11], data[13],data[15]]
leftvfpoint = [data[0]]
rightvfpoint = [data[1]]
leftantcom = [data[10]]
rightantcom = [data[11]]

flatline = (0.0,0.0)      	

graph = []
vidgraph = []
rangle = []
langle = []
times = []
frametotal = []
ldisp = []
rdisp = []
timesout = []
ldispout = []
rdispout = []
#seeing if there is enough information to analyze each frame: 
for i in range(len(data[1])):
    leftary_now = []
    rightary_now = []
    leftvf_now = []
    rightvf_now = []
    leftvfpoint_now = []
    rightvfpoint_now = []
    leftantcom_now = []
    rightantcom_now = []
    
    is_there = False
    
    for j in range(len(leftary)):
        if leftary[j][i] is not None:
            leftary_now.append(leftary[j][i])
            is_there = True
        else:
            leftary_now.append(None)
        
    for j in range(len(rightary)):
        if rightary[j][i] is not None:
            rightary_now.append(rightary[j][i])
            is_there = True
        else:
            rightary_now.append(None)
            
    for j in range(len(leftvf)):
        if leftvf[j][i] is not None:
            leftvf_now.append(leftvf[j][i])
            is_there = True
        else:
            leftvf_now.append(None)      
            
    for j in range(len(rightvf)):
        if rightvf[j][i] is not None:
            rightvf_now.append(rightvf[j][i])
            is_there = True
        else:
            rightvf_now.append(None)   
            
    for j in range(len(leftvfpoint)):
        if leftvfpoint[j][i] is not None:
            leftvfpoint_now.append(leftvfpoint[j][i])
            is_there = True
        else:
            leftvfpoint_now.append(None)          
            
    for j in range(len(rightvfpoint)):
        if rightvfpoint[j][i] is not None:
            rightvfpoint_now.append(rightvfpoint[j][i])
            is_there = True
        else:
            rightvfpoint_now.append(None)      
            
    for j in range(len(leftantcom)):
        if leftantcom[j][i] is not None:
            leftantcom_now.append(leftantcom[j][i])
            is_there = True
        else:
           leftantcom_now.append(None) 
            
    for j in range(len(rightantcom)):
        if rightantcom[j][i] is not None:
            rightantcom_now.append(rightantcom[j][i])
            is_there = True
        else:
            rightantcom_now.append(None) 
    num_pts = 0  # number of legitimate points in a structure

    for item in leftary_now:
        if item is not None:
            num_pts += 1
    if num_pts < 2:
        is_there = False
    num_pts = 0
    for item in rightary_now:
        if item is not None:
            num_pts += 1
    if num_pts < 2:
        is_there = False           
    num_pts = 0
    num_ant = 0
    for item in leftvf_now:
        if item is not None:
            num_pts += 1
    if leftantcom_now[0] is not None and rightantcom_now[0] is not None:
        if abs(leftantcom_now[0][0] - rightantcom_now[0][0]) < 15 and abs(leftantcom_now[0][1] - rightantcom_now[0][1]) < 15:
            num_ant +=1
    if num_pts < 2 and num_ant < 1:
        is_there = False           
    num_pts = 0
    num_ant = 0
    for item in rightvf_now:
        if item is not None:
            num_pts += 1
    if leftantcom_now[0] is not None and rightantcom_now[0] is not None:
        if abs(leftantcom_now[0][0] - rightantcom_now[0][0]) < 15 and abs(leftantcom_now[0][1] - rightantcom_now[0][1]) < 15:
            num_ant +=1
    if num_pts < 2 and num_ant < 1:
        is_there = False
    num_pts = 0
    for item in leftvfpoint_now:
        if item is not None:
            num_pts += 1
    
    if num_pts < 1: 
        is_there = False    
    num_pts = 0
    for item in rightvfpoint_now:
        if item is not None:
            num_pts += 1
    if num_pts < 1:   
        is_there = False 
        #performing the calculations: 
    if is_there:
        leftary_reg = my_reg_line (leftary_now)
        rightary_reg = my_reg_line (rightary_now)
        leftvf_reg = my_reg_line (leftvf_now)
        rightvf_reg = my_reg_line (rightvf_now)
        
        if leftantcom_now[0] is not None and rightantcom_now[0] is not None and abs(leftantcom_now[0][0] - rightantcom_now[0][0]) < 15 and abs(leftantcom_now[0][1] - rightantcom_now[0][1]) < 15:
            midpoint_top = ((leftantcom_now[0][0] + rightantcom_now[0][0])/2, (leftantcom_now[0][1] + rightantcom_now[0][1])/2)
        else:                                                                               
            midpoint_top = intersection(leftvf_reg, rightvf_reg)
        midpoint_bottom = intersection(leftary_reg, rightary_reg)
        
        midline = make_line(midpoint_top, midpoint_bottom)
        leftvfpoint_now_fix = leftvfpoint_now[0]
        leftvfmax_line = make_line(midpoint_top, leftvfpoint_now_fix)
        rightvfpoint_now_fix = rightvfpoint_now[0]        
        rightvfmax_line = make_line(midpoint_top, rightvfpoint_now_fix)
        
        
        leftvidpoint =intersection(leftvfmax_line, leftary_reg)
        rightvidpoint =intersection(rightvfmax_line, rightary_reg)
        
        leftregint = intersection(leftvf_reg, leftary_reg)
        rightregint = intersection(rightvf_reg, rightary_reg)

        
        rightvf_angle = calc_angle(rightvfmax_line, midline)
        leftvf_angle = calc_angle(leftvfmax_line, midline)
        rightary_angle = calc_angle(rightary_reg, midline)
        leftary_angle = calc_angle(leftary_reg, midline)
        ary_ratio = (rightary_angle/leftary_angle)
        
        leftdisplace = sqrt(abs((leftvfpoint_now_fix[0]-midpoint_bottom[0])**2)+abs((leftvfpoint_now_fix[1]-midpoint_bottom[1])**2))*sin(leftary_angle/57.2958)
        rightdisplace = sqrt(abs((rightvfpoint_now_fix[0]-midpoint_bottom[0])**2)+abs((rightvfpoint_now_fix[1]-midpoint_bottom[1])**2))*sin(rightary_angle/57.2958)
        disp_ratio = rightdisplace/leftdisplace
        
        slope_displines = -1/midline[0]
        yint_ldispline =leftary_now[0][1]-(slope_displines*leftary_now[0][0])
        ldispline = (slope_displines, yint_ldispline)
        ldispline_int = intersection(ldispline, midline)
        yint_rdispline =rightary_now[0][1]-(slope_displines*rightary_now[0][0])
        rdispline = (slope_displines, yint_rdispline)
        rdispline_int = intersection(rdispline, midline)
            
        graph.append([leftary_reg[0],leftary_reg[1], rightary_reg[0], rightary_reg[1], midline[0], midline[1], leftary_angle, rightary_angle, ary_ratio, leftdisplace, rightdisplace, disp_ratio, i, i/60])
        vidgraph.append([midpoint_top, midpoint_bottom, leftvidpoint, rightvidpoint, round(leftary_angle,2), round(rightary_angle,2), round(leftvf_angle,2), round(rightvf_angle,2),round(leftdisplace, 2),round(rightdisplace, 2),leftregint, rightregint, leftary_now[0], rightary_now[0],ldispline_int, rdispline_int])
        rangle.append(rightary_angle)
        langle.append(leftary_angle)
        times.append(i/frames)
        frametotal.append(i)
        timesout.append(i/frames)
        ldisp.append(leftdisplace)
        ldispout.append(leftdisplace)
        rdisp.append(rightdisplace)
        rdispout.append(rightdisplace)
        
    else:
        graph.append([None,None,None,None,None,None,None,None,None])
        vidgraph.append(None)
        timesout.append(i/frames)
        ldispout.append(None)
        rdispout.append(None)


#drawing lines on each video frame
width = int(cap.get(CAP_PROP_FRAME_WIDTH))
height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
s, im = cap.read()
videotype='.mp4'
count = 0
name = vid[vid.rfind('\\') + 1:vid.rfind('.')] + '_with_lines'
if videotype == '.mp4':
    fourcc = VideoWriter.fourcc('m', 'p', '4', 'v')
elif videotype == '.avi':
    fourcc = VideoWriter.fourcc('x', 'v', 'i', 'd')
else:
    fourcc = 0
out = ospath.join(outfile,'/' + name + videotype)
w = VideoWriter(out, fourcc, frames, (width, height))
print('Printing lines on your video.')
while s:
    
    if vidgraph[count] is not None:
        line(im,(int(vidgraph[count][0][0]),int(vidgraph[count][0][1])), (int(vidgraph[count][1][0]),int(vidgraph[count][1][1])),(0, 255, 0), 2)#midline
        """line(im,(int(vidgraph[count][0][0]),int(vidgraph[count][0][1])), (int(vidgraph[count][10][0]),int(vidgraph[count][10][1])),(255, 0, 255), 2)# vf line left
        line(im,(int(vidgraph[count][0][0]),int(vidgraph[count][0][1])), (int(vidgraph[count][11][0]),int(vidgraph[count][11][1])),(255, 0, 255), 2)"""# vf line right
        line(im,(int(vidgraph[count][1][0]),int(vidgraph[count][1][1])), (int(vidgraph[count][2][0]),int(vidgraph[count][2][1])),(255, 0, 0), 2)#ary line left
        line(im,(int(vidgraph[count][1][0]),int(vidgraph[count][1][1])), (int(vidgraph[count][3][0]),int(vidgraph[count][3][1])),(255, 0, 0), 2)#ary line right
        line(im,(int(vidgraph[count][12][0]),int(vidgraph[count][12][1])), (int(vidgraph[count][14][0]),int(vidgraph[count][14][1])),(0, 255, 255), 2)
        line(im,(int(vidgraph[count][13][0]),int(vidgraph[count][13][1])), (int(vidgraph[count][15][0]),int(vidgraph[count][15][1])),(0, 255, 255), 2)
        circle(im,(int(vidgraph[count][0][0]),int(vidgraph[count][0][1])),2,(255, 255, 255), 2)#midpoint high
        circle(im,(int(vidgraph[count][1][0]),int(vidgraph[count][1][1])),2,(255, 255, 255), 2)#midpoint low
        circle(im,(int(vidgraph[count][12][0]),int(vidgraph[count][12][1])),2,(255, 255, 255), 2)#left vf max
        circle(im,(int(vidgraph[count][13][0]),int(vidgraph[count][13][1])),2,(255, 255, 255), 2)#right vf max
    w.write(im)
    s, im = cap.read()
    count += 1
cap.release()
w.release()
destroyAllWindows()
ospath.join(vid[:vid.rfind('videos')], name)

rangle_arr = nparray(rangle)
langle_arr = nparray(langle)
rdisp_arr = nparray(rdisp)
ldisp_arr = nparray(ldisp)
times_arr = nparray(times)

#smoothing funciton for output measures:
langle_smooth = savgol_filter(langle_arr, bandwidth, order)
rangle_smooth = savgol_filter(rangle_arr, bandwidth, order)
ldisp_smooth = savgol_filter(ldisp_arr, bandwidth, order)
rdisp_smooth = savgol_filter(rdisp_arr, bandwidth, order)

#velocity calculation:
timesvelo = times[0:-1]
lvelo = []
for i in range(len(ldisp_smooth)-1):
    if ldisp_smooth[i] is not None and ldisp_smooth[i+1] is not None:
        lvelo.append((ldisp_smooth[i+1]-ldisp_smooth[i])/(times[i+1]-times[i]))
    else:
        lvelo.append(None)
rvelo = []
for i in range(len(rdisp_smooth)-1):
    if rdisp_smooth[i] is not None and rdisp_smooth[i+1] is not None:
        rvelo.append((rdisp_smooth[i+1]-rdisp_smooth[i])/(times[i+1]-times[i]))
    else:
        lvelo.append(None)
        
lvelo_use = lvelo[1:-1]
rvelo_use = rvelo[1:-1]  

r_l_velo_ratio = []
for i in range(len(lvelo_use)):
    r_l_velo_ratio.append(rvelo_use[i]/lvelo_use[i])

#finding peaks, valleys, and cycle ranges
def peaks_cycle_range(peakydata, wide):
    peaks_tuple = find_peaks(peakydata, width=wide)
    peaks_frame = peaks_tuple[0]
    peaks_val = []
    valley_val = []
    cyclerange = []
    for i in range(len(peaks_frame)):
        peaks_val.append(peakydata[peaks_frame[i]])
    for i in range(len(peaks_frame)-1):
        valleyrange = peakydata[peaks_frame[i]:peaks_frame[i+1]]
        valley_val.append(min(valleyrange))
    for i in range(len(valley_val)):
        cyclerange.append(peaks_val[i]-valley_val[i])
    return cyclerange

def peaks_cycle_maxmin(peakydata, wide):
    peaks_tuple = find_peaks(peakydata, width=wide)
    peaks_frame = peaks_tuple[0]
    peaks_val = []
    valley_val = []
    for i in range(len(peaks_frame)):
        peaks_val.append(peakydata[peaks_frame[i]])
    for i in range(len(peaks_frame)-1):
        valleyrange = peakydata[peaks_frame[i]:peaks_frame[i+1]]
        valley_val.append(min(valleyrange))
    return peaks_val, valley_val


ldisp_cycrange = peaks_cycle_range(ldisp_smooth,10)
rdisp_cycrange = peaks_cycle_range(rdisp_smooth,10)
langle_cycrange = peaks_cycle_range(langle_smooth,10)
rangle_cycrange = peaks_cycle_range(rangle_smooth,10)
lvelo_maxmin = peaks_cycle_maxmin(lvelo_use,5)
rvelo_maxmin = peaks_cycle_maxmin(rvelo_use,5)

#calculating means and medians
ldisp_cycrange_mean = mean(ldisp_cycrange)
ldisp_cycrange_median = median(ldisp_cycrange)
rdisp_cycrange_mean = mean(rdisp_cycrange)
rdisp_cycrange_median = median(rdisp_cycrange)
langle_cycrange_mean = mean(langle_cycrange)
langle_cycrange_median = median(langle_cycrange)
rangle_cycrange_mean = mean(rangle_cycrange)
rangle_cycrange_median = median(rangle_cycrange)
lvelo_max_mean = mean(lvelo_maxmin[0])
lvelo_max_median = median(lvelo_maxmin[0])
lvelo_min_mean = mean(lvelo_maxmin[1])
lvelo_min_median = median(lvelo_maxmin[1])
rvelo_max_mean = mean(rvelo_maxmin[0])
rvelo_max_median = median(rvelo_maxmin[0])
rvelo_min_mean = mean(rvelo_maxmin[1])
rvelo_min_median = median(rvelo_maxmin[1])

#exporting output data
data_out = pd.DataFrame(graph, columns=['left ary reg slope', 'left ary reg yint', 'right ary reg slope', 'right ary reg yint', 'midline slope', 'midline yint', 'left ary angle', 'right ary angle', 'ary angle ratio (R:L)', 'right displacement', 'left displacement', 'displacement ratio (R:L)', 'frame', 'time'])
data_out.to_csv(outfile+'/' +os.path.basename(vid).split('.')[0] +'_rawdata_out.csv', index=False)
sumstats = [os.path.basename(vid).split('.')[0], rdisp_cycrange_mean, ldisp_cycrange_mean, rdisp_cycrange_mean/ldisp_cycrange_mean, rdisp_cycrange_median, ldisp_cycrange_median, rdisp_cycrange_median/ldisp_cycrange_median, rangle_cycrange_mean, langle_cycrange_mean, rangle_cycrange_mean/langle_cycrange_mean, rangle_cycrange_median, langle_cycrange_median, rangle_cycrange_median/langle_cycrange_median, rvelo_max_mean, lvelo_max_mean, rvelo_max_mean/lvelo_max_mean, rvelo_max_median, lvelo_max_median, rvelo_max_median/lvelo_max_median, rvelo_min_mean, lvelo_min_mean, rvelo_min_mean/lvelo_min_mean,rvelo_min_median, lvelo_min_median, rvelo_min_median/lvelo_min_median]
sumstats_out = pd.DataFrame([sumstats], columns= ['name','rdisp mean', 'ldisp mean', 'disp mean rat', 'rdisp med', 'ldisp med', 'disp med rat', 'rang mean', 'lang mean', 'ang mean rat', 'rang med', 'lang med', 'ang med rat', 'rvelo max mean', 'lvelo max mean', 'velo max mean rat', 'rvelo max med', 'lvelo max med', 'velo max med rat', 'rvelo min mean', 'lvelo min mean', 'velo min mean rat', 'rvelo min med', 'lvelo min med', 'velo min med rat'])
sumstats_out.to_csv(outfile+'/' +os.path.basename(vid).split('.')[0] +'_sumstats.csv', index=False)
rangle_smooth_list = rangle_smooth.tolist()
langle_smooth_list = langle_smooth.tolist()
rdisp_smooth_list = rdisp_smooth.tolist()
ldisp_smooth_list = ldisp_smooth.tolist()
lvelo.append(None)
rvelo.append(None)
angles_disps_dic = {'frame':frametotal,'time':times, 'right angle':rangle, 'right angle smooth':rangle_smooth_list,'right disp':rdisp, 'right disp smooth':rdisp_smooth_list, 'right velo':rvelo, 'left angle':langle, 'left angle smooth':langle_smooth_list, 'left disp':ldisp, 'left disp smooth':ldisp_smooth_list, 'left velo':lvelo}
angles_disps_out = pd.DataFrame(angles_disps_dic, columns = ['frame','time', 'right angle', 'right angle smooth','right disp', 'right disp smooth','right velo', 'left angle', 'left angle smooth','left disp', 'left disp smooth','left velo'])
angles_disps_out.to_csv(outfile+'/' +os.path.basename(vid).split('.')[0] +'_disp_angle_velo.csv', index=False)

#exporting plots of displacement, angle, and velocities over time
plt.plot(times,ldisp_smooth, color='blue', linewidth=0.5)
plt.plot(times,rdisp_smooth, color='red', linewidth=0.5)
plt.title(os.path.basename(vid).split('.')[0] + " Both VF Disp")
plt.xlabel("Time (seconds)")
plt.ylabel("Displacement (pixels)")
plt.savefig(outfile+'/' +os.path.basename(vid).split('.')[0] +'_plot_disp.png')
plt.close()

plt.plot(times,langle_smooth, color='blue', linewidth=0.5)
plt.plot(times,rangle_smooth, color='red', linewidth=0.5)
plt.title(os.path.basename(vid).split('.')[0] + " Both Ary Angle")
plt.xlabel("Time (seconds)")
plt.ylabel("Angle (degrees)")
plt.savefig(outfile+'/' +os.path.basename(vid).split('.')[0] +'_plot_angle.png')
plt.close()

timesvelo = times[1:-2]
plt.plot(timesvelo,lvelo_use, color='blue', linewidth=0.5)
plt.plot(timesvelo,rvelo_use, color='red', linewidth=0.5)
plt.title(os.path.basename(vid).split('.')[0] + "Both Velo")
plt.xlabel("Time (seconds)")
plt.ylabel("Velocity (pixels/sec)")
plt.savefig(outfile+'/' +os.path.basename(vid).split('.')[0] +'_plot_velo.png')
plt.close()




