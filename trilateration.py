import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class Trilat:
   
   mode = ''
   
   def __init__(self):
      mode = "trilateration"
   
   def p2Ddist(self,x1,y1,x2,y2):
       x = x2-x1
       y = y2-y1
       return math.sqrt((x*x)+(y*y))
   
   # returns the two intersection points of for the two circles [x,y,radius]
   # None if the circles do not intersect
   def circle_intersections(self,beacon1,beacon2):
       x1 = beacon1[0]
       x2 = beacon2[0]
       y1 = beacon1[1]
       y2 = beacon2[1]
       r1 = beacon1[2]
       r2 = beacon2[2]
       dx = x2-x1
       dy = y2-y1
       dist = math.sqrt((dx*dx)+(dy*dy))
   	 # formula di erone per area triangolo interno http://2000clicks.com/MathHelp/GeometryConicSectionCircleIntersection.aspx
   	 # vedi anche http://www.ambrsoft.com/TrigoCalc/Circles3/Intersection.htm
       erone2 = (dist+r1+r2)*(-dist+r1+r2)*(dist-r1+r2)*(dist+r1-r2)
       if ( erone2 >= 0 ): # radicando non negativo
           area_erone = 0.25*math.sqrt(erone2)
     
           xbase = (0.5)*(x1+x2) + (0.5)*(x2-x1)*(r1*r1-r2*r2)/(dist*dist)
           xdiff = 2*(y2-y1)*area_erone/(dist*dist) 
           ybase = (0.5)*(y1+y2) + (0.5)*(y2-y1)*(r1*r1-r2*r2)/(dist*dist)
           ydiff = 2*(x2-x1)*area_erone/(dist*dist) 
           return (xbase+xdiff,ybase-ydiff),(xbase-xdiff,ybase+ydiff)
       else:
           return None # no intersection
   
   # TRILATERATION
   # find the two points of intersection between beacon0 and beacon1
   # will use beacon2 to determine which of the two
   def trilaterate(self,anchors,threshold=0.1):
   	x1 = anchors[0][0]
   	x2 = anchors[1][0]
   	y1 = anchors[0][1]
   	y2 = anchors[1][1]
   	r1 = anchors[0][2]
   	r2 = anchors[1][2]
   	isNotIntersecting = False
   	d12 = self.p2Ddist(x1,y1,x2,y2)
   	if(d12 > (r1+r2)):
   		isNotIntersecting = True
   		
   	if(isNotIntersecting):
   		ratio = d12/(r1+r2)
   		anchors[0][2] = anchors[0][2]*ratio + 0.001
   		anchors[1][2] = anchors[1][2]*ratio + 0.001
   		print("Increased distances by {0}".format(ratio))
   		if (ratio > threshold):
   			print("NOTE: anchor 0 and anchor 1: no reliable intersection")
   
   	intersections = self.circle_intersections(anchors[0],anchors[1])
   	dist1 = self.p2Ddist(anchors[2][0],anchors[2][1],
   					  intersections[0][0],intersections[0][1])
   	dist2 = self.p2Ddist(anchors[2][0],anchors[2][1],
   					  intersections[1][0],intersections[1][1])
   	# find the intersection closest to 3rd anchor
   	if ( math.fabs(dist1-anchors[2][2]) < math.fabs(dist2-anchors[2][2]) ):
   		point = intersections[0]
   		dist = dist1
   	else:
   		point = intersections[1]
   		dist  = dist2
   
   	# check distance threshold
   	deltaDiff = math.fabs(dist - anchors[2][2])
   	if deltaDiff < threshold*dist:
   		print("Found reliable intersection point: {0}".format(point))
   	else:
   		print('NOTE: anchor 2 distance to point exceeds threshold: {0}-{1}'.format(deltaDiff,threshold*dist))
   	return point


   # given anchor (antenna) positions
   def getPointCoord(self,distances,antennas):
   	beacons = [[antennas.iloc[0].x,antennas.iloc[0].y,distances[0]],
   				  [antennas.iloc[1].x,antennas.iloc[1].y,distances[1]],
   				  [antennas.iloc[2].x,antennas.iloc[2].y,distances[2]]] # x, y, distances
   
   	# compute the intersection between the two closest beacons and use the farther beacon to tie-break.
   	idxDist = np.argsort(distances)
   	beacons = [[antennas.iloc[idxDist[0]].x,antennas.iloc[idxDist[0]].y,distances[idxDist[0]]],
   				  [antennas.iloc[idxDist[1]].x,antennas.iloc[idxDist[1]].y,distances[idxDist[1]]],
   				  [antennas.iloc[idxDist[2]].x,antennas.iloc[idxDist[2]].y,distances[idxDist[2]]]] # x, y, distances
   	threshold=2.5  # point is within tresh% of the distance from the 3rd beacon
   	point = self.trilaterate(beacons,threshold)
   	print("Intersection: {0}".format(point))
   	return point