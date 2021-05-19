import math
import numpy as np
import torch
# assume a kernel maximum height of 1

'''
Define the peak of a triangle whose  peak
lies at x = threshold and y = 1 in the x-y plane,
and whose total length along the x axis = kernelLength
'''
def getThreshold(kernelLength, interpolationVal):
    print(interpolationVal)
    print(kernelLength)
    print(interpolationVal * kernelLength)
    print(int(interpolationVal * kernelLength))
    return int(interpolationVal * kernelLength)

'''
Define the bottom left and right angles of a triangle whose 
peak lies at x = threshold and y = 1 in the x-y plane
'''
def getAngles(threshold, kernelLength):
    if threshold == kernelLength: # edge case 1
        print("edge case 1")
        leftAngle = math.tan(1.0 / (float(kernelLength) - 1))
        rightAngle = None
    elif threshold == 0: # edge case 2
        print("edge case 2")
        leftAngle = None
        rightAngle = math.tan(1.0 / (float(kernelLength) - 1))
    else: # default
        leftAngle = math.tan(1.0 / (threshold))
        rightAngle = math.tan(1.0 / (kernelLength - threshold - 1))

    return leftAngle, rightAngle

'''
generates a 1-d kernel that is either a downward slanting right triangle, 
an upward slanting right triangle, or some interpolation between those two.
'''
def defineTriangularKernel(kernelLength, interpolationVal=0):
    assert interpolationVal <= 1 and interpolationVal >=0, "Error, the following condition must be satisfied: 0 <= interpolationVal <= 1"

    threshold = getThreshold(kernelLength, interpolationVal)
    leftAngle, rightAngle = getAngles(threshold, kernelLength)

    kernel = []

    if rightAngle == None: # edge case 1 (downward slanting right triangle)
        for i in range(kernelLength):
            triangleHeight = i * math.tan(leftAngle)
            kernel.append(triangleHeight)
    elif leftAngle == None:  # edge case 2 (upward slanting right triangle)
        for i in range(kernelLength):
            triangleHeight = (kernelLength - i - 1) * math.tan(rightAngle)
            kernel.append(triangleHeight)
    else: # default
        for i in range(kernelLength):
            if i < threshold: # treat as a mini upward-slanting right trnagle
                b = i
                tanTheta = math.tan(leftAngle)
                triangleHeight = b * tanTheta
            elif i == threshold:
                triangleHeight = 1
            elif i > threshold: # treat as a mini downward-slanting right triangle
                b = kernelLength - i  - 1
                tanTheta = math.tan(rightAngle)
                triangleHeight = b * tanTheta

            kernel.append(triangleHeight)

    kernel = np.asarray(kernel)
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.unsqueeze(0)
    kernel = kernel.unsqueeze(0)
    return kernel

