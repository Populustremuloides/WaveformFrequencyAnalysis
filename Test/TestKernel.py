from Kernel import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mlp


print("started")
numDivisions = 2
kernelLength = 3

assert numDivisions < kernelLength, "num divisions must be less than kernel length"

norm = mlp.colors.Normalize(vmin=0,vmax=1)
cmap = cm.summer
m = cm.ScalarMappable(norm=norm, cmap=cmap)

for i in range(0,numDivisions + 1):
     proportion = float(i) / numDivisions
     kernel = defineTriangularKernel(kernelLength,proportion)
     plt.plot(kernel, c=m.to_rgba(proportion))

plt.title("example kernel distribution from 0-1, length " + str(kernelLength))
plt.savefig("testKernel-" + str(kernelLength) + "-" + str(numDivisions))



