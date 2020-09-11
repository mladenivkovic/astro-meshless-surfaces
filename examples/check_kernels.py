#!/usr/bin/env python3



#=======================================
# Various kernel checks
#=======================================

from matplotlib import pyplot as plt
import numpy as np
try:
    import meshless as ms
except ImportError:
    print("Didn't find 'meshless' module... be sure to add it to your pythonpath!")
    quit(2)





#=======================================
# Plot kernels and their derivatives
#=======================================


kernels = ms.kernels
kernel_derivatives = ms.kernel_derivatives

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


x = np.linspace(0, 1.1, 200)
h = 1

for kernel in kernels:
    W = [ms.W(i, h, kernel=kernel) for i in x]
    ax1.plot(x, W, label=kernel)

for kernel in kernel_derivatives:
    dWdr = [ms.dWdr(i, h, kernel=kernel) for i in x]
    ax2.plot(x, dWdr, label=kernel)


ax1.set_xlabel('x')
ax1.set_ylabel('W(x)')
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\frac{dW(x)}{dx}$')
ax1.legend()
ax2.legend()


plt.show()






#==============================
# Check kernel gradients
#==============================


xk = 1
yk = 1
xl = 1.50
yl = 1.50
hl = 1
dx = xl-xk
dy = yl-yk
r = np.sqrt(dx**2 + dy**2)
d = 1e-10

print("Checking kernel gradients")
print('{0:20} {1:20} {2:20} {3:20}'.format('kernel:', 'analytical', 'finite difference', 'ratio'))



for kernel in kernel_derivatives:
    estimate_x = (ms.psi(xl+d, yl, xk, yk, hl, kernel=kernel, L=100) - ms.psi(xl, yl, xk, yk, hl, kernel=kernel, L=100)) / d
    estimate_y = (ms.psi(xl, yl+d, xk, yk, hl, kernel=kernel, L=100) - ms.psi(xl, yl, xk, yk, hl, kernel=kernel, L=100)) / d
    #  estimate_r = (ms.psi(xl+d, yl+d, xk, yk, hl, kernel=kernel, L=100) - ms.psi(xl, yl, xk, yk, hl, kernel=kernel, L=100)) / d

    exx = ms.dWdr(r, hl, kernel=kernel) * dx / r
    exy = ms.dWdr(r, hl, kernel=kernel) * dy / r
    #  exr = ms.dWdr(r, hl, kernel=kernel)

    print('{0:20} {1:20.6f} {2:20.6f} {3:20.6f}'.format(kernel, exx, estimate_x, exx/estimate_x))
    print('{0:20} {1:20.6f} {2:20.6f} {3:20.6f}'.format( '', exy, estimate_y, exy/estimate_y))

    print()


