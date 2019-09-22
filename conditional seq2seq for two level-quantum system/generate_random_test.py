import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# def gen_test(t,w0=1,w=0.5,omega0=1.0,n=11):

# rannum2 = np.random.rand(12)
# print(rannum1,rannum2)



def Ef(t):
    w0 = 1
    w = 0.5
    Omega0 = 1.
    T = 2. * np.pi / w

    # dt = T/1000.
    dt = 2. * np.pi / 0.5 / 10000.
    n = 10
    trange = (0., n * T)
    tls = np.arange(0, n * T, dt)

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # generate random envelop for the pulse
    Omega0Ls = moving_average(np.random.rand(12), 3)
    Omega0tls = np.linspace(0.1, 1, 10)

    Omega0f = interpolate.splrep(Omega0tls, Omega0Ls)

    newtls = np.linspace(0.1, 1, 100000)
    newOmega0Ls = interpolate.splev(newtls, Omega0f)

    # plt.plot(newtls,newOmega0Ls)

    # generate random frequency for the pulse

    wls = moving_average(np.random.rand(12), 3)

    wf = interpolate.splrep(np.linspace(0.1, 1., 10), moving_average(np.random.rand(12), 3))
    newwl = np.linspace(0.1, 1, 100000)
    newwls = interpolate.splev(newwl, wf)

    # random pulse is a product of the envelop and frequency

    Els = newOmega0Ls * np.sin(80*np.pi * newwls * newtls)
    # print(np.amax(Els))

    EEf = interpolate.splrep(newtls, Els)



    return 2*interpolate.splev(t/(n*T), EEf)

#
# x = Ef(np.linspace(4*np.pi,40*np.pi,100000))
# # # print(x.shape,x)
# #
# #
# # # print(t)
# #
# plt.plot(x)
# plt.savefig('test_gen.png')

    # return  Els
# print(max(newtls))
# print(max(trange))
# print(n*T)
# plt.plot(Ef(np.linspace(0.1*120,120,1000)))
# plt.savefig('test_gen.png')
plt.close()


