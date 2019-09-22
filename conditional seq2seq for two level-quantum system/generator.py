from multi_level_solver_python.multi_level_python_test import *
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from solver import MultilevelSolver
from generate_random_test import *
import os
import random
#training data

# for i in range(200):
#     dt, Et = generator(Omega0=i/100,w0=1.0,w=1.5,n=30)
#     np.save('data/data-sin-w1.5/dls'+str(i)+'.npy',dt)
#     np.save('data/data-sin-w1.5/Els'+str(i)+'.npy',Et)
#     print('file '+str(i)+' finished')

#test data
# path = 'data/testing-data-threelevelw1w1.5_pulse_test/'
path = 'data/training-data-threelevel_sinw1w1.5_step1000/'
if not os.path.exists(path):
    os.makedirs(path)
for i in range(1,16):
    for j in range(1,16):
        # Omega0 = random.uniform(0,2)
        # w = random.uniform(0,2)
        # Omega0 = 1.
        # w = 1.
        Omega0 = i/10
        w = j/10
        dt, Et = generator(Omega0=Omega0,w=w,n=20*w)
        np.save(path + 'dls' + str(15 * (i - 1) + j ) + '.npy', dt)
        np.save(path + 'Els' + str(15* (i - 1) + j ) + '.npy', Et)
        print('file '+str(15*(i-1)+j)+' finished')

        # np.save(path + 'dls' + str(40 * (i - 1) + 2*(j-1)+1) + '.npy', dt)
        # np.save(path + 'Els' + str(40 * (i - 1) + 2*j-1) + '.npy', Et)
        # print('file ' + str(40 * (i - 1) + 2 * j-1) + ' finished')
        # dt = dt+np.random.random(size=dt.shape)*1e-3
        # # print(dt.shape)
        # # dt, Et = generator()
        # # dt = dt[:Et.shape[0]]
        # # np.save('data/testing-data-random3/dls'+str(25*(i-1)+j)+'.npy',dt)
        # # np.save('data/testing-data-random3/Els'+str(25*(i-1)+j)+'.npy',Et)
        # np.save(path+'dls' + str(40 * (i - 1) + 2*j) + '.npy', dt)
        # np.save(path+'Els' + str(40 * (i - 1) + 2*j) + '.npy', Et)
        # print('file '+str(40*(i-1)+2*j)+' finished')

#hidden test
# for i in range(1,11):
#     dt, Et = generator(Omega0=i/10.,w0=1.0,w=1.1,n=22)
#     dt = dt[:100000:10]
#     Et = Et[:100000:10]
#     r1 = r1[:100000:10]
#     r2 = r2[:100000:10]
#     i1 = i1[:100000:10]
#     i2 = i2[:100000:10]
#     # dt, Et = generator()
#     dt = dt[:Et.shape[0]]
#     # print(dt.shape,Et.shape)
#     if not os.path.exists('hidden_neural_analysis_rabi_pulse/hidden'+str(i/10.)):
#         os.makedirs('hidden_neural_analysis_rabi_pulse/hidden'+str(i/10.))
#     dt.astype('float32').tofile('hidden_neural_analysis_rabi_pulse/hidden'+str(i/10.)+'/dls.dat')
#     Et.astype('float32').tofile('hidden_neural_analysis_rabi_pulse/hidden'+str(i/10.)+'/Els.dat')
#     r1.astype('float32').tofile('hidden_neural_analysis_rabi_pulse/hidden' + str(i / 10.) + '/r1.dat')
#     r2.astype('float32').tofile('hidden_neural_analysis_rabi_pulse/hidden' + str(i / 10.) + '/r2.dat')
#     i1.astype('float32').tofile('hidden_neural_analysis_rabi_pulse/hidden' + str(i / 10.) + '/i1.dat')
#     i2.astype('float32').tofile('hidden_neural_analysis_rabi_pulse/hidden' + str(i / 10.) + '/i2.dat')
#     # print(dt.shape)
#
#     print('file '+str(i)+' finished')

#test random
# for i in range(1,5):
#
#     dt, Et = generator()
#     dt = dt[:Et.shape[0]]
#     np.save('data/test-data-random/dls'+str(i)+'.npy',dt)
#     np.save('data/test-data-random/Els'+str(i)+'.npy',Et)
#     print('file '+str(i)+' finished')
# dt, Et = generator()
# dt = dt[:Et.shape[0]]
# np.save('data/test-data-random/dls'+str(4)+'.npy',dt)
# np.save('data/test-data-random/Els'+str(4)+'.npy',Et)
# Omega0=1.
# w0=1.0
# w=2.
# n=40
# # #
# dt, Et,c = generator(Omega0=Omega0,w0=w0,w=w,n=n)
# #
# dt = dt[:100000:10]
# Et = Et[:100000:10]
# c = c[:100000:10]
# data = np.stack([dt,Et],1)
# np.save('mix_18.npy',data)
# Et.astype('float32').tofile('mix_18_Et.dat')
# dt.astype('float32').tofile('mix_18_dt.dat')
# c.astype('float32').tofile('c.dat')

# np.save('mix_18_Et.dat',Et)
# np.save('mix_18_dt.dat',dt)
# print(dt.shape,Et.shape)
# dt = np.load('data/training-data-random/dls'+str(400)+'.npy')
# Et = np.load('data/training-data-random/Els' + str(400) + '.npy')

# np.save('data/rabidt.npy',dt)
# np.save('data/rabiet.npy',Et)
plt.plot(dt.T,label='d(t)')
plt.plot(Et.T,label='E(t)')
plt.legend(loc='upper left')
plt.savefig('gen', format='png')
plt.close()
# print('step{0} finished'.format(i))
