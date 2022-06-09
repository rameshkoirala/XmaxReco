#!/usr/bin/env python3
# coding: utf-8

# This is the latest code to perform shower maximum (Xmax) reconstruction using convolutional neural network (CNN).
# 
# 
# Archived codes:
# 
#     4 . GRAND_ReCon_grid_archive4.ipynb           (archived on Jun 02, 2022)
#         Reason: 
#         Rbf interpolation is replaced by Fourier interpolation in this version. Interpolation is done 
#         on shower plane rather than the ground plane. This produced a smooth interpolation. SRTCleaning
#         function is introduced for the first time in this version. Also events with less than 5 hits after
#         SRTCleaning was removed from training. 
#         The reason to archive this version is because reconstruction of energy, zenith, azimuth, Xmax is 
#         done at once. Only Xmax reconstruction will be done in the next version.
# 
#     3 . GRAND_ReCon_grid_archive3.ipynb           (archived on May 19, 2022)
#         Reason: 
#         Rbf interpolation produced negative p2p values. Rbf will be replaced by Fourier interpolation. 
#         Cuts: p2p<800, and time<0.1 produces inconsistent patterns on hit antennas. Cuts will be investigated.
#         Reconstruction of energy, zenith, azimuth, Xmax is done. Only Xmax reconstruction will be done.
# 
#     2 . GRAND_ReCon_starshaped_archive2.ipynb      (archived on April, 2022)
#         Reason: 
#         CNN prefers grid layout rather than star-shaped layout. Interpolate signal and time for 1km grid spacing. 
#         Rbf interpolation will be used for signal, and 
#         code used in Radio Morphing will be used for time interpolation.
#     
#     1. GRAND_ReCon_archive1.ipynb                  (archived on April, 2022) 
#         Reason: 
#         Uses ZHAireS simulations by Matias. Only 1500 sims. Low simulation data for ML.
#         Started to use PengXiong's simulations:- 32k H, 26k Fe, 26k Gamma rays.
# 
#     0 . ReCon_IceCube_YuWang.ipynb                 (archived on March, 2022)
#         Note: First template code given to us by Yu Wang.

# In[1]:


print("\nImporting necessary packages.")

name    = "xmax" #"fourier"
savefig = True

from time import time
start = time()

import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import LogNorm
from matplotlib import cm

params = {'legend.fontsize': 12,
          'axes.labelsize' : 22,
          'axes.titlesize' : 23,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'figure.figsize' : (8, 6),
          #'axes.grid'      : True
          }
plt.rcParams.update(params)

print("\nImporting necessary functions:")
import interpolation_fourier as interpF
from interpolation_time import ComputeTimeAntennas
from misc import *
from build_cnn import *


fmin = 50.e6  # minimum radio frequency in hertz.
fmax = 200.e6 # maximum radio frequency in hertz.
time_bins = np.logspace(-2,6,201)
p2p_bins  = np.logspace(-1,6,141)

xmax_coord_bins= np.linspace(-1e5,1e5,1000)
xmax_bins      = np.linspace(0,1000,101)
diff_xmaxRbins = np.linspace(0,5e4,101)
diff_xmaxbins  = np.linspace(-300,300,101)

# Grid antennae position at grid layout for CNN. 357 total.
X, Y   = np.meshgrid(np.linspace(-10e3, 10e3, 21), np.linspace(-8e3, 8e3, 17))
antx   = X.flatten()
anty   = Y.flatten()
antz   = 2100*np.ones(len(antx)) # 2100m is the altitude used in simulations.
antpos = np.column_stack((antx, anty, antz))

print("\nLoading data.")
unshuffled_label_collection   = np.load('label_grid_1km_%s.npy'%name)
unshuffled_feature_collection = np.load('feature_grid_1km_%s.npy'%name)
print('Loaded labels and features from saved files. \n')

if savefig:
    plt.figure(figsize=(8,6))
    a,b,c,d = plt.hist2d(unshuffled_feature_collection[:,0,...].flatten(), 
                         unshuffled_feature_collection[:,1,...].flatten(), 
                         bins=[time_bins, p2p_bins], norm=LogNorm(), cmap=cm.Spectral_r)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(ls='--', alpha=0.3)
    plt.colorbar(label=r'N$_{hit}$')
    plt.xlabel('Time [a.u]')
    plt.ylabel(r'P2P [$\mu$V/m]')
    plt.xlim(1e-1, 1e5)
    plt.ylim(1, 2e5)
    plt.savefig('Plots/t0_vs_p2p_2d_%s.png'%name, bbox_inches='tight')
    plt.close()

    # ---------
    h0, b0 = np.histogram(unshuffled_feature_collection[:,0,...], bins=time_bins)
    h1, b1 = np.histogram(unshuffled_feature_collection[:,1,...], bins=p2p_bins)

    grid = gs.GridSpec(10, 20)

    fig = plt.figure(figsize=(15,6))
    fig.clf()

    # Add axes which can span multiple grid boxes
    ax1 = fig.add_subplot(grid[:, :9])
    ax2 = fig.add_subplot(grid[:, 11:])

    ax1.plot(b0[:-1], h0, 'k')
    ax2.plot(b1[:-1], h1, 'k')
    ax1.set_xlabel('Time [a.u]')
    ax1.set_ylabel('Number')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xlabel(r'P2P [$\mu$V/m]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax1.set_xlim(1e-1, 1e5)
    ax2.set_xlim(1, 2e5)

    ax1.grid(ls='--', alpha=0.3)
    ax2.grid(ls='--', alpha=0.3)
    plt.savefig('Plots/t0_and_p2p_hist_%s.png'%name, bbox_inches='tight')
    plt.close()

    print("Shuffle features and labels. Also add x and y coordinates of antennae in grid layout.")

# Get random integer to use it as index to shuffle label and features.
shuffle_index = np.arange(len(unshuffled_label_collection))
np.random.shuffle(shuffle_index)

# shuffle label and features
label_collection  = unshuffled_label_collection[shuffle_index]
feature_collection= unshuffled_feature_collection[shuffle_index]

# add antennae position (x and y) as features. Position feature is added here
# because the position of antennae does not change with events. No hit antennae
# position for each events are replaced by 0.
f_shape    = list(feature_collection.shape)
f_shape[1] = 1                    # to add a new features.
f_xhits    = np.ones(f_shape)*X.T # repeat same antennae positions for all events
f_yhits    = np.ones(f_shape)*Y.T # repeat same antennae positions for all events
feature_collection = np.append(feature_collection, f_xhits, axis=1) # append xhits in feature collections.
feature_collection = np.append(feature_collection, f_yhits, axis=1) # append yhits in feature collections.
print("features:", feature_collection.shape, "labels:", label_collection.shape, '\n')

print("Run SRTCleaning on all events and remove events whose conditions are not passed.")

delT = np.zeros(feature_collection[:,0,...].shape)
P2P  = np.zeros(feature_collection[:,1,...].shape)
hitX = np.zeros(feature_collection[:,2,...].shape)
hitY = np.zeros(feature_collection[:,3,...].shape)
xmax_x = label_collection[:,0]
xmax_y = label_collection[:,1]
xmax_z = label_collection[:,2]
xmax = label_collection[:,3]

keep_events = []      # remove events that does not trigger at least 5 antennae.
ngridy      = feature_collection.shape[-1]

# Run loop for all events and perform SRTCleaning. Delete events that does not trigger at least 5 antennae.
for i in range(feature_collection.shape[0]):
    # prepare data to run SRTCleaning.
    t = feature_collection[i,0].flatten()
    p = feature_collection[i,1].flatten()
    x = feature_collection[i,2].flatten()
    y = feature_collection[i,3].flatten()

    # Run SRTCleaning on hits to get rid of background hits.
    # Index (based on flatten) of hits that pass SRT cleaning conditions is returned.
    clusterIndex = SRTCleaning(x, y, t, p)
    
    # Replace values only for hits that pass SRT cleaning conditions and has 5 or more hits.
    if len(clusterIndex)>=5: #make sure there are atleast 5 antennae hits.
        # finding index for (21,17) shape.
        indx = np.int_(np.floor(clusterIndex/ngridy))
        indy = np.int_(clusterIndex - indx*ngridy)
        delT[i, indx, indy] = t[clusterIndex]
        P2P[i, indx, indy]  = p[clusterIndex]
        hitX[i, indx, indy] = x[clusterIndex]
        hitY[i, indx, indy] = y[clusterIndex]
        
        keep_events.append(True)  # track events which has at least 5 hits.
    else:
        keep_events.append(False) # track events which has at least 5 hits.
        
# Remove events that does not trigger at least 5 antennae.
delT = delT[keep_events]
P2P  = P2P[keep_events]
hitX = hitX[keep_events]
hitY = hitY[keep_events]
xmax_x = xmax_x[keep_events]
xmax_y = xmax_y[keep_events]
xmax_z = xmax_z[keep_events]
xmax   = xmax[keep_events]

delTNorm, delTMinimum, delTMaximum, delTOffset = minmax(np.log10(delT+1), offset=0.001)
p2pNorm, p2pMinimum, p2pMaximum, p2pOffset     = minmax(np.log10(P2P+1), offset=0.001)
hitxNorm, hitxMinimum, hitxMaximum, hitxOffset = minmax(hitX, offset=0.001)
hityNorm, hityMinimum, hityMaximum, hityOffset = minmax(hitY, offset=0.001)

#xmaxNorm, xmaxMinimum, xmaxMaximum, xmaxOffset         = minmax(xmax, offset=0.001)

features_stacked = np.column_stack((delTNorm[:,np.newaxis, ...], 
                             p2pNorm[:,np.newaxis, ...], 
                             hitxNorm[:,np.newaxis, ...], 
                             hityNorm[:,np.newaxis, ...]))

labels_stacked   = np.column_stack((xmax_x, xmax_y, xmax_z, xmax))

print(features_stacked.shape, labels_stacked.shape)


# Convert from numpy.ndarray to torch tensor.
x1 = torch.from_numpy(features_stacked)
y1 = torch.from_numpy(labels_stacked)
#y1 = torch.from_numpy(xmaxNorm[:,np.newaxis])
# Float dtype is necessary for matmul of kernel and image.
x1 = x1.float()    # features
y1 = y1.float()    # labels
print('features:', x1.shape, '  labels:', y1.shape, " \n")

#net = resnet_block(in_channels=9, out_channels=2, num_residuals=3, dimension='3D')
net       = ReCon()
optimizer = optim.AdamW(net.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False)

# # Tranining
use_gpu = torch.cuda.is_available()
if use_gpu:
    # start with 0. If higher number is give "RuntimeError: CUDA error: invalid device ordinal" is shown.
    net.cuda(0)
    device = torch.device('cuda:0')
    print ('USE GPU')
else:
    device = torch.device('cpu')
    print ('USE CPU')


ntrain     = int(0.8*x1.shape[0]) # use 80% of events for training and 20% for testing.
train_iter = DataLoader(TensorDataset(x1[:ntrain], y1[:ntrain]), batch_size=500, shuffle=True, num_workers=0)
test_iter  = DataLoader(TensorDataset(x1[ntrain:], y1[ntrain:]), batch_size=500, shuffle=True, num_workers=0)

loss_list  = []

num_epochs = 20000
for epoch in range(num_epochs):
    net.train() 
    train_l_sum = 0.
    for img0, label in train_iter:
        if use_gpu:
            img0, label =  img0.to(device), label.to(device)
        optimizer.zero_grad()          ## Zero weights before calculating gradients
        pred, predErr = net(img0)
        l = loss(pred, predErr, label) ## Calculate Loss
        l.backward()                   ## Calculate Gradients
        optimizer.step()               ## Update Weights
        train_l_sum += l.item()
    
    loss_list.append(train_l_sum / len(train_iter))
    if (epoch%100 == 0):
        net.eval() 
        print('Epoch:',epoch, 'Loss:',train_l_sum / len(train_iter))
        
print('Epoch:',epoch, 'Loss:',train_l_sum / len(train_iter))

if savefig:
    print('Plot loss.')

    plt.figure(figsize=(8,6))
    plt.plot(loss_list, color='k')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    #plt.xlim(0,500)
    plt.grid(ls='--', alpha=0.3)
    plt.savefig('Plots/loss_%s.png'%name)
    plt.close()

net.eval()

test_l_sum = 0.
pred_sum = 0

for img0, label in test_iter:
    if use_gpu:
        img0, label =  img0.to(device), label.to(device)
    pred, predErr = net(img0)
    l = loss(pred, predErr, label)
    test_l_sum += l.item()
    
print('\n Loss:',test_l_sum / len(test_iter) )

relErr = ((pred-label)/(label+1))*100   # +1 because some label is 0 and produces inf.

if savefig:
    print('\n Plot Xmax relative error.')

    plt.figure()
    plt.hist(relErr.detach().cpu().numpy()[:,0], histtype='step', label=r'X$_{max}$_X', color='k', alpha=1, bins=51, lw=1.5)
    plt.hist(relErr.detach().cpu().numpy()[:,1], histtype='step', label=r'X$_{max}$_Y', color='k', alpha=0.7, bins=51, lw=1.5)
    plt.hist(relErr.detach().cpu().numpy()[:,2], histtype='step', label=r'X$_{max}$_Z', color='k', alpha=0.4, bins=51, lw=1.5)
    plt.yscale('log')
    plt.grid(ls='--', alpha=0.3)
    plt.legend()
    plt.ylabel('Number')
    plt.title('Relative Error [%]', fontsize=16)
    plt.savefig('Plots/relative_error_coord_%s.png'%name, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist(relErr.detach().cpu().numpy()[:,3], label=r'X$_{max}$', color='m', alpha=0.6, bins=51, lw=1.5)
    plt.yscale('log')
    plt.grid(ls='--', alpha=0.3)
    plt.legend()
    plt.ylabel('Number')
    plt.title('Relative Error [%]', fontsize=16)
    plt.savefig('Plots/relative_error_%s.png'%name, bbox_inches='tight')
    plt.close()

    print('\n Plot true - predicted Xmax.')

    hist_xmaxR = 0
    hist_xmax  = 0
    for img0, label in test_iter:
        if use_gpu:
            img0, label =  img0.to(device), label.to(device)
        pred, predErr = net(img0)
        rx =  pred.detach().cpu().numpy()[:,0]-label.detach().cpu().numpy()[:,0]
        ry =  pred.detach().cpu().numpy()[:,1]-label.detach().cpu().numpy()[:,1]
        rz =  pred.detach().cpu().numpy()[:,2]-label.detach().cpu().numpy()[:,2]
        hist_xmaxR += np.histogram(np.sqrt(rx**2 + ry**2 + rz**2), bins=diff_xmaxRbins)[0]
        hist_xmax  += np.histogram(pred.detach().cpu().numpy()[:,3]-label.detach().cpu().numpy()[:,3], bins=diff_xmaxbins)[0]

    plt.figure()
    plt.plot(diff_xmaxbins[:-1], hist_xmax, color='k')
    plt.legend()
    plt.grid(ls='--', alpha=0.3)
    plt.xlabel('Predicted - True [g/cm$^2$]')
    plt.ylabel('Number')
    plt.title(r'X$_{max}$')
    plt.savefig('Plots/xmax_diff_%s.png'%name, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(diff_xmaxRbins[:-1]/1000., hist_xmaxR, color='m')
    plt.legend()
    plt.grid(ls='--', alpha=0.3)
    plt.xlabel('Predicted - True [km]')
    plt.ylabel('Number')
    plt.title(r'Distance between True and Reco X$_{max}$')
    plt.savefig('Plots/xmax_diff_dist_%s.png'%name, bbox_inches='tight')
    plt.close()

    diffX = label.detach().cpu().numpy()[:,3]-pred.detach().cpu().numpy()[:,3]
    print("standard deviation: ", np.std(diffX))
    print(np.column_stack((label.detach().cpu().numpy()[:,3], pred.detach().cpu().numpy()[:,3])))


    # Plot histogram of true and predicted value.
    Thist_xmaxX, Thist_xmaxY, Thist_xmaxZ, Thist_xmax = 0,0,0,0
    Phist_xmaxX, Phist_xmaxY, Phist_xmaxZ, Phist_xmax = 0,0,0,0
    for img0, label in test_iter:
        if use_gpu:
            img0, label =  img0.to(device), label.to(device)
        pred, predErr = net(img0)
        Thist_xmaxX += np.histogram(label.detach().cpu().numpy()[:,0], bins=xmax_coord_bins)[0]
        Thist_xmaxY += np.histogram(label.detach().cpu().numpy()[:,1], bins=xmax_coord_bins)[0]
        Thist_xmaxZ += np.histogram(label.detach().cpu().numpy()[:,2], bins=xmax_coord_bins)[0]
        Phist_xmaxX += np.histogram(pred.detach().cpu().numpy()[:,0], bins=xmax_coord_bins)[0]
        Phist_xmaxY += np.histogram(pred.detach().cpu().numpy()[:,1], bins=xmax_coord_bins)[0]
        Phist_xmaxZ += np.histogram(pred.detach().cpu().numpy()[:,2], bins=xmax_coord_bins)[0]

        Thist_xmax  += np.histogram(label.detach().cpu().numpy()[:,3], bins=xmax_bins)[0]
        Phist_xmax  += np.histogram(pred.detach().cpu().numpy()[:,3], bins=xmax_bins)[0]

    plt.figure()
    plt.plot(xmax_coord_bins[:-1], Thist_xmaxX, color='k', label=r'X$_{max}$_X True')
    plt.plot(xmax_coord_bins[:-1], Phist_xmaxX, color='k', ls='--', lw=1.5)
    plt.plot(xmax_coord_bins[:-1], Thist_xmaxY, color='k', alpha=0.7, label=r'X$_{max}$_Y True')
    plt.plot(xmax_coord_bins[:-1], Phist_xmaxY, color='k', alpha=0.7, ls='--', lw=1.5)
    plt.plot(xmax_coord_bins[:-1], Thist_xmaxZ, color='k', alpha=0.4, label=r'X$_{max}$_Z True')
    plt.plot(xmax_coord_bins[:-1], Phist_xmaxZ, color='k', alpha=0.4, ls='--', lw=1.5)
    plt.legend()
    plt.grid(ls='--', alpha=0.3)
    plt.xlabel(r'$X_{max}$ [a.u]')
    plt.ylabel('Number')
    plt.title("Comparison between True [solid] and Predicted [dashed]")
    plt.savefig('Plots/xmax_comparison_%s.png'%name, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(xmax_bins[:-1], Thist_xmax, label=r'X$_{max}$ T', color='m')
    plt.plot(xmax_bins[:-1], Phist_xmax, label=r'X$_{max}$ P', color='m', ls='--', lw=1.5)
    plt.legend()
    plt.grid(ls='--', alpha=0.3)
    plt.xlabel(r'$X_{max}$ [a.u]')
    plt.ylabel('Number')
    plt.savefig('Plots/xmax_comparison_%s.png'%name, bbox_inches='tight')
    plt.close()


end = time()

time_taken = end-start
mins       = int(time_taken/60)
secs       = time_taken%60


print('\n Time taken: %i mins, %.2f secs'%(mins, secs))

