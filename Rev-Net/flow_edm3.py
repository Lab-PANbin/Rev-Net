import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

#from datasets import get_CIFAR10, get_SVHN
from model5 import FlowNet, UN

import sys
import scipy as sp
import scipy.linalg as splin
import scipy.io
import numpy as np
from scipy.stats import multivariate_normal as mvnorm


import numpy as np
import matplotlib.pyplot as plt 

import xlrd
import xlwt

#############################################
# Internal functions
#############################################

def excel_to_matrix(path):
  table = xlrd.open_workbook(path).sheets()[0]
  row = table.nrows
  col = table.ncols
  datamatrix = np.zeros((row-1, col-1))
  for x in range(1,col):
    cols = table.col_values(x)
    datamatrix[:, x-1] = cols[1:]
  return datamatrix
edm=excel_to_matrix('syn3SV_edm_3_30.xls')# (N,P)
edm_vca=torch.from_numpy(edm).float()
#im=im.reshape(-1, L).T
#edm_vca,indice,Yp=vca(im,P)
#edm_vca=torch.from_numpy(edm_vca).float()
edm_vca.requires_grad=False

for k in range(10):

  L=156
  P=3
  im = scipy.io.loadmat('syn3SV_data3cor30.mat') #height 100 width 100 bands 198
  im=im['Y']
  #im=im.reshape(-1, L).T
  scipy.io.savemat('edm3_vca1.mat', {'M':edm_vca})


  im = scipy.io.loadmat('syn3SV_data3cor30.mat') #height 100 width 100 bands 198
  im=im['Y']
  B=np.shape(im)[0]
  height=int(np.sqrt(np.shape(im)[1]))
  weight=int(np.sqrt(np.shape(im)[1]))
  
  #print(np.shape(im))



  im=im.T

  plt.imshow(im.reshape(weight ,height,B)[:,:,[90,60,30]]) 
  plt.savefig("pic/edm3.png")
  im=torch.from_numpy(im).float().reshape(-1,B).unsqueeze(dim=2).unsqueeze(dim=3).cuda()
  rdn1=torch.randn(weight*height,B,1,1)
  rdn1=rdn1.cuda()
  rdn2=torch.randn(weight*height,B,1,1)
  rdn2=rdn2.cuda()
  rdn3=torch.randn(weight*height,B,1,1)
  rdn3=rdn3.cuda()
  mu=np.zeros(B)
  p1=mvnorm.pdf(rdn1.reshape(weight*height,B).cpu().detach().numpy(),mean=mu)
  p2=mvnorm.pdf(rdn2.reshape(weight*height,B).cpu().detach().numpy(),mean=mu)
  p3=mvnorm.pdf(rdn3.reshape(weight*height,B).cpu().detach().numpy(),mean=mu)

  hidden_channels=4
  #L=5
  K=50
  P=3
  lr=3e-4
  image_shape=[height*weight,B,1,1]
  edm_shape=[1,1,B]
  actnorm_scale=1.2
  flow_permutation='invconv'
  flow_coupling="affine"
  LU_decomposed=False
  model = UN(image_shape,edm_shape,hidden_channels,K,P,actnorm_scale,flow_permutation,flow_coupling,LU_decomposed)
  model = model.cuda()

  optimizer = optim.Adam(model.parameters(), lr=lr)

  def pre_loss_function(x, recon_x, abd, Mn, logdet1, logdet2, logdet3):
      """
      :param recon_x: generated image
      :param x: original image
      :param mu: latent mean of z
      :param logvar: latent log variance of z
      """
      reconstruction_loss1 = ((recon_x - x) ** 2).sum() / x.shape[0]
      base_loss=torch.mean(torch.acos(torch.sum(recon_x.reshape(-1,B) * im.reshape(-1,B), dim=-1) /(torch.norm(recon_x.reshape(-1,B), dim=-1, p=2)*torch.norm(im.reshape(-1,B), dim=-1, p=2))))
      abd_loss= torch.mean(torch.sum(torch.sqrt(torch.abs(abd)), dim=-1))
      #edm_loss
      L_VCA = torch.pow(torch.norm(Mn - edm_vca.cuda().unsqueeze(dim=0)),2)/x.shape[0]/P/L

      #logdet_loss=sum(logdet)
      pm1=np.exp(np.log(p1)-logdet1.cpu().detach().numpy())
      pm2=np.exp(np.log(p2)-logdet2.cpu().detach().numpy())
      pm3=np.exp(np.log(p3)-logdet3.cpu().detach().numpy())

      pm1=torch.from_numpy(pm1).float().cuda()
      pm2=torch.from_numpy(pm2).float().cuda()
      pm3=torch.from_numpy(pm3).float().cuda()

      a=torch.cat((pm1.reshape(-1,1),pm2.reshape(-1,1),pm3.reshape(-1,1)),dim=1)
      #print('a:',torch.sum(a!=a))
      logdet_loss=1/((abd.squeeze(dim=2) * a).mean()/L+1e-5)
      if logdet_loss<1e-40:
        logdet_loss=0
      #else:
      #  logdet_loss=-(abd.squeeze(dim=2) * a).mean()
      print(logdet_loss)
      #print(logdet_loss)
      

      #return logdet_loss
      return base_loss + 1e-2*abd_loss + 100*L_VCA + 1e-10*logdet_loss#+ 1e-10*logdet_loss#+ 3*loss_sad

  def loss_function(x, recon_x, abd, Mn, logdet1, logdet2, logdet3):
      """
      :param recon_x: generated image
      :param x: original image
      :param mu: latent mean of z
      :param logvar: latent log variance of z
      """
      base_loss=torch.mean(torch.acos(torch.sum(recon_x.reshape(-1,B) * im.reshape(-1,B), dim=-1) /(torch.norm(recon_x.reshape(-1,B), dim=-1, p=2)*torch.norm(im.reshape(-1,B), dim=-1, p=2))))
      reconstruction_loss1 = ((recon_x - x) ** 2).sum() / x.shape[0]
      abd_loss=torch.mean(torch.sum(torch.sqrt(torch.abs(abd)), dim=-1))
      #edm_loss
      M_l=Mn.permute(0,2,1)
      em_bar = M_l.mean(dim=0, keepdim=True)  # [1,5,198] [1,z_dim,Channel]
      aa = (M_l * em_bar).sum(dim=2)
      em_bar_norm = em_bar.square().sum(dim=2).sqrt()
      em_tensor_norm = M_l.square().sum(dim=2).sqrt()
      em_bar = M_l.mean(dim=1, keepdim=True)
      loss_minvol = ((M_l - em_bar) ** 2).sum() / x.shape[0] / P / L
      #logdet_loss=sum(logdet)
      pm1=np.exp(np.log(p1)-logdet1.cpu().detach().numpy())
      pm2=np.exp(np.log(p2)-logdet2.cpu().detach().numpy())
      pm3=np.exp(np.log(p3)-logdet3.cpu().detach().numpy())

      pm1=torch.from_numpy(pm1).float().cuda()
      pm2=torch.from_numpy(pm2).float().cuda()
      pm3=torch.from_numpy(pm3).float().cuda()

      a=torch.cat((pm1.reshape(-1,1),pm2.reshape(-1,1),pm3.reshape(-1,1)),dim=1)
      #print('a:',torch.sum(a!=a))
      logdet_loss=1/((abd.squeeze(dim=2) * a).mean()/L+1e-5)
      if logdet_loss<1e-40:
        logdet_loss=0
      
      return base_loss + 1e-2*abd_loss + 10*loss_minvol + logdet_loss #+ 3*loss_sad


  epochs=5001

  model = model.cuda()
  criterion=torch.nn.MSELoss()
  for epoch in range(epochs):
      optimizer.zero_grad()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=1)  #梯度裁剪
      abd, edm, rec, logdet1, logdet2, logdet3 = model(im,rdn1,rdn2,rdn3)
      if (epoch+1)%200==0:
        for i, dic in enumerate(optimizer.param_groups): #opt.param_groups 输出是二维列表
          dic['lr'] *= 0.9  #前面定义的优化器opt 将学习率缩小
      if epoch < 3000:
          losses=pre_loss_function(im.reshape(-1,B),rec.squeeze(dim=2),abd,edm,logdet1, logdet2, logdet3)
      else:
          losses=loss_function(im.reshape(-1,B),rec.squeeze(dim=2),abd,edm, logdet1, logdet2, logdet3)
      losses.backward()
      print(losses)
      optimizer.step()
      if (epoch+1) in [3000,3500,4000,4500,5000]:
        code=abd.cpu()
        code=code.data.numpy()
        scipy.io.savemat('flow_edm3_abd2_%d_%d.mat'%(epoch+1,k+1), {'A':code})
        #print(code)
        m_hat=edm.cpu()
        m_hat=m_hat.data.numpy().mean(0)
        scipy.io.savemat('flow_edm3_edm2_%d_%d.mat'%(epoch+1,k+1), {'M':m_hat})
        abd=abd.reshape(height,weight,P).cpu()
        for i in range(P):
            plt.subplot(1,P,i+1)
            plt.imshow(abd[:,:,i].detach().numpy(),cmap='jet') #丰度矩阵
            plt.savefig("pic/edm3_abd2_%d.png"%(epoch+1))

  #print(logit)
  print('logdet1.shape:',logdet1.shape)
  print('logdet2.shape:',logdet2.shape)
  print('logdet3.shape:',logdet3.shape)
  #print(abd.shape)
  #print(edm.shape)
  #print(edm_vca.shape)
  print(np.shape(rec))
  rec_plot=rec.reshape(height, weight, B)
  print(np.shape(rec_plot))
  import matplotlib.pyplot as plt 

  mu=np.zeros(B)
  p1=mvnorm.pdf(rdn1.reshape(weight*height,B).cpu().detach().numpy(),mean=mu)
  p1=p1*np.exp(-logdet1.cpu().detach().numpy())
  p2=mvnorm.pdf(rdn2.reshape(weight*height,B).cpu().detach().numpy(),mean=mu)
  p2=p2*np.exp(-logdet2.cpu().detach().numpy())
  p3=mvnorm.pdf(rdn3.reshape(weight*height,B).cpu().detach().numpy(),mean=mu)
  p3=p3*np.exp(-logdet3.cpu().detach().numpy())
  #print(p1/(p1+p2+p3))
  #print(p2/(p1+p2+p3))
  #print(p3/(p1+p2+p3))

  code=abd.cpu()
  code=code.data.numpy()
  scipy.io.savemat('flow_edm3_abd2.mat', {'A':code})
  #print(code)
  m_hat=edm.cpu()
  m_hat=m_hat.data.numpy()
  scipy.io.savemat('flow_edm3_edm2.mat', {'M':m_hat})







