# nonlinear least-squares calibration routine

import torch
from torch.autograd import Variable

def calibrate(x, y, alpha_ref, beta_ref, learning_rate=1e-4, epochs=4000, r=1):

    dtype = torch.FloatTensor

    x = torch.from_numpy(x).type(dtype)
    y = torch.from_numpy(y).type(dtype)

    # clip x, y to shortest one
    x = x[:,:min(x.shape[1], y.shape[1])]
    y = y[:,:min(x.shape[1], y.shape[1])]

    alpha_ref = torch.from_numpy(alpha_ref)
    beta_ref = torch.from_numpy(beta_ref)

    alpha = Variable(alpha_ref.type(dtype), requires_grad=True)
    beta = Variable(beta_ref.type(dtype), requires_grad=True)

    # calibrate alpha, beta using backprop
    
    for t in range(epochs):
        
        # forward pass
        y_pred = (x - beta) * (3300.0) / (1023.0 * alpha)
        
        # compute loss
        loss = (y_pred - y).pow(2).sum() + r * ((alpha - alpha_ref).pow(2).sum() + (beta - beta_ref).pow(2).sum())
        
        # backprop
        loss.backward()
        
        # update parameters
        alpha.data -= learning_rate * alpha.grad.data
        beta.data -= learning_rate * beta.grad.data
        
        # zero gradients
        alpha.grad.data.zero_()
        beta.grad.data.zero_()
            
        if t % 500 == 0:
            print("t: ", t, " loss: ", loss.data)
        
    return alpha, beta

beta_ref = np.array([-500,-500,500]).reshape(3,1)
alpha_ref = np.array([25,25,25]).reshape(3,1)
alpha_accel, beta_accel = calibrate(accel, vicon_orientations, alpha_ref, beta_ref)
print("alpha accel: ", alpha_accel.data)
print("beta accel: ", beta_accel.data) 

'''
beta_ref = np.array([375,375,375]).reshape(3,1)
alpha_ref = np.array([250,250,250]).reshape(3,1)
alpha_gyro, beta_gyro = calibrate(gyro, vicon_omegas, alpha_ref, beta_ref, r=1)
print("alpha gyro: ", alpha_gyro.data)
print("beta gyro: ", beta_gyro.data)'''