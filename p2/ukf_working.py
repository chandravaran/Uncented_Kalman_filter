from matplotlib.pyplot import axis
import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    # imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]
    T = 5500
    # T = 4600
    # T = 3400

    # your code goes here
    vicon_pose = np.zeros((3, T))
    est_pose = np.zeros((3, T))
    est_pose_cov = np.zeros((3, T))
    gyro_readings = np.zeros((3, T))
    ang_vels = np.zeros((3, T))
    ang_vels_cov = np.zeros((3, T))

    # accelerometer and gyroscope parameters
    acc_sen = 33.86
    acc_trans = np.array([[-1, 0, 0],[0, -1, 0],[0, 0, 1]]) * 1023.0 * acc_sen / 3300.0
    acc_bias = np.array([511, 501, 503])
    gyro_sen = 193.55
    gyro_trans = np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]]) * 1023.0 * gyro_sen / 3300.0
    gyro_bias = np.array([369.5, 371.5, 377])

    # mean and covariance of the state
    x_mean = [Quaternion(), np.array([0.0, 0.0, 0.0])]
    x_cov = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # process and measurement noise
    # process_cov = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    # measure_cov = np.diag([2.5, 2.5, 2.5, 3.0, 3.0, 3.0])
    process_cov = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    measure_cov = np.diag([1.5, 1.5, 1.5, 4.0, 4.0, 4.0])
    
    for t in range(T):
        if t == 0:
            dt = 0.01
        else:
            dt = imu['ts'][0, t] - imu['ts'][0, t - 1]
        
        # add noise to state variance and do Cholesky decomposition
        x_cov += process_cov * dt
        S = np.linalg.cholesky(x_cov).T
        n = x_cov.shape[0]
        W = np.concatenate((np.sqrt(2 * n) * S, -np.sqrt(2 * n) * S), axis=1)

        # generate sigma points
        X = []
        for i in range(2 * n):
            w = W[:, i]
            w_q = Quaternion()
            w_q.from_axis_angle(w[:3])
            q = x_mean[0] * w_q
            X.append([q, w[3:] + x_mean[1]])
        
        # transform sigma points with process model
        Y = []
        for i in range(2 * n):
            sigma = X[i]
            dq = Quaternion()
            dq.from_axis_angle(sigma[1] * dt)
            Y.append([sigma[0] * dq, sigma[1]])
        
        # priori estimate
        es = np.zeros((3, 2 * n))
        e_mean = np.ones(3)

        while np.linalg.norm(e_mean) > 5e-2:
            for i in range(2 * n):
                sigma = Y[i]
                e = sigma[0] * x_mean[0].inv()
                es[:, i] = e.axis_angle()
            e_mean = np.mean(es, axis=1)
            e_mean_q = Quaternion()
            e_mean_q.from_axis_angle(e_mean)
            x_mean[0] = e_mean_q * x_mean[0]

        omegas = [Y[i][1] for i in range(2 * n)]
        omegas = np.array(omegas).T
        x_mean[1] = np.mean(omegas, axis=1)

        W_prime = np.concatenate((es, omegas - x_mean[1][:, np.newaxis]), axis=0)
        x_cov = 1.0 / (2 * n) * W_prime @ W_prime.T

        # regenerate sigma points
        S = np.linalg.cholesky(x_cov).T
        W = np.concatenate((np.sqrt(2 * n) * S, -np.sqrt(2 * n) * S), axis=1)
        X = []
        for i in range(2 * n):
            w = W[:, i]
            q_w = Quaternion()
            q_w.from_axis_angle(w[:3])
            q = x_mean[0] * q_w
            X.append([q, w[3:] + x_mean[1]])

        # transform sigma points with measurement model
        Z = np.zeros((6, 2 * n))
        for i in range(2 * n):
            q_z, w_z = X[i][0], X[i][1]
            
            acc_q = q_z.inv() * Quaternion(0, [0, 0, 9.81]) * q_z
            acc_vec = acc_q.vec()
            acc_vec = acc_trans @ acc_vec
            acc_vec += acc_bias
            Z[:3, i] = acc_vec

            gyro_vec = gyro_trans @ w_z
            gyro_vec += gyro_bias
            Z[3:, i] = gyro_vec

        # measurement estimate
        z_mean = np.mean(Z, axis=1)
        Z_prime = Z - z_mean[:, np.newaxis]
        P_zz = 1.0 / (2 * n) * Z_prime @ Z_prime.T
        P_vv = P_zz + measure_cov
        P_xz = 1.0 / (2 * n) * W_prime @ Z_prime.T

        # innovation
        measurements = np.concatenate((accel[:, t], gyro[:, t]), axis=0)
        innovation = measurements - z_mean

        # Kalman gain
        K = P_xz @ np.linalg.inv(P_vv)

        # posterior estimate
        new_x = K @ innovation[:, np.newaxis]
        new_q = Quaternion()
        new_q.from_axis_angle(new_x[:3, 0])
        x_mean[0] = new_q * x_mean[0]
        x_mean[1] = x_mean[1] + new_x[3:, 0]
        x_cov -= K @ P_vv @ K.T

        # save variables
        est_pose[:, t] = x_mean[0].euler_angles()
        est_pose_cov[:, t] = np.sqrt(np.diag(x_cov)[:3])
        ang_vels[:, t] = x_mean[1]
        ang_vels_cov[:, t] = np.sqrt(np.diag(x_cov)[3:])

        gyro_readings[:, t] = np.linalg.inv(gyro_trans) @ (gyro[:, t] - gyro_bias)

        R = vicon['rots'][:, :, t]
        q = Quaternion()
        q.from_rotm(R)
        vicon_pose[:, t] = q.euler_angles()

        # print(t, est_pose[:, t])

    
    # visualize
    plt.figure(1)
    plt.subplot(231)
    plt.plot(vicon['ts'][0, 0:T], vicon_pose[0, :], label='vicon')
    plt.plot(imu['ts'][0, 0:T], est_pose[0, :], label='estimation')
    plt.gca().fill_between(imu['ts'][0, 0:T], est_pose[0, :] - est_pose_cov[0, :], est_pose[0, :] + est_pose_cov[0, :], color="#dddddd")
    plt.xlabel('t (s)')
    plt.ylabel('q (rad)')
    plt.legend()
    
    plt.subplot(232)
    plt.plot(vicon['ts'][0, 0:T], vicon_pose[1, :], label='vicon')
    plt.plot(imu['ts'][0, 0:T], est_pose[1, :], label='estimation')
    plt.gca().fill_between(imu['ts'][0, 0:T], est_pose[1, :] - est_pose_cov[1, :], est_pose[1, :] + est_pose_cov[1, :], color="#dddddd")
    plt.xlabel('t (s)')
    plt.ylabel('q (rad)')
    plt.legend()

    plt.subplot(233)
    plt.plot(vicon['ts'][0, 0:T], vicon_pose[2, :], label='vicon')
    plt.plot(imu['ts'][0, 0:T], est_pose[2, :], label='estimation')
    plt.gca().fill_between(imu['ts'][0, 0:T], est_pose[2, :] - est_pose_cov[2, :], est_pose[2, :] + est_pose_cov[2, :], color="#dddddd")
    plt.xlabel('t (s)')
    plt.ylabel('q (rad)')
    plt.legend()

    plt.subplot(234)
    plt.plot(imu['ts'][0, 0:T], gyro_readings[0, :], label='gyroscope')
    plt.plot(imu['ts'][0, 0:T], ang_vels[0, :], label='estimation')
    plt.gca().fill_between(imu['ts'][0, 0:T], ang_vels[0, :] - ang_vels_cov[0, :], ang_vels[0, :] + ang_vels_cov[0, :], color="#dddddd")
    plt.xlabel('t (s)')
    plt.ylabel('w (rad/s)')
    plt.legend()

    plt.subplot(235)
    plt.plot(imu['ts'][0, 0:T], gyro_readings[1, :], label='gyroscope')
    plt.plot(imu['ts'][0, 0:T], ang_vels[1, :], label='estimation')
    plt.gca().fill_between(imu['ts'][0, 0:T], ang_vels[1, :] - ang_vels_cov[1, :], ang_vels[1, :] + ang_vels_cov[1, :], color="#dddddd")
    plt.xlabel('t (s)')
    plt.ylabel('w (rad/s)')
    plt.legend()

    plt.subplot(236)
    plt.plot(imu['ts'][0, 0:T], gyro_readings[2, :], label='gyroscope')
    plt.plot(imu['ts'][0, 0:T], ang_vels[2, :], label='estimation')
    plt.gca().fill_between(imu['ts'][0, 0:T], ang_vels[2, :] - ang_vels_cov[2, :], ang_vels[2, :] + ang_vels_cov[2, :], color="#dddddd")
    plt.xlabel('t (s)')
    plt.ylabel('w (rad/s)')
    plt.legend()

    plt.show()
    

    roll = est_pose[0, :]
    pitch = est_pose[1, :]
    yaw = est_pose[2, :]
    # roll, pitch, yaw are numpy arrays of length T
    return roll, pitch, yaw

roll, pitch, yaw = estimate_rot(1)

# print(roll.shape, pitch.shape, yaw.shape)
