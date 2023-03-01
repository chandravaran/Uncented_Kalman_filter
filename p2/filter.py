import numpy as np
import scipy.linalg as la
from quaternion import Quaternion
from state import FilterState

class KalmanFilter:
    ''' Kalman filter class. '''

    def __init__(self, Q, R, g, x0):
        self.Q = Q
        self.R = R
        # print(self.R.shape)
        self.g = g
        self.x = x0

class UncentedKalmanFilter(KalmanFilter):
    ''' Unscented Kalman filter class. '''

    def __init__(self, Q, R, g, x0, sigma0):
        super().__init__(Q, R, g, x0)

        self.mu = x0
        self.sigma = sigma0 # should be the intial sigma
        self.dynamic_noise = 0
        self.measurement_noise = 0
        self.n = self.sigma.shape[0]
        self.g = Quaternion(0, np.array([0,0,9.81]))
        self.Q = Q
        self.error_matrix = np.zeros((3,2*self.n))

    # This will calculate the mean and the covariance of the quaternion
    # mu(k+1/k) and sigma(k+1/k)
    def propagate(self, dt):
        ''' Propagate state distribution through state equation. '''
        self.w = 1/(2*self.n)

        sigma_points = self.sigma_points_gen(dt, self.mu, self.sigma)
        sigma_points_new = np.zeros((2*self.n,), dtype=FilterState)

        # average of the angular velocity
        # average_angular = np.zeros(3,dtype=np.float64)
        for i in range(2*self.n):
            sigma_points_new[i] = self.dynamics(sigma_points[i], dt)
            # print("average angular velocity: ", average_angular, " current iteration: ",sigma_points[i].angular_velocity)
        #     average_angular += sigma_points_new[i].angular_velocity

        # average_angular = average_angular / (2*self.n)

        # # Coveriance of the angular velocity
        # # cov_angular = np.zeros((3,3))
        # # for i in range(2*self.n):
        # #     cov_angular += np.outer(sigma_points_new[i].angular_velocity - average_angular, (sigma_points_new[i].angular_velocity - average_angular).T)
        # # cov_angular = cov_angular / (2*self.n)

        # # Mean and covariance of the quaternion
        # quat_mean, quat_cov, _ = self.quat_stats(sigma_points_new)

        # ## Covairaance of the quaternion and the angular velocity
        # cov = np.zeros(self.sigma.shape)
        # for i in range(2*self.n):
        #     cov += np.hstack([self.error_matrix[:,i], sigma_points_new[i].angular_velocity - average_angular]) @ np.hstack([self.error_matrix[:,i], sigma_points_new[i].angular_velocity - average_angular]).T
        # cov = cov / (2*self.n)        

        # mean_state = FilterState(quat_mean, average_angular)

        # # # print("\nmu angular velocity: ", self.mu.angular_velocity)

        # # self.sigma[:3, :3] = quat_cov
        # # self.sigma[3:, 3:] = cov_angular

        # # print("dynamics sigma: ", self.sigma)

        # self.sigma = cov_angular
        return sigma_points_new



    # Using th sigma points to calculate the mean and the covariance of the quaternion
    def find_stats(self, points_sigma):
        ''' Compute mean of quaternion distribution. '''        
        
        quat_mean = self.mu.quaternion # Could be a problem
        avg = np.ones(3)
        threshold = 0.0001
        while(la.norm(avg) > threshold):
            for i in range(2*self.n):
                quat_e_i = points_sigma[i].quaternion * quat_mean.inv()
                # quat_e_i.normalize()        
                e_i = quat_e_i.axis_angle()
                # if np.linalg.norm(e_i) == 0: # not rotate
                #     e_i = np.zeros(3)
                # else:
                #     e_i = (-np.pi + np.mod(np.linalg.norm(e_i) + np.pi, 2 * np.pi)) / np.linalg.norm(e_i) * e_i

                self.error_matrix[:,i] = e_i
            
            avg = np.mean(self.error_matrix, axis=1)

            quat_mean = Quaternion().from_axis_angle(avg) * quat_mean
            # quat_mean.normalize()
        
        # I have the quant mean need to angular velocity
        # angular_vel = np.zeros((3,2*self.n))
        # for i in range(2*self.n):
        #     angular_vel[:,i] = points_sigma[i].angular_velocity
        angular_vel = [points_sigma[i].angular_velocity for i in range(2 * self.n)]
        angular_vel = np.array(angular_vel).T  

        avg_vel = np.mean(angular_vel, axis=1)
        print("avg_vel: ", avg_vel.shape)

        W_prime = np.concatenate((self.error_matrix, angular_vel - avg_vel[:, np.newaxis]), axis=0)
        covariance = 1.0 / (2 * self.n) * W_prime @ W_prime.T

        # error_vel = angular_vel - np.expand_dims(avg_vel,axis=1)
        # print("error_vel: ", error_vel.shape)

        # error_total = np.concatenate([self.error_matrix, error_vel], axis=0)
        # print("error_total: ", error_total.shape)

        # covariance = error_total @ error_total.T / (2*self.n)
        # print("covariance: ", covariance.shape)

        average_state = FilterState(quat_mean, avg_vel)

        return average_state, covariance, W_prime

    def measurment_update(self, observation, sigma_points,  average_state, covariance, error_total):
        ''' Update state distribution with measurement. '''

        # Calculating the mean and the covariance of the measurement
        z_mean = np.zeros(6)
        z_cov_yy = np.zeros((6,6))

        # print("observation: ", observation.shape)
        
        sigman_points = self.sigma_points_gen(0, average_state, covariance)
        transformed_points = np.zeros((2*self.n,), dtype=object)
        for i in range(2*self.n):
            transformed_points[i] = self.H(sigma_points[i])
        # There could be a problem with the mean of the measurement
        z_mean = np.mean(transformed_points, axis=0)
        print("z_mean: ", z_mean.shape)
        for i in range(2*self.n):
            z_cov_yy += (transformed_points[i] - z_mean)@(transformed_points[i] - z_mean).T
        z_cov_yy = z_cov_yy / (2*self.n)

        z_cov_yy += self.R # updating the yy covariance with the dynamic noise

        # z_cov_yy[:3,3:] = 0
        # z_cov_yy[3:,:3] = 0

        # Calculating the cross covariance
        cross_cov = np.zeros((self.n,6))
        for i in range(2*self.n):
            cross_cov += error_total[:,i]@(transformed_points[i] - z_mean).T
        cross_cov = cross_cov / (2*self.n)

        # Calculating the Kalman gain
        kalman_gain = np.dot(cross_cov, la.inv(z_cov_yy))
        # kalman_gain[:3,3:] = 0
        # kalman_gain[3:,:3] = 0

        # print("Kalman gain: ", kalman_gain.shape, " cross_cov: ", cross_cov.shape, " z_cov_yy: ", z_cov_yy.shape)

        # Calculating the mean and the covariance of the state
        # print(FilterState.from_vector(kalman_gain @ (observation - z_mean)).angular_velocity, self.mu.angular_velocity)
        # print("innovation shape: ", (observation - z_mean).shape, (observation - z_mean)[:3].shape)
        # observation[3:] = observation[3:] / la.norm(observation[3:])
        mean_update = kalman_gain @ (observation - z_mean)
        # print("mean update shape: ",mean_update.shape, mean_update[:3].shape)
        q_mean = Quaternion().from_axis_angle(mean_update[3:])
        self.mu = self.mu + FilterState(q_mean, mean_update[:3])

        self.sigma = self.sigma - (kalman_gain @ z_cov_yy @ kalman_gain.T)

        print("sigma: \n", self.sigma)

        self.innovation = observation - z_mean

        return self.mu, self.sigma, self.innovation

    def H(self, x):
        ''' Measurement equation. Z = omega + gausian noise'''

        noise_rot = np.random.normal(0, 0.01, x.angular_velocity.shape[0])
        noise_acc = np.random.normal(0, 0.01, x.angular_velocity.shape[0])
        g_bar = x.quaternion.inv() * self.g * x.quaternion
        # g_bar.normalize()
        output = np.zeros(6)
        output[3:] = x.angular_velocity #+ noise_rot
        output[:3] = g_bar.vec() #+ noise_acc
        return output

    def sigma_points_gen(self, dt, mean, covariance):
        ''' Generate sigma points. '''

        # sigma_root = np.linalg.cholesky((self.Q * dt + covariance)).T
        sigma_root = la.sqrtm((self.Q * dt + covariance)).T
        W = np.concatenate((np.sqrt(self.n) * sigma_root, -np.sqrt(self.n) * sigma_root), axis=1)
        sigma_points = np.zeros((2*self.n,), dtype=object)
        for i in range(2*self.n):
            w = W[:,i]
            w_q = Quaternion()
            w_q.from_axis_angle(w[:3])
            q = mean.quaternion * w_q
            sigma_points[i] = FilterState(q, mean.angular_velocity + w[3:])

        print("Sigma points: ", sigma_points[0])

        # sigma_root = la.sqrtm((self.Q * dt + self.sigma)).T
        # print( "\n\n\n\nSigma root: ",sigma_root, " Shape:", sigma_root.shape,)
        # print("\n\n Sigma: ", self.sigma, " Shape: ", self.sigma.shape)
        # sigma_root = sigma_root
        # sigma_points = np.zeros((2*self.n,), dtype=object)
        # for i in range(self.n):
        #     sigma_points[i] = self.mu + FilterState.from_vector(np.sqrt(self.n)*sigma_root[i,:])
        #     # sigma_points[i].quaternion.normalize()
        #     sigma_points[i+self.n] = self.mu - FilterState.from_vector(np.sqrt(self.n)*sigma_root[i,:])
        #     # sigma_points[i+self.n].quaternion.normalize()

        return sigma_points

    def dynamics(self, x, dt):
        ''' State equation. '''
        x_1 = x
        delta_quat = Quaternion()
        delta_quat.from_axis_angle(x.angular_velocity*dt)
        x_1.quaternion = x.quaternion* delta_quat 
        x_1.quaternion.normalize()

        return x_1
    
    # returns the array of euler angles
    def get_mean(self):
        return self.mu.quaternion.euler_angles()
    


