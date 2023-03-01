import numpy as np
from scipy import io

np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from filter import UncentedKalmanFilter
from quaternion import Quaternion
from state import FilterState


# Function to preprocess accelerometer readings
def preprocess_accel(accel_in):
    accel_processed = np.zeros_like(accel_in)
    accel_processed[0,:] = -accel_in[0,:]
    accel_processed[1,:] = -accel_in[1,:]
    accel_processed[2,:] = accel_in[2,:] 
    return accel_processed

# Function to preprocess gyro readings
def preprocess_gyro(gyro_in):
    gyro_processed = np.zeros_like(gyro_in)
    gyro_processed[2,:] = gyro_in[0,:]
    gyro_processed[1,:] = gyro_in[1,:]
    gyro_processed[0,:] = gyro_in[2,:] 
    return gyro_processed

# Function to convert rotation matrix to heading vector
def rot_to_heading(rot,T,g):
    heading = np.array([0,0,g])
    ori_vicon = np.zeros((3,T))
    ori_vicon = np.matmul(rot.T,heading).T # why
    return ori_vicon

# function to convert raw readings to SI units
def convert_to_SI(data, beta, alpha):
    return (data - beta) * 3300.0/(1023.0*alpha)

# Function to compute the derivative from viceon data
def compute_derivative(T_vicon, rotation_vicon, delta_t_vicon):
    #first convert to quaternion
    q_vicon = []
    q_temp = Quaternion()
    q_temp.from_rotm(rotation_vicon[:,:,0])
    q_vicon.append(q_temp)
    omegas = np.zeros((3,T_vicon))
    for i in range(1, T_vicon):
        q = Quaternion()
        q.from_rotm(rotation_vicon[:,:,i])
        q_vicon.append(q)
        temp = q_vicon[i-1].inv() * q_vicon[i] 
        omegas[:,i] = temp.axis_angle(dt=delta_t_vicon)
    
    return omegas
        
def main():
    data_num = 1
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
    accel = imu['vals'][0:3,:].astype(float)
    gyro = imu['vals'][3:6,:].astype(float)
    time_series = imu['ts'][0:].astype(float).T
    T = np.shape(imu['ts'])[1]
    delta_t = np.mean(np.diff(time_series))

    g = 9.81
    
    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    rotation_vicon = vicon['rots'].astype(float)
    time_series_vicon = vicon['ts'][0,:]
    time_series_vicon = time_series_vicon.astype(float).T
    T_vicon  = np.shape(vicon['ts'])[1]
    delta_t = np.mean(np.diff(time_series))
    delta_t_vicon = np.mean(np.diff(time_series_vicon))    
    
    accel_processed = preprocess_accel(accel)
    gyro_processed = preprocess_gyro(gyro)

    vicon_data = rot_to_heading(rotation_vicon,T,g)



    alphas_accel = np.array([[34.5826],
            [34.2690],
            [34.4407]])
    betas_accel = np.array([[-511.2068],
            [-500.470],
            [ 500.8867]])

    alphas_gyro = np.array([[249.9296],
            [250.0695],
            [250.0364]])
    betas_gyro = np.array([[374.5049],
            [375.2100],
            [373.0922]])

    accel_converted = convert_to_SI(accel_processed, betas_accel, alphas_accel)
    gyro_converted = convert_to_SI(gyro_processed, betas_gyro, alphas_gyro)

    # # Ploting accelerometer data
    # plt.figure()
    # plt.plot(time_series, accel_converted[0,:], label="x", color="red")
    # plt.plot(time_series, accel_converted[1,:], label="y", color="green")
    # plt.plot(time_series, accel_converted[2,:], label="z", color="blue")
    # plt.legend()
    # plt.title("Calibrated Accelerometer Data")
    # plt.xlabel("Time (s) ->")
    # plt.ylabel("Acceleration (m/s^2) ->")

    # # Ploting gyroscope data
    # plt.figure()
    # plt.plot(time_series_vicon, vicon_data[0,:], label="x", color="red")
    # plt.plot(time_series_vicon, vicon_data[1,:], label="y", color="green")
    # plt.plot(time_series_vicon, vicon_data[2,:], label="z", color="blue")
    # plt.legend()
    # plt.title("Ground Truth Data")
    # plt.xlabel("Time (s) ->")
    # plt.ylabel("Acceleration (m/s^2) ->")

    omega_vicon = compute_derivative(T_vicon, rotation_vicon, delta_t_vicon)   

    # #plot out vicon omega data
    # plt.figure()
    # plt.plot(time_series_vicon, omega_vicon[0,:], label="x", color="red")
    # plt.plot(time_series_vicon, omega_vicon[1,:], label="y", color="green")
    # plt.plot(time_series_vicon, omega_vicon[2,:], label="z", color="blue")
    # plt.legend()
    # plt.title("Vicon Omega Data")
    # plt.ylabel("Omega (rad/s)")
    # # plot out gyro data

    # plt.figure()
    # plt.plot(time_series_vicon[:T_vicon], gyro_converted[1,:T_vicon], label="y", color="green")
    # plt.plot(time_series_vicon[:T_vicon], gyro_converted[0,:T_vicon], label="x", color="red")
    # plt.plot(time_series_vicon[:T_vicon], gyro_converted[2,:T_vicon], label="z", color="blue")
    # plt.legend()
    # plt.title("Gyro Data")
    # plt.ylabel("Omega (rad/s)")
    
    # initial state (orientation, omega) from vicon
    q0 = Quaternion()
    q0.from_axis_angle(vicon_data[:,0])
    
    x0 = FilterState(q0, omega_vicon[:,0])
    
    # initial covariance, 6x6
    sigma = np.diag(np.ones(6)) * 1
    
    # measurement noise covariance, 6x6
    Q = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    # process noise covariance, 6x6
    R = np.diag([1.5, 1.5, 1.5, 4.0, 4.0, 4.0])
    # # state transition function
    # f = IMUModel.f
    # # measurement function
    # g = IMUModel.g

    ukf = UncentedKalmanFilter(Q, R, g, x0, sigma)
    past_state = np.zeros((3, T))
    innovation = np.zeros((6, T))
    print(T//5*4)
    for i in range(T//4, T//4*2):
        print("timestep: ", i , " of ", T)
        dt = time_series[i] - time_series[i-1]
        if (dt < 0):
            print("dt is negative")
            break
        y_hat = np.zeros(6)
        y_hat[:3] = gyro_converted[:,i].reshape(3,)
        y_hat[3:] = accel_converted[:,i].reshape(3,)
        # print(y_hat)
        sigma_points = ukf.propagate(dt)
        average_state, covariance, error_total = ukf.find_stats(sigma_points)
        _, _, innovation[:,i] = ukf.measurment_update(y_hat, sigma_points, average_state, covariance, error_total)
        past_state[:,i] = ukf.get_mean()
        
    # plot
    plt.figure()
    plt.plot(time_series, past_state[0,:], label="x", color="red")
    plt.plot(time_series, past_state[1,:], label="y", color="green")
    plt.plot(time_series, past_state[2,:], label="z", color="blue")
    plt.plot(time_series_vicon, vicon_data[0,:], label="vicon x", alpha=0.2, color="red", linewidth=10)
    plt.plot(time_series_vicon, vicon_data[1,:], label="vicon y", alpha=0.2, color="green", linewidth=10)
    plt.plot(time_series_vicon, vicon_data[2,:], label="vicon z", alpha=0.2, color="blue", linewidth=10)
    plt.legend()
    plt.show()

    # plot the innovation
    plt.figure()
    plt.plot(time_series, innovation[0,:], label="x", color="red")
    plt.plot(time_series, innovation[1,:], label="y", color="green")
    plt.plot(time_series, innovation[2,:], label="z", color="blue")
    plt.legend()
    plt.show() 

if __name__ == "__main__":
    main()