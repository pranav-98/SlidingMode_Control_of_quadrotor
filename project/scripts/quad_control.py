#!/usr/bin/env python3
from math import pi, sqrt, atan2, cos, sin, asin
from turtle import position
import numpy as np
from numpy import NaN
import rospy
import tf
from std_msgs.msg import Empty, Float32
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Twist, Pose2D
import pickle
import os


m=27*(10**-3)
l=46*(10**-3)
Ix=16.57171*(10**-6)
Iy=16.57171*(10**-6)
Iz=29.261652*(10**-6)
Ip=12.65625 *(10**-8)
kf=1.28192*(10**-8)
km=5.964552*(10**-3)
g=9.8








Kp=20
Kd=-3.5





lamda1=0.5
rhok1=1
#For u2
lamda2=5
#rhok2=25
rhok2=145
#For u3
lamda3=10
#rhok3=25
rhok3=200
#For u4
lamda4=10
rhok4=5



class Quadrotor():
    def __init__(self):

        self.w1=0
        self.w2=0
        self.w3=0
        self.w4=0
        

        # Declaring the variables for motor speed
        

        #Allocation Matrix
        self.alloc=np.array([[1/(4*kf),(2**0.5)*-1/(4*kf*l),(2**0.5)*-1/(4*kf*l),-1/(4*km*kf)],[1/(4*kf),(2**0.5)*-1/(4*kf*l),(2**0.5)*1/(4*kf*l), 1/(4*km*kf)],[1/(4*kf),(2**0.5)*1/(4*kf*l), (2**0.5)*1/(4*kf*l), -1/(4*km*kf)],[1/(4*kf),(2**0.5)*1/(4*kf*l), (2**0.5)*-1/(4*kf*l), 1/(4*km*kf)]])


        # publisher for rotor speeds
        self.motor_speed_pub = rospy.Publisher("/crazyflie2/command/motor_speed", Actuators, queue_size=10)
        # subscribe to Odometry topic
        self.odom_sub = rospy.Subscriber("/crazyflie2/ground_truth/odometry", Odometry, self.odom_callback)
        self.t0 = None
        self.t = None
        self.t_series = []
        
        self.x_series = []
        self.y_series = []
        self.z_series = []
        self.mutex_lock_on = False


        #Additional variables

        #These have been calculated using matlab
        self.coef_matrix=np.array([[-2.35155790711349*(10**-8),5.76131687242802*(10**-6),-0.000442680776014111, 0.0116255144032922,-0.0477660199882423, 0],[7.52498530276309*(10**-8), -1.15226337448560*(10**-5),	0.000557319223985892, -0.00868312757201648, 0.0308759553203999, 0],[8.79120879120884*(10**-8), -1.53846153846155*(10**-5), 0.000978021978021982, -0.0273076923076924, 0.313956043956044, 0]])
        self.coef_matrix_ddot=np.array([[0,0,-35535738377855475/75557863725914323419136,20405287271659089/295147905179352825856,-3062257118056293/1152921504606846976,113/4860],[0,0,28428590702284085/18889465931478580854784,-5101321817914755/36893488147419103232,1927635954792405/576460752303423488,-211/12150],[0,0,33212247791610875/18889465931478580854784,-3/16250 ,267/45500 ,-71/1300]])
        
        
        self.coef_matrix_dot=np.array([[0,-35535738377855475/302231454903657293676544,6801762423886363/295147905179352825856,-3062257118056293/2305843009213693952,113/4860,-325/6804],[0,28428590702284085/75557863725914323419136, -1700440605971585/36893488147419103232,1927635954792405/1152921504606846976, -211/12150, 1313/42525],[0,33212247791610875/75557863725914323419136, -1/16250,267/91000, -71/1300,2857/9100]])
                                                             

        self.x_des=0
        self.y_des=0
        self.z_des=0

        self.x_desdot=0
        self.y_desdot=0
        self.z_desdot=0

        self.x_desddot=0
        self.y_desddot=0
        self.z_desddot=0
        rospy.on_shutdown(self.save_data)
        



    def traj_evaluate(self):
        # TODO: evaluating the corresponding trajectories designed in Part 1to return the desired positions, velocities and accelerations
        T_matrix=np.array([self.t**5,self.t**4,self.t**3,self.t**2,self.t**1,1])
        T_matrix=np.transpose(T_matrix)
        
        self.x_des=self.coef_matrix[0]@T_matrix
        self.y_des=self.coef_matrix[1]@T_matrix
        self.z_des=self.coef_matrix[2]@T_matrix
        
        self.x_desdot=self.coef_matrix_dot[0]@T_matrix
        self.y_desdot=self.coef_matrix_dot[1]@T_matrix
        self.z_desdot=self.coef_matrix_dot[2]@T_matrix

        self.x_desddot=self.coef_matrix_ddot[0]@T_matrix
        self.y_desddot=self.coef_matrix_ddot[1]@T_matrix
        self.z_desddot=self.coef_matrix_ddot[2]@T_matrix
        
        


        

    def smc_control(self, xyz, xyz_dot, rpy, rpy_dot):
        # obtain the desired values by evaluating the corresponding trajectories
        if self.t<=65:
            self.traj_evaluate()
            omega=self.w1-self.w2+self.w3-self.w4
            
            x=xyz[0,0]
            y=xyz[1,0]
            z=xyz[2,0]

            xdot=xyz_dot[0,0]
            ydot=xyz_dot[1,0]
            zdot=xyz_dot[2,0]

            r=rpy[0,0]
            p=rpy[1,0]
            yaw=rpy[2,0]

            rd=rpy_dot[0,0]
            pd=rpy_dot[1,0]
            yawd=rpy_dot[2,0]

            
            print(x,y,z)
            Fx=m*(-Kp*(x-self.x_des)-Kd*(xdot-self.x_desdot)+ self.x_desddot)
            Fy=m*(-Kp*(y-self.y_des)-Kd*(ydot-self.y_desdot)+ self.y_desddot)
            
            s1=(zdot-self.z_desdot)+lamda1*(z-self.z_des)
            if s1>0:
                sgns1=+1
            elif s1<0:
                sgns1=-1
            else:
                sgns1=0

            u1=m*(g-lamda1*(zdot-self.z_desdot)+ self.z_desddot - rhok1 * sgns1)*(1/(np.cos(r)*np.cos(p)))
            
            fxu1=Fx/u1  
            fyu1=Fy/u1 
            fxu1=min(fxu1,1)
            fxu1=max(fxu1,-1)
            fyu1=min(fyu1,1)
            fyu1=max(fyu1,-1)
            theta_des=asin(fxu1)
            phi_des=asin(-fyu1)
          
            s2=rd+lamda2*np.arctan2(np.sin(r-phi_des),np.cos(r-phi_des))
            
            s3=pd+lamda3*np.arctan2(np.sin(p-theta_des),np.cos(p-theta_des))
            s4=yawd+lamda4*np.arctan2(np.sin(yaw),np.cos(yaw))

            if s2>0:
                sgns2=+1
            elif s2<0:
                sgns2=-1
            else:
                sgns2=0


            if s3>0:
                sgns3=+1
            elif s3<0:
                sgns3=-1
            else: 
                sgns3=0

            if s4>0:
                sgns4=+1
            elif s4<0:
                sgns4=-1
            else: 
                sgns4=0

            


            u2=Ix*((-pd*yawd*((Iy-Ix)/Ix))+Ip*omega*pd/Ix-lamda2*(rd)-rhok2*sgns2)
            u3=Iy*((-rd*yawd*((Iz-Ix)/Iy))+Ip*omega*rd/Ix-lamda3*(pd)-rhok3*sgns3)
            u4=Iz*(((-rd*pd*((Ix-Iy)/Iz))-lamda4*(yawd)-rhok4*sgns4))
            
            U=np.array([u1,u2,u3,u4])

            
            vel_sq=self.alloc@U
            
            for i in range(0,len(vel_sq)):
                vel_sq[i]=abs(vel_sq[i])#max(0,vel_sq[i])
            vel=vel_sq**(1/2)
            
            for i in range(0,len(vel)):
                vel[i]=min(vel[i],2618)


            
            
            motor_vel=vel
            
            motor_speed = Actuators()
            
            zero_vel=[0,0,0,0]
            motor_speed.angular_velocities = [motor_vel[0], motor_vel[1], motor_vel[2], motor_vel[3]]
            self.w1=motor_vel[0]
            self.w2=motor_vel[1]
            self.w3=motor_vel[2]
            self.w4=motor_vel[3]
            #motor_speed.angular_velocities = zero_vel
            self.motor_speed_pub.publish(motor_speed)
            #print(s1,s2,s3,s4)
        else:
            motor_speed1=Actuators()
                
            motor_speed1.angular_velocities=[0,0,0,0]
              

        
        
        


# odometry callback function (DO NOT MODIFY)
    def odom_callback(self, msg):
        if self.t0 == None:
            self.t0 = msg.header.stamp.to_sec()
        self.t = msg.header.stamp.to_sec() - self.t0
        # convert odometry data to xyz, xyz_dot, rpy, and rpy_dot
        w_b = np.asarray([[msg.twist.twist.angular.x], [msg.twist.twist.angular.y], [msg.twist.twist.angular.z]])
        v_b = np.asarray([[msg.twist.twist.linear.x], [msg.twist.twist.linear.y], [msg.twist.twist.linear.z]])
        xyz = np.asarray([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.pose.pose.position.z]])
        q = msg.pose.pose.orientation
        T = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0:3, 3] = xyz[0:3, 0]
        R = T[0:3, 0:3]
        xyz_dot = np.dot(R, v_b)
        rpy = tf.transformations.euler_from_matrix(R, 'sxyz')
        rpy_dot = np.dot(np.asarray([[1, np.sin(rpy[0])*np.tan(rpy[1]), np.cos(rpy[0])*np.tan(rpy[1])],[0, np.cos(rpy[0]), -np.sin(rpy[0])], [0, np.sin(rpy[0])/np.cos(rpy[1]), np.cos(rpy[0])/np.cos(rpy[1])]]), w_b)
        rpy = np.expand_dims(rpy, axis=1)
        # store the actual trajectory to be visualized later
        if (self.mutex_lock_on is not True):
            self.t_series.append(self.t)
            self.x_series.append(xyz[0, 0])
            self.y_series.append(xyz[1, 0])
            self.z_series.append(xyz[2, 0])


        # call the controller with the current states
        self.smc_control(xyz, xyz_dot, rpy, rpy_dot)



    # save the actual trajectory data
    def save_data(self):
        # TODO: update the path below with the correct path
        with open("/home/pranav/Desktop/rbe_502_project/src/project/scripts/log.pkl", "wb") as fp:
            self.mutex_lock_on = True
            pickle.dump([self.t_series,self.x_series,self.y_series,self. z_series], fp)


if __name__ == '__main__':
    rospy.init_node("quadrotor_control")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = Quadrotor()
    try:
       rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")