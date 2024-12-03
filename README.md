#!/usr/bin/env python

import argparse
import sys
import time

import adapy
import numpy as np
import rospy

from adarrt import AdaRRT

if sys.version_info[0] < 3:
    current_time = time.clock
    user_input = raw_input
else:
    current_time = time.process_time
    user_input = input

def createBw():
    """
    Creates the bounds matrix for the TSR.

    :returns: A 6x2 array Bw
    """
    ### FILL in your code here (Q2 and Q3)
    Bw = np.array([
        [0, 0], # xmin, xmax
        [0, 0], # ymin, ymax
        [0, 0], # zmin, zmax
        [0, 0], # roll_min, roll_max
        [0, 0], # pitch_min, pitch_max
        [0, np.pi]  # yaw_min, yaw_max
    ])

    ###
    return Bw

def createSodaTSR(soda_pose, hand):
    """
    Create the TSR for grasping a soda can.

    :param soda_pose: SE(3) transform from world to soda can.
    :param hand: ADA hand object
    :returns: A fully initialized TSR.
    """
    sodaTSR = adapy.get_default_TSR()
    sodaTSR.set_T0_w(soda_pose)

    rot_trans = np.eye(4)
    rot_trans[0:3, 0:3] = np.array([[0,  0, -1],
                                    [1,  0,  0],
                                    [0, -1,  0]])
    sodaTSR_Tw_e = np.matmul(
        rot_trans, hand.get_endeffector_transform("cylinder"))
    sodaTSR_Tw_e[0,3] = -0.04
    sodaTSR_Tw_e[2,3] = 0.06

    sodaTSR.set_Tw_e(sodaTSR_Tw_e)
    Bw = createBw()
    sodaTSR.set_Bw(Bw)
    return sodaTSR

def van_der_corput(n_sample, base=2):
    """
    Van der Corput sequence.

    :param n_sample: number of elements in the sequence.
    :param base: base of the sequence.
    :return: The Van der Corput as a list.
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence

def try_shortcut(
        start_pos, end_pos, ada, collision_constraint, sequence):
    """
    Sample between 2 points to check for collision.
    """
    skeleton = ada.get_arm_skeleton()
    state_space = ada.get_arm_state_space()

    for point in sequence:
        interpolated = start_pos + (end_pos - start_pos) * point
        if not collision_constraint.is_satisfied(
                state_space, skeleton, interpolated):
            return False
    return True

def shortcut(waypoints, ada, collision_constraint, time_limit=7.0):
    """
    Smooth a path by sparsifying it.
    """
    sequence = van_der_corput(300)
    if collision_constraint is None:
        return waypoints
    start_time = time.time()
    elapsed_time = time.time() - start_time
    while elapsed_time < time_limit:
        length = len(waypoints)
        start_idx = np.random.randint(0, length - 1)
        end_idx = np.random.randint(start_idx + 1, length)

        if try_shortcut(waypoints[start_idx],
                        waypoints[end_idx],
                        ada,
                        collision_constraint,
                        sequence):
            print("shortcutting...")
            del waypoints[start_idx + 1: end_idx]

        elapsed_time = time.time() - start_time

def close_hand(hand, preshape):
    """
    Close the hand on the ADA.

    :param hand: ADA hand object
    :param preshape: The joint values as (f1, f2)
    :returns: None
    """
    if (len(preshape) != 2 or
        preshape[0] < 0. or preshape[0] > 1.6 or
        preshape[1] < 0. or preshape[1] > 1.6):
        print("bad preshape input.")
        return
    hand.execute_preshape(preshape)

def main(if_sim):
    # initialize roscpp, if it's for real robot
    # http://wiki.ros.org/ROS/Tutorials/Using%20a%20C%2B%2B%20class%20in%20Python
    if not if_sim:
        from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
        roscpp_init('soda_grasp_ik', [])

    # instantiate an ada
    ada = adapy.Ada(if_sim)

    # launch viewer
    viewer = ada.start_viewer("dart_markers/sode_grasp", "map")
    world = ada.get_world()
    hand = ada.get_hand()
    hand_node = hand.get_endeffector_body_node()
    arm_skeleton = ada.get_arm_skeleton()
    arm_state_space = ada.get_arm_state_space()

    # joint positions of the starting pose
    arm_home = [-1.5, 3.22, 1.23, -2.19, 1.8, 1.2]

    if if_sim:
        ada.set_positions(arm_home)
    else:
        user_input("Please move arm to home position with the joystick. " +
            "Press ENTER to continue...")

    viewer.add_frame(hand_node)

    # add objects to world
    soda_pose = np.eye(4)
    soda_pose[0, 3] = 0.25
    soda_pose[1, 3] = -0.35
    sodaURDFUri = "package://pr_assets/data/objects/can.urdf"
    soda = world.add_body_from_urdf_matrix(sodaURDFUri, soda_pose)

    bowl_pose = np.eye(4)
    bowl_pose[0, 3] = 0.40
    bowl_pose[1, 3] = -0.25
    bowlURDFUri = "package://pr_assets/data/objects/plastic_bowl.urdf"
    bowl = world.add_body_from_urdf_matrix(bowlURDFUri, bowl_pose)

    tableURDFUri = "package://pr_assets/data/furniture/uw_demo_table.urdf"
    # x, y, z, rw, rx, ry, rz
    table_pose = [0.3, 0.0, -0.75, 0.707107, 0., 0., 0.707107]
    table = world.add_body_from_urdf(tableURDFUri, table_pose)

    # add collision constraints
    collision_free_constraint = ada.set_up_collision_detection(
        ada.get_arm_state_space(),
        ada.get_arm_skeleton(),
        [soda, bowl, table])
    full_collision_constraint = ada.get_full_collision_constraint(
        ada.get_arm_state_space(),
        ada.get_arm_skeleton(),
        collision_free_constraint)

    rospy.sleep(1.)
    user_input("Press ENTER to generate the TSR...")

    # create TSR
    sodaTSR = createSodaTSR(soda_pose, hand)
    marker = viewer.add_tsr_marker(sodaTSR)
    user_input("Press ENTER to start planning goals...")

    # set up IK generator
    ik_sampleable = adapy.create_ik(
        arm_skeleton,
        arm_state_space,
        sodaTSR,
        hand_node)
    ik_generator = ik_sampleable.create_sample_generator()
    configurations = []

    samples = 0
    maxSamples = 10
    while samples < maxSamples and ik_generator.can_sample():
        goal_state = ik_generator.sample(arm_state_space)
        samples += 1
        if len(goal_state) == 0:
            continue
        configurations.append(goal_state)

    if len(configurations) == 0:
        print("No valid configurations found!")

    if if_sim:
        ada.set_positions(arm_home)

    user_input("Press ENTER to start RRT planning...")
    trajectory = None
    for configuration in configurations:
        # Your AdaRRT planner
        ### FILL in your code here (Q4)
        adaRRT = AdaRRT(
            start_state=np.array(arm_home),
            goal_state=np.array(configuration),
            ada=ada,
            ada_collision_constraint=full_collision_constraint,
            step_size=0.25,
            goal_precision=1.0
        )
        rospy.sleep(1.0)
        trajectory = adaRRT.build()

        ###
        if trajectory:
            break

    if not trajectory:
        print("Failed to find a solution!")
        sys.exit(1)
    else:
        print("Found a trajectory!")

    # smooth the RRTs trajectory
    shortcut(trajectory, ada, full_collision_constraint)
    waypoints = []
    for i, waypoint in enumerate(trajectory):
        waypoints.append((0.0 + i, waypoint))

    # compute trajectory in joint space
    t0 = current_time()
    traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
    retimed_traj = ada.compute_retime_path(ada.get_arm_skeleton(), traj)
    t = current_time() - t0
    print(str(t) + " seconds elapsed")
    user_input('Press ENTER to execute the trajectory...')

    # execute the trajectory
    if not if_sim:
        ada.start_trajectory_executor()
    ada.execute_trajectory(retimed_traj)
    time.sleep(0.05)
    user_input('Press ENTER after robot has approached the can...')
    if not if_sim:
        ada.set_positions(waypoints[-1][1])

    # execute the grasp
    print("Closing hand")
    ### FILL in your code here (Q5)
    preshape = [1,1]
    close_hand(hand, preshape)
    ###

    user_input('Press ENTER after robot has succeeded closing the hand...')
    if if_sim:
        hand.grab(soda)

    # compute the Jacobian pseudo-inverse for moving the hand upwards
    ### FILL in your code here (Q6 and Q7)
    q = arm_skeleton.get_positions()

    def move_dir(q, delt_x, steps):
        for i in range(steps):
            J = arm_skeleton.get_jacobian(hand.get_endeffector_body_node())
            Jt = np.transpose(J)
            J_Jt_Dot = np.dot(J, Jt)
            inverse = np.linalg.inv(J_Jt_Dot)
            J_dot_inv = np.dot(Jt,inverse)
            delt_q_err = np.dot(J_dot_inv,  delt_x)
            print(delt_q_err)
            q -= delt_q_err

            if if_sim:
                ada.set_positions(q)
                viewer.update()
                time.sleep(0.10)
        
        if not if_sim:
		    # in real world
            traj = ada.plan_to_configuration(ada.get_arm_state_space(), ada.get_arm_skeleton(), q)
            retimed_traj = ada.compute_retime_path(ada.get_arm_skeleton(), traj)
            ada.execute_trajectory(retimed_traj)
            time.sleep(0.50)
        
        return q
    

    print("Lifting can...")
    q = move_dir(q, np.array([0,0,0,-0.01,0,0]), 50)
    time.sleep(5)

    # This section onwards is for the in-person lab

    print("Lowering can...")
    q = move_dir(q, np.array([0,0,0,0.01,0,0]), 42)
    time.sleep(5)
    
    print("Moving forward...")
    q = move_dir(q, np.array([0,0,0,0,0,-0.01]), 20)
    time.sleep(5)

    print("Rotating can...")
    q = move_dir(q, np.array([0,0,0.03,0,0,0]), 50)
    user_input('Press ENTER after robot has succeeded rotating can...')

    # clean the scene
    # world.remove_skeleton(soda)
    # world.remove_skeleton(table)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--sim', dest='if_sim', action='store_true')
   parser.add_argument('--real', dest='if_sim', action='store_false')
   parser.set_defaults(if_sim=True)
   args = parser.parse_args()
   main(args.if_sim)
