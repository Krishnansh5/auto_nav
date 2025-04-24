#!/usr/bin/env python3
"""
Trajectory Planning Module

This module provides various trajectory planners for robot motion control.
Each planner generates smooth trajectories with different characteristics:
- Linear: Constant velocity motion
- Cubic: Smooth motion with zero velocity at endpoints
- Quintic: Smooth motion with zero velocity and acceleration at endpoints
- Sinusoidal: Smooth cyclic motion with sinusoidal profile

Author: Amritanshu Manu
Date: December 2024
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class TrajectoryState:
    """Data class to hold trajectory state information."""
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    yaw: np.ndarray
    time_points: np.ndarray

class TrajectoryPlanner(ABC):
    """
    Abstract base class for trajectory planners.
    
    Attributes:
        start_pos (np.ndarray): Starting joint positions
        end_pos (np.ndarray): Ending joint positions
        duration (float): Duration of the trajectory
        dt (float): Time step for trajectory points
        num_joints (int): Number of joints to plan for
    """
    
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                 dt: float = 0.01) -> None:
        """
        Initialize the trajectory planner.

        Args:
            start_pos: Starting joint positions
            end_pos: Ending joint positions
            duration: Motion duration in seconds
            dt: Time step for trajectory points
        """
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        # self.duration = duration
        self.dt = dt
        self.num_joints = len(start_pos)
        # self.time_points = np.arange(0, duration + dt, dt)
        # self.num_points = len(self.time_points)
        
    @abstractmethod
    def generate_trajectory(self) -> TrajectoryState:
        """Generate trajectory waypoints. Must be implemented by child classes."""
        pass
    
    # def _initialize_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize arrays for positions, velocities, and accelerations."""
        # positions = np.zeros((self.num_joints, self.num_points))
        # velocities = np.zeros((self.num_joints, self.num_points))
        # accelerations = np.zeros((self.num_joints, self.num_points))
        # return positions, velocities, accelerations

class LinearTrajectoryPlanner(TrajectoryPlanner):
    """
    Generates linear trajectories with either constant or trapezoidal velocity profile.
    The trapezoidal profile provides smooth acceleration and deceleration phases.
    """
    
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                 duration: float, dt: float = 0.01,
                 acceleration_time: float = None,
                 max_velocity: np.ndarray = None,
                 use_trapezoidal: bool = True) -> None:
        """
        Initialize the linear trajectory planner.

        Args:
            start_pos: Starting joint positions
            end_pos: Ending joint positions
            duration: Motion duration in seconds
            dt: Time step for trajectory points
            acceleration_time: Time for acceleration/deceleration phases (seconds)
            max_velocity: Maximum velocity for each joint (if None, computed automatically)
            use_trapezoidal: Whether to use trapezoidal velocity profile (True) or constant velocity (False)
        """
        super().__init__(start_pos, end_pos, duration, dt)
        self.use_trapezoidal = use_trapezoidal
        
        if use_trapezoidal:
            # Set acceleration time (default: 15% of total duration for each phase)
            self.acceleration_time = acceleration_time or (0.15 * duration)
            if self.acceleration_time * 2 >= duration:
                raise ValueError("Acceleration time too large for given duration")
            
            # Calculate required velocities and accelerations
            self.displacement = self.end_pos - self.start_pos
            
            if max_velocity is None:
                # Calculate minimum required max velocity to achieve displacement
                self.max_velocity = np.abs(self.displacement) / (
                    duration - self.acceleration_time
                )
            else:
                self.max_velocity = max_velocity
                
            # Calculate accelerations
            self.acceleration = self.max_velocity / self.acceleration_time
            
            # Ensure signs match the direction of motion
            self.max_velocity *= np.sign(self.displacement)
            self.acceleration *= np.sign(self.displacement)

    def generate_trajectory(self) -> TrajectoryState:
        """
        Generate linear trajectory with either constant or trapezoidal velocity profile.
        
        Returns:
            TrajectoryState: Object containing positions, velocities, and accelerations
        """
        positions, velocities, accelerations = self._initialize_arrays()
        
        if not self.use_trapezoidal:
            # Simple constant velocity profile
            joint_velocities = (self.end_pos - self.start_pos) / self.duration
            
            for i in range(self.num_joints):
                positions[i] = self.start_pos[i] + joint_velocities[i] * self.time_points
                velocities[i] = np.full_like(self.time_points, joint_velocities[i])
                # Accelerations remain zero
        else:
            # Trapezoidal velocity profile
            for i in range(self.num_joints):
                for j, t in enumerate(self.time_points):
                    if t <= self.acceleration_time:
                        # Acceleration phase
                        velocities[i, j] = self.acceleration[i] * t
                        accelerations[i, j] = self.acceleration[i]
                        positions[i, j] = (self.start_pos[i] + 
                                         0.5 * self.acceleration[i] * t**2)
                        
                    elif t <= self.duration - self.acceleration_time:
                        # Constant velocity phase
                        velocities[i, j] = self.max_velocity[i]
                        accelerations[i, j] = 0
                        positions[i, j] = (self.start_pos[i] +
                                         0.5 * self.acceleration[i] * self.acceleration_time**2 +
                                         self.max_velocity[i] * (t - self.acceleration_time))
                        
                    else:
                        # Deceleration phase
                        time_left = self.duration - t
                        velocities[i, j] = self.acceleration[i] * time_left
                        accelerations[i, j] = -self.acceleration[i]
                        
                        # Position during deceleration
                        constant_distance = (self.max_velocity[i] * 
                                          (self.duration - 2*self.acceleration_time))
                        positions[i, j] = (self.end_pos[i] - 
                                         0.5 * self.acceleration[i] * time_left**2)
                        
        return TrajectoryState(positions, velocities, accelerations, self.time_points)

class CubicTrajectoryPlanner(TrajectoryPlanner):
    """Generates cubic polynomial trajectories with zero velocity at endpoints and velocity constraints."""
    
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                 duration: float, dt: float = 0.01,
                 max_velocity: Optional[np.ndarray] = None) -> None:
        """
        Initialize the cubic trajectory planner.

        Args:
            start_pos: Starting joint positions
            end_pos: Ending joint positions
            duration: Motion duration in seconds
            dt: Time step for trajectory points
            max_velocity: Maximum allowed velocity for each joint
        """
        super().__init__(start_pos, end_pos, duration, dt)
        self.max_velocity = max_velocity

        # If max_velocity not provided, set to a reasonable default
        if self.max_velocity is None:
            # Default to 2x the average velocity needed
            displacement = np.abs(end_pos - start_pos)
            self.max_velocity = 2 * displacement / duration

    def check_velocity_feasibility(self, max_computed_velocity: np.ndarray) -> bool:
        """
        Check if the computed trajectory violates velocity constraints.

        Args:
            max_computed_velocity: Maximum computed velocity for each joint

        Returns:
            bool: True if trajectory is feasible, False otherwise
        """
        if np.any(np.abs(max_computed_velocity) > self.max_velocity):
            infeasible_joints = np.where(np.abs(max_computed_velocity) > self.max_velocity)[0]
            print(f"Warning: Velocity limits exceeded for joints {infeasible_joints}")
            print(f"Max computed velocities: {max_computed_velocity}")
            print(f"Velocity limits: {self.max_velocity}")
            return False
        return True

    def generate_trajectory(self) -> TrajectoryState:
        """
        Generate cubic polynomial trajectory with velocity checks.
        
        Returns:
            TrajectoryState: Object containing positions, velocities, and accelerations

        Raises:
            ValueError: If trajectory cannot satisfy velocity constraints
        """
        positions, velocities, accelerations = self._initialize_arrays()
        
        # Calculate coefficients and check velocities for each joint
        for i in range(self.num_joints):
            # Cubic polynomial coefficients
            a0 = self.start_pos[i]
            a1 = 0  # Zero initial velocity
            a2 = 3 * (self.end_pos[i] - self.start_pos[i]) / (self.duration**2)
            a3 = -2 * (self.end_pos[i] - self.start_pos[i]) / (self.duration**3)
            
            # Calculate maximum velocity for this joint
            # For cubic: max velocity occurs at t = duration/2
            max_velocity = abs(a1 + 2*a2*(self.duration/2) + 3*a3*(self.duration/2)**2)
            
            # Check if velocity is feasible
            if max_velocity > self.max_velocity[i]:
                min_duration = 1.5 * abs(self.end_pos[i] - self.start_pos[i]) / self.max_velocity[i]
                raise ValueError(
                    f"Joint {i} velocity constraint violated. "
                    f"Computed max velocity: {max_velocity:.2f}, "
                    f"Limit: {self.max_velocity[i]:.2f}. "
                    f"Try increasing duration to at least {min_duration:.2f} seconds."
                )
            
            # Generate trajectory
            t = self.time_points
            positions[i] = a0 + a1*t + a2*t**2 + a3*t**3
            velocities[i] = a1 + 2*a2*t + 3*a3*t**2
            accelerations[i] = 2*a2 + 6*a3*t
        positions = positions.T
        velocities = velocities.T
        accelerations = accelerations.T
        return TrajectoryState(positions, velocities, accelerations, self.time_points)

class CubicTrajectorySplinePlanner(TrajectoryPlanner):
    """Generates cubic polynomial trajectories with via points and continuous velocities."""
    
    def __init__(self, waypoints: List[np.ndarray], 
                 dt: float = 0.01, max_velocity: List[np.ndarray] = None, max_yaw_rate = np.pi/4) -> None:
        """
        Initialize the cubic trajectory planner with via points.

        Args:
            waypoints: List of joint positions including start, via points, and end.
            times: List of corresponding times for each waypoint.
            dt: Time step for trajectory points.
            max_velocity: Maximum allowed velocity for each joint.
        """
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required.")
        # if len(waypoints) != len(times):
        #     raise ValueError("Waypoints and times must have the same length.")
        
        super().__init__(waypoints[0], waypoints[-1], dt)
        self.duration_factor = 1.33
        self.waypoints = waypoints
        # self.times = np.array(times)
        self.num_joints = waypoints[0].size
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate

        self.duration = 0
        distances = []
        for i in range(len(waypoints)-1):
            xi,yi = waypoints[i][0],waypoints[i][1]
            xe,ye = waypoints[i+1][0],waypoints[i+1][1]
            seg_distance = np.linalg.norm(np.array([ye, xe]) - np.array([yi, xi]))
            distances.append(seg_distance)

            mvx,mvy = max_velocity[0],max_velocity[1]
            max_speed = np.linalg.norm([mvx,mvy])

            min_time=seg_distance/max_speed
            self.duration+=min_time

        self.duration*=2
        total_distance = np.sum(distances)
        self.distance_ratios = np.array(distances)/total_distance

        self.times = self.get_times_from_duration()

        # Set default max_velocity based on overall displacement
        # if self.max_velocity is None:
        #     displacement = np.abs(waypoints[-1] - waypoints[0])
        #     total_duration = times[-1] - times[0]
        #     self.max_velocity = 2 * displacement / total_duration

    def get_times_from_duration(self,):
        time_diff = [self.duration*r for r in self.distance_ratios]
        times = [0]
        for i in range(len(time_diff)):
            times.append(time_diff[i]+times[-1])
        return times


    def check_velocity_feasibility(self, max_computed_velocity: np.ndarray) -> bool:
        """
        Check if the computed trajectory violates velocity constraints.

        Args:
            max_computed_velocity: Maximum computed velocity for each joint

        Returns:
            bool: True if feasible, False otherwise
        """
        if np.any(np.abs(max_computed_velocity) > self.max_velocity):
            infeasible_joints = np.where(np.abs(max_computed_velocity) > self.max_velocity)[0]
            print(f"Velocity limits exceeded for joints: {infeasible_joints}")
            print(f"Max velocities: {max_computed_velocity}")
            print(f"Velocity limits: {self.max_velocity}")
            return False
        return True

    def _solve_tridiagonal(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Solve tridiagonal system using Thomas algorithm."""
        n = len(B)
        # Forward sweep
        for i in range(1, n):
            w = A[i] / B[i-1]
            B[i] -= w * C[i-1]
            D[i] -= w * D[i-1]
        # Back substitution
        x = np.zeros(n)
        x[-1] = D[-1] / B[-1]
        for i in range(n-2, -1, -1):
            x[i] = (D[i] - C[i] * x[i+1]) / B[i]
        return x

    def _calculate_yaw_from_velocity(self, velocities: np.ndarray) -> np.ndarray:
        """Calculate yaw angles from velocity vectors with smoothing."""
        # Calculate raw yaw angles
        vx = velocities[:, 0]
        vy = velocities[:, 1]
        raw_yaw = np.arctan2(vy, vx)
        
        # Unwrap angles to avoid 2π jumps
        unwrapped_yaw = np.unwrap(raw_yaw)
        
        # Smooth yaw with rate limiting
        return self._smooth_and_limit_yaw(unwrapped_yaw)

    def _smooth_and_limit_yaw(self, yaw: np.ndarray) -> np.ndarray:
        """Apply moving average and rate limiting to yaw angles."""
        # Moving average filter
        window_size = min(5, len(yaw))
        if window_size > 1:
            yaw_smooth = np.convolve(yaw, np.ones(window_size)/window_size, mode='same')
        else:
            yaw_smooth = yaw.copy()
        
        # Yaw rate limiting
        delta_t = self.dt
        for i in range(1, len(yaw_smooth)):
            delta_yaw = yaw_smooth[i] - yaw_smooth[i-1]
            max_delta = self.max_yaw_rate * delta_t
            if abs(delta_yaw) > max_delta:
                yaw_smooth[i] = yaw_smooth[i-1] + np.sign(delta_yaw)*max_delta
        
        return yaw_smooth

    def generate_trajectory(self) -> TrajectoryState:
        """Generate cubic spline trajectory through all waypoints."""
        feasable = False
        while(not feasable):
            self.time_points = np.arange(self.times[0], self.times[-1] + self.dt, self.dt)
            self.time_points = np.clip(self.time_points, self.times[0], self.times[-1])
            num_points = len(self.time_points)
            positions = np.zeros((num_points, self.num_joints))
            velocities = np.zeros_like(positions)
            accelerations = np.zeros_like(positions)

            for joint in range(self.num_joints):
                wp_joint = np.array([wp[joint] for wp in self.waypoints])
                times_joint = self.times
                n_segments = len(wp_joint) - 1

                if n_segments == 0:
                    positions[:, joint] = wp_joint[0]
                    continue

                # Single segment case
                if n_segments == 1:
                    duration = times_joint[1] - times_joint[0]
                    t_segment = self.time_points - times_joint[0]
                    valid_mask = t_segment <= duration
                    t_segment = t_segment[valid_mask]

                    # Calculate coefficients
                    a0 = wp_joint[0]
                    a1 = 0
                    a2 = 3*(wp_joint[1] - a0)/duration**2
                    a3 = -2*(wp_joint[1] - a0)/duration**3

                    positions[valid_mask, joint] = a0 + a1*t_segment + a2*t_segment**2 + a3*t_segment**3
                    velocities[valid_mask, joint] = a1 + 2*a2*t_segment + 3*a3*t_segment**2
                    accelerations[valid_mask, joint] = 2*a2 + 6*a3*t_segment
                    continue

                # Multiple segments case
                delta_t = np.diff(times_joint)
                delta_p = np.diff(wp_joint)
                m = n_segments - 1  # Number of interior via points

                # Setup tridiagonal system
                A = np.zeros(m)
                B = np.zeros(m)
                C = np.zeros(m)
                D = np.zeros(m)

                for i in range(m):
                    prev_seg = i
                    next_seg = i + 1
                    A[i] = 2 / delta_t[prev_seg]
                    B[i] = 4/delta_t[prev_seg] + 4/delta_t[next_seg]
                    C[i] = 2 / delta_t[next_seg]
                    D[i] = 6 * (delta_p[next_seg]/delta_t[next_seg]**2 + 
                                delta_p[prev_seg]/delta_t[prev_seg]**2)

                # Solve for interior velocities
                v_interior = self._solve_tridiagonal(A, B, C, D)
                velocities_via = np.zeros(n_segments + 1)
                velocities_via[1:-1] = v_interior

                # Generate trajectory for each segment
                for seg in range(n_segments):
                    t_start = times_joint[seg]
                    t_end = times_joint[seg+1]
                    duration = delta_t[seg]
                    p_start = wp_joint[seg]
                    p_end = wp_joint[seg+1]
                    v_start = velocities_via[seg]
                    v_end = velocities_via[seg+1]
                    # print(joint, seg, t_start, t_end, p_start, p_end, v_start, v_end, duration)
                    # Calculate coefficients
                    a2 = (3*(p_end - p_start) - (2*v_start + v_end)*duration) / duration**2
                    a3 = (v_end - v_start - 2*a2*duration) / (3*duration**2)

                    # Calculate trajectory points
                    mask = (self.time_points >= t_start) & (self.time_points <= t_end)
                    zero_indices = np.where(mask == 0)[0]

                    t_seg = self.time_points[mask] - t_start

                    positions[mask, joint] = p_start + v_start*t_seg + a2*t_seg**2 + a3*t_seg**3
                    velocities[mask, joint] = v_start + 2*a2*t_seg + 3*a3*t_seg**2
                    accelerations[mask, joint] = 2*a2 + 6*a3*t_seg

            # Check velocity constraints
            max_velocities = np.max(np.abs(velocities), axis=0)
            if not self.check_velocity_feasibility(max_velocities):
                # raise ValueError("Trajectory violates velocity constraints")
                self.duration*=self.duration_factor
                print("Recalculating with total duration:",self.duration)
                self.times = self.get_times_from_duration()
                continue
            feasable = True
            print("Executing with durations:",self.times)

        yaw = self._calculate_yaw_from_velocity(velocities)

        return TrajectoryState(positions, velocities, accelerations, yaw, self.time_points)

def test_linear_planner():
    """Test function to demonstrate both constant and trapezoidal velocity profiles."""
    # Example joint positions
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 0.5, -0.5])
    duration = 2.0
    
    # Test both profiles
    planners = [
        LinearTrajectoryPlanner(start, end, duration, use_trapezoidal=False),
        LinearTrajectoryPlanner(start, end, duration, use_trapezoidal=True, 
                              acceleration_time=0.3)
    ]
    
    for planner in planners:
        profile_type = "trapezoidal" if planner.use_trapezoidal else "constant"
        print(f"\nTesting Linear Planner with {profile_type} velocity profile")
        
        trajectory = planner.generate_trajectory()
        
        print(f"Start position: {trajectory.positions[:, 0]}")
        print(f"End position: {trajectory.positions[:, -1]}")
        print(f"Max velocity: {np.max(np.abs(trajectory.velocities), axis=1)}")
        print(f"Max acceleration: {np.max(np.abs(trajectory.accelerations), axis=1)}")

def test_cubic_planner():
    """Test function to demonstrate velocity constraint handling."""
    # Example joint positions
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 0.5, -0.5])
    duration = 1.0
    
    # Set velocity limits
    max_velocity = np.array([0.3, 0.3, 0.3])  # Intentionally low for testing
    
    try:
        # Test cubic planner
        print("\nTesting Cubic Planner with velocity constraints")
        cubic_planner = CubicTrajectoryPlanner(start, end, duration, max_velocity=max_velocity)
        cubic_traj = cubic_planner.generate_trajectory()
        print("Cubic trajectory generation successful")
    except ValueError as e:
        print(f"Cubic planner error: {str(e)}")

# if __name__ == "__main__":
#     test_cubic_planner()
#     test_linear_planner()