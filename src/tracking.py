"""
Tracking-detector simulation and momentum reconstruction.

Simulation of charged-particle trajectories through a segmented tracking
detector with a homogeneous magnetic field, reconstruction of the track
parameters by weighted least squares, and reconstruction of the transverse
momentum from the bending angle across the magnet.

Detector geometry and cell size are module-level constants; the quantities that
are varied in the experiments (pT, B, number of trajectories) are passed as
arguments.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from scipy.stats import norm

# --------------------------------------------------------------------------
# Detector geometry (all lengths in metres, angles in radians)
# --------------------------------------------------------------------------

n_beforeB = 5                             # detection planes before the magnet
n_afterB = 3                              # detection planes after the magnet
D_z = 2e-2                                # plane spacing along z
L = 10e-2                                 # length of the magnetic field along z
cell_width = 0.0005                       # 500 um cell pitch
n_planes = n_beforeB                      # planes used in the field-free part

z_start_after_B = n_beforeB * D_z + L     # z of the first plane after the magnet
z_begin_magnetic_field = 0.1              # z where the field starts
z_end_magnetic_field = 0.2                # z where the field ends


# --------------------------------------------------------------------------
# Part A - tracking resolution (field-free)
# --------------------------------------------------------------------------

# To use curve_fit, we need to have a function defined, so we use a function to calculate the x_positions
def trajectory_fun(z_position, x0, angle): 
    return x0 + z_position * np.tan(angle) # Calculating the vertical position using trigonometry

def trajectory_fun_linear(z_position, x0, slope): 
    return x0 + z_position * slope # Calculating the vertical position using trigonometry

def manual_least_squares(z, x, sigma):
    """Perform a weighted least squares linear fit to data (z, x),
    with constant uncertainty sigma (cell_width / np.sqrt(12)) on each x.
    Returns: x0_fit, slope_fit, std_x0, std_slope
    """
    w = np.ones_like(z) / sigma**2  # weights (array)

    # We want to minimise the weighted sum of residuals. 
    # For this we derive for x_0 and the slope and get two normal equations: 
    # sum( w_i * [x_i - x_0 - m*z_i]) = 0
    # sum( w_i * z_i * [x_i - x_0 - m*z_i]) = 0

    # Calculate the values of the important sums
    S_w = np.sum(w)
    S_wz = np.sum(w * z)
    S_wx = np.sum(w * x)
    S_wz2 = np.sum(w * z**2)
    S_wzx = np.sum(w * z * x)

    # Then 
    # S_wz = x_0 * S_w + m * S_wz = (S_w, S_wz) * (x_0, m)^T
    # S_wzx = x_0 * S_wz + m * S_wz2 = (S_wz, S_wz2) * (x_0, m)^T

    # We now solve this by multiplying the inverse of the matrix from the left to both sides (using the adjoint)
    # 1/det * (S_wz2 * S_wz - S_wz * S_wzx) = x0
    # 1/det * (-S_wz * S_wz + S_w * S_wzx) = slope = m

    # Calculate the determinant
    determinant_of_mat = S_w * S_wz2 - S_wz**2
    if determinant_of_mat <= 0:
        raise ValueError(f"Non-positive Delta encountered: {determinant_of_mat}")

    # Use the equations from above
    x0_fit = (S_wz2 * S_wx - S_wz * S_wzx) / determinant_of_mat
    slope_fit = (S_w * S_wzx - S_wz * S_wx) / determinant_of_mat

    # Covariance matrix is inverse of matrix
    # [[S_w, S_wz],
    # [S_wz, S_wz2]]

    # Again calculated with adjoint:
    # 1/det * [[S_wz^2, - S_wz],
    #          [- S_wz, S_w]]

    # Use the diagonal elements for the variance -> take root for std_err
    std_x0 = np.sqrt(S_wz2 / determinant_of_mat)
    std_slope = np.sqrt(S_w / determinant_of_mat)

    return x0_fit, slope_fit, std_x0, std_slope

def manual_least_squares_after(z, x, sigma):
    z = z - 0.2
    w = np.ones_like(z) / sigma**2  # weights (array)

    # We want to minimise the weighted sum of residuals. 
    # For this we derive for x_0 and the slope and get two normal equations: 
    # sum( w_i * [x_i - x_0 - m*z_i]) = 0
    # sum( w_i * z_i * [x_i - x_0 - m*z_i]) = 0

    # Calculate the values of the important sums
    S_w = np.sum(w)
    S_wz = np.sum(w * z)
    S_wx = np.sum(w * x)
    S_wz2 = np.sum(w * z**2)
    S_wzx = np.sum(w * z * x)

    # Then 
    # S_wz = x_0 * S_w + m * S_wz = (S_w, S_wz) * (x_0, m)^T
    # S_wzx = x_0 * S_wz + m * S_wz2 = (S_wz, S_wz2) * (x_0, m)^T

    # We now solve this by multiplying the inverse of the matrix from the left to both sides (using the adjoint)
    # 1/det * (S_wz2 * S_wz - S_wz * S_wzx) = x0
    # 1/det * (-S_wz * S_wz + S_w * S_wzx) = slope = m

    # Calculate the determinant
    determinant_of_mat = S_w * S_wz2 - S_wz**2
    if determinant_of_mat <= 0:
        raise ValueError(f"Non-positive Delta encountered: {determinant_of_mat}")

    # Use the equations from above
    x0_fit = (S_wz2 * S_wx - S_wz * S_wzx) / determinant_of_mat
    slope_fit = (S_w * S_wzx - S_wz * S_wx) / determinant_of_mat

    # Covariance matrix is inverse of matrix
    # [[S_w, S_wz],
    # [S_wz, S_wz2]]

    # Again calculated with adjoint:
    # 1/det * [[S_wz^2, - S_wz],
    #          [- S_wz, S_w]]

    # Use the diagonal elements for the variance -> take root for std_err
    std_x0 = np.sqrt(S_wz2 / determinant_of_mat)
    std_slope = np.sqrt(S_w / determinant_of_mat)

    return x0_fit, slope_fit, std_x0, std_slope

def single_trajectory(n_planes = 5, delta_z = 0.02, cell_width = 0.0005):
    x0_true = np.random.normal(0, 0.001)
    s0_true = np.random.normal(0, 0.1)

    slope_true = np.tan(s0_true)

    z_positions = np.arange(1, n_planes + 1) * delta_z
    x_positions = trajectory_fun_linear(z_positions, x0_true, slope_true) 

    # Calculate which cell was hit
    cells_hit = np.floor(x_positions / cell_width) # Casting to int (.astype(int)) is optional; float cell indices suffice for midpoint computation; Also works if the x_position is negative -> e.g. cell "-1"
    cells_hit_middle = cells_hit * cell_width + cell_width / 2

    # Calculate the uncertainty
    std_hit = cell_width / np.sqrt(12) # Modelled as explained by Valeriia

    # Add Gaussian noise to simulate measurement uncertainty (realistic detector resolution)
    # Increases the uncertainty a lot - maybe not what we want
    # cells_hit_middle += np.random.normal(0, std_hit, size=cells_hit_middle.shape)

    """
    # Use curve_fit, popt = optimal values for the parameters s.t. rss is minimised, pcov = estimated approximate covariance of popt
    popt, pcov = curve_fit(trajectory_fun_linear, z_positions, cells_hit_middle, p0 = None, sigma = std_hit, absolute_sigma = True)

    x0_reco, slope_reco = popt
    std_x0_reco, std_slope_reco = np.sqrt(np.diag(pcov)) # Uncertainties on the fitted parameters, see documentation (used later for pull distributions)
    """

    # Use weighted linear regression:
    x0_reco, slope_reco, std_x0_reco, std_slope_reco = manual_least_squares(z_positions, cells_hit_middle, std_hit)

    s0_reco = np.arctan(slope_reco)

    # We now do error propagation by using y = f(x) -> sigma_y = abs(df/dx (x_mean)) * sigma_x
    std_s0_reco = std_slope_reco / (1 + slope_reco**2) 

    return s0_true, s0_reco, std_s0_reco, slope_true, slope_reco, std_slope_reco, x0_true, x0_reco, std_x0_reco, z_positions, x_positions, cells_hit_middle, std_hit

# --------------------------------------------------------------------------
# Part B - momentum resolution (with magnetic field)
# --------------------------------------------------------------------------

def trajectory_before_B(z, starting_xpoint, starting_angle):
    return starting_xpoint + np.tan(starting_angle) * z

def trajectory_before_B_linear(z, starting_xpoint, slope): 
    return starting_xpoint + z * slope

def generating_x0(n):
    mean_x = 0
    std_x = 1e-3             # in metres
    x0 = np.random.normal(loc=mean_x, scale=std_x, size=n) # Generate n random starting positions
    
    #for i in range(n):
        #x0.append(stats.norm.rvs(loc=mean_x, scale=std_x))
    
    return x0

def generating_s0(n):
    mean_s = 0
    std_s = 0.1             # rad
    s0 = np.random.normal(loc=mean_s, scale=std_s, size=n)
    # Generate n random angles
    #for i in range(n):
        #s0.append(stats.norm.rvs(loc=mean_s, scale=std_s))
    return s0

def circular_path_center_coordinates(x_start, z_start, radius, slope_first_line, charge):
    
    s0_angle = np.arctan(slope_first_line)
    
    # Circle center coordinates
    if slope_first_line >= 0 and charge > 0:
        z_center = z_start + radius * np.abs(np.sin(s0_angle))
        x_center = x_start - radius * np.abs(np.cos(s0_angle))
    elif slope_first_line < 0 and charge < 0:
        z_center = z_start + radius * np.abs(np.sin(s0_angle))
        x_center = x_start + radius * np.abs(np.cos(s0_angle))
    elif slope_first_line >= 0 and charge < 0:
        z_center = z_start - radius * np.abs(np.sin(s0_angle))
        x_center = x_start + radius * np.abs(np.cos(s0_angle))
    else: # slope_first_line >= 0 and charge < 0:
        z_center = z_start - radius * np.abs(np.sin(s0_angle))
        x_center = x_start - radius * np.abs(np.cos(s0_angle))
    
    
    return z_center, x_center

def circular_path_coefficients(x_start, z_start, radius, L, slope_first_line, charge, B, pT_true):
    
    z_center, x_center = circular_path_center_coordinates(x_start, z_start, radius, slope_first_line, charge)
        
    # Arc's initial angle
    theta_start = np.arctan2(x_start - x_center, z_start - z_center) 

    # Reset to find theta_end so that z_end - z_start = L
    def z_arc_length_error(theta_end):
        z_end = z_center + radius * np.cos(theta_end)
        return (z_end - z_start) - L
    
    # Try to find theta_end numerically
    bracket = [theta_start - 1.25, theta_start + 1.25]
    f_low = z_arc_length_error(bracket[0])
    f_high = z_arc_length_error(bracket[1])

    if f_low * f_high < 0:
        sol = root_scalar(z_arc_length_error, bracket=bracket, method='brentq')
        theta_end = sol.root
    else:
        #s0_angle = np.arctan(slope_first_line)
        # Fallback: estimate angle from straight path assumption
        theta = L * charge * B / pT_true # Originally implemented: L / (radius * np.cos(s0_angle))
        theta_end = theta_start - theta # np.sign(charge) * theta

    # Circular path's points
    theta_arc = np.linspace(theta_start, theta_end, 100)
    z_arc = z_center + radius * np.cos(theta_arc)
    x_arc = x_center + radius * np.sin(theta_arc)
    
    # Return the points of the arc
    return z_arc, x_arc

def starting_angle_after_B(starting_zpoint, starting_xpoint, z_center, x_center):
    # Calculate the slope of the line after the magnetic field
    # Take the normal of the slope to the last point of the radius
    slope_last_point_of_arc = (starting_xpoint - x_center) / (starting_zpoint - z_center)
    slope_after_B = -1 / slope_last_point_of_arc
    
    return np.arctan(slope_after_B)

def trajectory_after_B(z, starting_xpoint, starting_angle, z0_after_magnet):
    return starting_xpoint + np.tan(starting_angle) * (z - z0_after_magnet)

def trajectory_after_B_linear(z, starting_xpoint, slope): 
    return starting_xpoint + z * slope

def compute_pT_reco_classic(angle_before_B_field, angle_after_B_field, angle_before_B_field_err, angle_after_B_field_err, q, B, L):
    
    theta = angle_after_B_field - angle_before_B_field # np.abs(angle_after_B_field - angle_before_B_field)
    
    #delta_theta = angle_after_B_field - angle_before_B_field
    #theta2 = np.abs(np.arctan2(np.sin(delta_theta), np.cos(delta_theta)))

    # print(theta, theta2)

    pT_reco = np.abs((q * B * L) / theta)
    
    # Error propagation for pT_reco
    pT_reco_err = np.sqrt((angle_before_B_field_err * (q * B * L) / theta**2)**2 + (angle_after_B_field_err * (q * B * L) / theta**2)**2)
    
    # This is the same, just written differently
    #theta_err = np.sqrt(angle_before_B_field_err**2 + angle_after_B_field_err**2) 
    #pT_reco_err = np.abs((q * B * L) / theta**2) * theta_err

    #print(np.sign(np.abs(theta) - 0.16666666666666666), np.sign(pT_reco - 0.3)) # Strong inverse correlation (makes mathematically sense), as pT_reco too small -> theta too big -> angle_after_B_field too big or angle_before_B_field too small
    
    return pT_reco, pT_reco_err

def compute_pT_reco_mc(angle_before_B_field, angle_after_B_field, angle_before_B_field_err, angle_after_B_field_err, q, B, L, n_samples=10000):
    # Reconstruct pT using Monte Carlo sampling

    # Monte Carlo sampling of angles
    theta_before_samples = np.random.normal(angle_before_B_field, angle_before_B_field_err, n_samples)
    theta_after_samples = np.random.normal(angle_after_B_field, angle_after_B_field_err, n_samples)

    # As before
    theta_samples = theta_after_samples - theta_before_samples

    # Avoid division by 0
    valid = np.abs(theta_samples) > 1e-6
    theta_samples = theta_samples[valid]
    pT_samples = np.abs((q * B * L) / theta_samples)

    # Mean and standard deviation as reconstructed value and uncertainty
    pT_reco = np.mean(pT_samples)
    pT_reco_err = np.std(pT_samples)

    return pT_reco, pT_reco_err

def simulate_trajectories(nr_trajectories, pT_true, B, plot_circular_path = True):
    
    curvature_radius = pT_true / B
    
    x0 = generating_x0(nr_trajectories)
    s0 = generating_s0(nr_trajectories)
    rng = np.random.default_rng()
    q = rng.choice(a=np.array([-1, 1]), size=nr_trajectories)
    #m_before_B = np.tan(s0)                                 # slopes
    
    # Returning values (lists)
    z_positions_before_B_list = []
    z_positions_after_B_list = []
    x_positions_before_B_list = []
    x_positions_after_B_list = []
    grid_before_B_list = []
    grid_after_B_list = []
    true_traj_z_before_B_list = []
    true_traj_z_after_B_list = []
    reco_traj_z_before_B_list = []
    reco_traj_z_after_B_list = []
    cells_hit_middle_before_B_list = []
    cells_hit_middle_after_B_list = []
    std_hit_list = []
    s_after_B_list = []

    s0_reco_before_B_list = []
    s0_reco_after_B_list = []

    std_x0_reco_before_B_list = []
    std_s0_reco_before_B_list = []
    std_x0_reco_after_B_list = []
    std_s0_reco_after_B_list = []
    

    for i in range(nr_trajectories):
        
        q_true = q[i]
        x0_true = x0[i]             # x0 is a list, the element is the float value
        s0_true = s0[i]             # s0 is a list, the element is the float value

        # Calculate the z and x positions of the planes 
        # To use curve_fit, we need to have a function defined, so we use a function to calculate the x_positions
        z_positions_before_B = np.arange(1, n_beforeB + 1) * D_z
        slope_true = np.tan(s0_true)
        x_positions_before_B = trajectory_before_B_linear(z_positions_before_B, x0_true, slope_true)

        z_positions_after_B = np.arange(n_beforeB + L/D_z, n_afterB + n_beforeB + L/D_z) * D_z # positions in z after the magnetic field
        z0_arc = n_beforeB * D_z                # z coordinate of the starting point of the arc
        x0_arc = x0_true + slope_true * z0_arc       # x coordinates of the starting point of the arc

        z_arc, x_arc = circular_path_coefficients(x0_arc, z0_arc, curvature_radius, L, slope_true, q_true, B, pT_true)
        
        if plot_circular_path:
            if i == 0:
                plt.plot(z_arc, x_arc*1000, label="Trajectory during the magnetic field B", linestyle='-', color="#654321")
            else:
                plt.plot(z_arc, x_arc*1000, linestyle='-', color="#654321")

        z_center, x_center = circular_path_center_coordinates(x0_arc, z0_arc, curvature_radius, slope_true, q_true)
        s_after_B = starting_angle_after_B(z_arc[-1], x_arc[-1], z_center, x_center)
        x_positions_after_B = trajectory_after_B(z_positions_after_B, x_arc[-1], s_after_B, z_arc[-1])  # positions in z after the magnetic field

        # Calculate which cell was hit
        cells_hit_before_B = np.floor(x_positions_before_B / cell_width)
        cells_hit_middle_before_B = cells_hit_before_B * cell_width + cell_width / 2
        cells_hit_after_B = np.floor(x_positions_after_B / cell_width)
        cells_hit_middle_after_B = cells_hit_after_B * cell_width + cell_width / 2
        # Calculate the uncertainty
        std_hit = cell_width / np.sqrt(12) # Modelled as explained by Valeriia
        
        """
        # Use curve_fit, popt = optimal values for the parameters s.t. rss is minimised, pcov = estimated approximate covariance of popt
        popt_before_B, pcov_before_B = curve_fit(trajectory_before_B_linear, z_positions_before_B, cells_hit_middle_before_B, p0 = None, sigma = std_hit, absolute_sigma = True)
        popt_after_B, pcov_after_B = curve_fit(trajectory_after_B_linear, z_positions_after_B, cells_hit_middle_after_B, p0 = None, sigma = std_hit, absolute_sigma = True)

        x0_reco_before_B, slope_reco_before_B = popt_before_B
        std_x0_reco_before_B, std_slope_reco_before_B = np.sqrt(np.diag(pcov_before_B)) # Uncertainties on the fitted parameters, see documentation (used later for pull distributions)

        x0_reco_after_B, slope_reco_after_B = popt_after_B
        std_x0_reco_after_B, std_slope_reco_after_B = np.sqrt(np.diag(pcov_after_B))
        """

        x0_reco_before_B, slope_reco_before_B, std_x0_reco_before_B, std_slope_reco_before_B = manual_least_squares(z_positions_before_B, cells_hit_middle_before_B, std_hit)
        x0_reco_after_B, slope_reco_after_B, std_x0_reco_after_B, std_slope_reco_after_B = manual_least_squares_after(z_positions_after_B, cells_hit_middle_after_B, std_hit)
        
        #print(x0_reco_after_B)  # This calculates the x0_reco at the beginning of the track -> for z = 0.0, not z = 0.2
        
        s0_reco_before_B = np.arctan(slope_reco_before_B)
        s0_reco_after_B = np.arctan(slope_reco_after_B)

        # We now do error propagation by using y = f(x) -> sigma_y = abs(df/dx (x_mean)) * sigma_x
        std_s0_reco_before_B = std_slope_reco_before_B / (1 + slope_reco_before_B**2) 
        std_s0_reco_after_B = std_slope_reco_after_B / (1 + slope_reco_after_B**2)

        # print(np.sign(z_arc[-1] - 0.2), np.sign(s_after_B - s0_reco_after_B)) # Does not seem to be the problem

        # Plotting details
        grid_before_B = np.linspace(0, z_positions_before_B[-1] + D_z/2, num = 1001)
        true_traj_z_before_B = trajectory_before_B(grid_before_B, x0_true, s0_true)
        reco_traj_z_before_B = trajectory_before_B_linear(grid_before_B, x0_reco_before_B, slope_reco_before_B)

        grid_after_B = np.linspace(z_positions_after_B[0], z_positions_after_B[-1] + D_z/2, num=1001)
        true_traj_z_after_B = trajectory_after_B(grid_after_B, x_arc[-1], s_after_B, z_arc[-1])
        reco_traj_z_after_B = trajectory_after_B_linear(grid_after_B - 0.2, x0_reco_after_B, slope_reco_after_B)
        
        # Appending the values to the lists
        z_positions_before_B_list.append(z_positions_before_B)
        z_positions_after_B_list.append(z_positions_after_B)
        x_positions_before_B_list.append(x_positions_before_B)
        x_positions_after_B_list.append(x_positions_after_B)
        grid_before_B_list.append(grid_before_B)
        grid_after_B_list.append(grid_after_B)
        true_traj_z_before_B_list.append(true_traj_z_before_B)
        true_traj_z_after_B_list.append(true_traj_z_after_B)
        reco_traj_z_before_B_list.append(reco_traj_z_before_B)
        reco_traj_z_after_B_list.append(reco_traj_z_after_B)
        cells_hit_middle_before_B_list.append(cells_hit_middle_before_B)
        cells_hit_middle_after_B_list.append(cells_hit_middle_after_B)
        std_hit_list.append(std_hit)
        s_after_B_list.append(s_after_B)

        s0_reco_before_B_list.append(s0_reco_before_B)
        s0_reco_after_B_list.append(s0_reco_after_B)
        
        std_x0_reco_before_B_list.append(std_x0_reco_before_B)
        std_s0_reco_before_B_list.append(std_s0_reco_before_B)
        std_x0_reco_after_B_list.append(std_x0_reco_after_B)
        std_s0_reco_after_B_list.append(std_s0_reco_after_B)
    
    return (z_positions_before_B_list, z_positions_after_B_list, x_positions_before_B_list, x_positions_after_B_list, grid_before_B_list, grid_after_B_list, true_traj_z_before_B_list, true_traj_z_after_B_list, reco_traj_z_before_B_list, reco_traj_z_after_B_list, cells_hit_middle_before_B_list, cells_hit_middle_after_B_list, std_hit_list, 
            s_after_B_list, s0_reco_before_B_list, s0_reco_after_B_list, std_x0_reco_before_B_list, std_s0_reco_before_B_list, std_x0_reco_after_B_list, std_s0_reco_after_B_list, s0, q)

def display_simulation(z_positions_before_B, z_positions_after_B, x_positions_before_B, x_positions_after_B, grid_before_B, grid_after_B, true_traj_z_before_B, true_traj_z_after_B, reco_traj_z_before_B, reco_traj_z_after_B, cells_hit_middle_before_B, cells_hit_middle_after_B, std_hit, s_after_B, std_x0_reco_before_B, std_s0_reco_before_B, std_x0_reco_after_B, std_s0_reco_after_B, s0, q, nr_trajectories, pT_true, B):
    
    for z in z_positions_before_B[0]:
        plt.axvline(x=z, color="darkblue", linestyle='-', linewidth=1.5, zorder=1)  # Detector planes before B
    for z in z_positions_after_B[0]:
        plt.axvline(x=z, color="darkblue", linestyle='-', linewidth=1.5, zorder=1)  # Detector planes after B

    for i in range(nr_trajectories):

        if i == 0:
            plt.plot(grid_before_B[i], 1000 * true_traj_z_before_B[i], "k--", label="True, original trajectory before B field")
            plt.plot(grid_before_B[i], 1000 * reco_traj_z_before_B[i], "r-", label="Reconstructed trajectory before B field")
            plt.scatter(z_positions_before_B[i], 1000 * x_positions_before_B[i], color="g", label="True, original hit positions before B field", zorder=3)
            plt.errorbar(z_positions_before_B[i], 1000 * cells_hit_middle_before_B[i], yerr=1000 * std_hit[i], fmt="o", color="darkred", linewidth=2.5, label="Recorded hit positions before B field")
            plt.plot(grid_after_B[i], 1000 * true_traj_z_after_B[i], "b--", label="True, original trajectory after B field")
            plt.plot(grid_after_B[i], 1000 * reco_traj_z_after_B[i], "m-", label="Reconstructed trajectory after B field")
            plt.scatter(z_positions_after_B[i], 1000 * x_positions_after_B[i], color="c", label="True, original hit positions after B field", zorder=3)
            plt.errorbar(z_positions_after_B[i], 1000 * cells_hit_middle_after_B[i], yerr=1000 * std_hit[i], fmt="o", color="darkorange", linewidth=2.5, label="Recorded hit positions after B field")
        else:
            plt.plot(grid_before_B[i], 1000 * true_traj_z_before_B[i], "k--")
            plt.plot(grid_before_B[i], 1000 * reco_traj_z_before_B[i], "r-")
            plt.scatter(z_positions_before_B[i], 1000 * x_positions_before_B[i], color="g", zorder=3)
            plt.errorbar(z_positions_before_B[i], 1000 * cells_hit_middle_before_B[i], yerr=1000 * std_hit[i], fmt="o", color="darkred", linewidth=2.5)
            plt.plot(grid_after_B[i], 1000 * true_traj_z_after_B[i], "b--")
            plt.plot(grid_after_B[i], 1000 * reco_traj_z_after_B[i], "m-")
            plt.scatter(z_positions_after_B[i], 1000 * x_positions_after_B[i], color="c", zorder=3)
            plt.errorbar(z_positions_after_B[i], 1000 * cells_hit_middle_after_B[i], yerr=1000 * std_hit[i], fmt="o", color="darkorange", linewidth=2.5)
        
    plt.axvspan(z_begin_magnetic_field, z_end_magnetic_field, color='orange', alpha=0.3, label = "Magnetic field B", zorder = 0)
    plt.xlabel("z [m]")
    plt.ylabel("x [mm]")
    plt.grid()
    plt.title(f"True vs. Reconstructed trajectory for {nr_trajectories} trajectories, true $p_T$ = {pT_true} GeV, B = {B}")
    plt.legend()
    plt.tight_layout()
    filename = f"proj1_momentum_res_{nr_trajectories}_trajectories.png"
    plt.savefig(filename)
    plt.show()
    plt.close()    

# --------------------------------------------------------------------------
# Residual and pull analysis
# --------------------------------------------------------------------------

def residuals_and_pulls(s_after_B, nr_trajectories, s0_reco_before_B, s0_reco_after_B, std_s0_reco_before_B, std_s0_reco_after_B, q_list, B, L, pT_true, pt_calculation_way = "classic"):

    # Compute pT_reco
    pT_reco_list = []
    pT_reco_err_list = []

    for i in range(nr_trajectories):
        if pt_calculation_way == "classic":
            pT_reco, pT_reco_err = compute_pT_reco_classic(s0_reco_before_B[i], s0_reco_after_B[i], std_s0_reco_before_B[i], std_s0_reco_after_B[i], q_list[i], B, L)
        elif pt_calculation_way == "mc":
            pT_reco, pT_reco_err = compute_pT_reco_mc(s0_reco_before_B[i], s0_reco_after_B[i], std_s0_reco_before_B[i], std_s0_reco_after_B[i], q_list[i], B, L)
        else:
            raise ValueError("Unknown calculation way for pT")
        
        pT_reco_list.append(pT_reco) # * const.c / 1e9
        pT_reco_err_list.append(pT_reco_err) # * const.c / 1e9

        #print(np.sign(s_after_B[i] - s0_reco_after_B[i]), np.sign(pT_reco - 0.3)) # There does not seem to be a correlation

    # Convert the result in GeV
    pT_reco = np.array(pT_reco_list)   # GeV
    pT_reco_err = np.array(pT_reco_err_list) # GeV

    # Residuals
    resi_pT = pT_reco - pT_true

    # pull = (reconstructed_quantity - generated_quantity)/uncertainty_on_reconstructed_quantity
    pulls_pT = (pT_reco - pT_true)/pT_reco_err

    finite_resi_pT = resi_pT[np.isfinite(resi_pT) & (resi_pT < 1e11)]
    finite_pulls_pT = pulls_pT[np.isfinite(resi_pT) & (resi_pT < 1e11)]

    # Taking the mean and standard deviation for the residuals
    mean_resi_pT = np.mean(finite_resi_pT)
    std_resi_pT = np.std(finite_resi_pT)

    # Uncertainties of estimators of residuals assuming gaussian distribution
    mean_resi_pT_error = std_resi_pT/np.sqrt(nr_trajectories)
    std_resi_pT_error = std_resi_pT/np.sqrt(2 * nr_trajectories - 2)

    # Taking the mean and standard deviation for the pulls
    mean_pulls_pT = np.mean(finite_pulls_pT)
    std_pulls_pT = np.std(finite_pulls_pT)

    # Uncertainties of estimators of pulls assuming gaussian distribution
    mean_pulls_pT_error = std_pulls_pT/np.sqrt(nr_trajectories)
    std_pulls_pT_error = std_pulls_pT/np.sqrt(2 * nr_trajectories - 2)

    # 2x2 subplot figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Plot the distribution of the residuals as histograms
    axs[0].hist(finite_resi_pT, bins = 25, color = "blue", edgecolor = "black")
    axs[0].set_title(f"Residual: $p_T$\nμ={mean_resi_pT:.4f} GeV/c ± {mean_resi_pT_error:.5f} GeV/c, σ={std_resi_pT:.3f} GeV/c ± {std_resi_pT_error:.5f} GeV/c")
    axs[0].set_xlabel("pT_reco - pT_true [GeV/c]")
    axs[0].grid()

    # Plot the distribution of the pulls as histograms with overlayed Gaussian distributions
    axs[1].hist(pulls_pT, bins=25, density=True, color="blue", edgecolor="black", label="Pull distribution")
    x_vals = np.linspace(min(pulls_pT), max(pulls_pT), 500)
    axs[1].plot(x_vals, norm.pdf(x_vals, 0, 1), 'r--', label="N(0,1)")
    axs[1].plot(x_vals, norm.pdf(x_vals, mean_pulls_pT, std_pulls_pT), 'g-', label=f"Fit N({mean_pulls_pT:.2f}, {std_pulls_pT:.2f})")
    axs[1].set_title(f"Pulls: $p_T$\nμ={mean_pulls_pT:.3f} ± {mean_pulls_pT_error:.3f}, σ={std_pulls_pT:.3f} ± {std_pulls_pT_error:.3f}")
    axs[1].set_xlabel("Pull: (pT_reco - pT_true)/σ_pT_reco")
    axs[1].legend()
    axs[1].grid()

    plt.suptitle(f"Histograms of Residuals and Pulls of $p_T$ when $p_T$ = {pT_true} GeV/c, B = {B} T", fontsize = 14)

    plt.tight_layout()
    filename = f"Residuals_Pulls_{pT_true}_{B}_trajectories.png"
    plt.savefig(filename)
    plt.show()
