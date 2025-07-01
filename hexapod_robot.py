import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d import Axes3D

def get_world_to_base(orientation, position):
  R, P, Y = orientation
  x, y ,z = position

  Rz = np.array([[np.cos(Y), -np.sin(Y), 0],
                 [np.sin(Y),  np.cos(Y), 0],
                 [        0,          0, 1]])
  
  Ry = np.array([[ np.cos(P), 0, np.sin(P)],
                 [        0,  1,         0],
                 [-np.sin(P), 0, np.cos(P)]])
  
  Rx = np.array([[1,         0,          0],
                 [0, np.cos(R), -np.sin(R)],
                 [0, np.sin(R),  np.cos(R)]])

  R = Rz @ Ry @ Rx
  p = np.array([x, y, z]).reshape(3, 1)
  return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def get_base_to_world(pose):
  Twb = get_world_to_base(pose[0], pose[1])
  R = Twb[:3, :3]
  p = Twb[:3, -1]
  return np.r_[np.c_[R.T, -R.T@p], [[0, 0, 0, 1]]]

def get_homogeneous_transformation_matrix(params):
  a, al, d, th = params
  return np.array([[           np.cos(th),           -np.sin(th),           0,             a],
                   [np.sin(th)*np.cos(al), np.cos(th)*np.cos(al), -np.sin(al), -np.sin(al)*d],
                   [np.sin(th)*np.sin(al), np.cos(th)*np.sin(al),  np.cos(al),  np.cos(al)*d],
                   [                    0,                     0,           0,             1]])

def forward_kinematics(dh_params, pose):
  T = np.eye(4)
  T_list = []
  p_list = []
  for params in dh_params:
    T = np.dot(T, get_homogeneous_transformation_matrix(params))
    T_list.append(T)
  
  Twb = get_world_to_base(pose[0], pose[1])
  for i in range(len(T_list)):
    T_list[i] = Twb @ T_list[i]
    p_list.append(T_list[i][:3, -1].reshape(3))

  return np.array(T_list), np.array(p_list)

def inverse_kinematics(idx, point):
  l1, l2, l3, l4 = 0.35, 0.25, 0.7, 1.2
  x, y, z = point

  x01 = x - l1
  th1 = np.arctan2(y, x01)
  
  x02 = x01 / np.cos(th1)
  z02 = z - l2

  c3 = (x02**2 + z02**2 - l3**2 - l4**2) / (2 * l3 * l4)
  s3 = np.sqrt(1 - c3**2)
  th3 = np.arctan2(s3, c3) if idx >= 3 else np.arctan2(-s3, c3)

  m = l3 + l4 * np.cos(th3) 
  n = l4 * np.sin(th3) if idx < 3 else l4 * np.sin(-th3)
  th2 = np.arctan2(z02, x02) - np.arctan2(n, m)
  th2 = -th2 if idx >= 3 else th2
  return th1, th2, th3

def compute_inverse_kinematics(idx, point):
  th1, th2, th3 = inverse_kinematics(idx, point)
  
  offset_th2 = np.deg2rad(45) if idx >= 3 else -np.deg2rad(45)
  offset_th3 = -np.deg2rad(135) if idx >= 3 else np.deg2rad(135)
  
  th2 += offset_th2
  th3 += offset_th3
  return th1, th2, th3

def leg_center_of_mass(T_list, p_list, origin_list, mass_list):
  leg_com = np.zeros((4, 1))
  leg_mass = 0.0
  for i in range(3):
    leg_com += (T_list[i+2] @ origin_list[i].reshape(4, 1)) * mass_list[i]
    leg_mass += mass_list[i]
  return leg_com, leg_mass

def slider_callback(idx, th1, th2, th3, pose, dh_params, line, point, text, text2, is_ver2=False):
  offset_th2 = -np.deg2rad(45) if idx >= 3 else np.deg2rad(45)
  offset_th3 = np.deg2rad(135) if idx >= 3 else -np.deg2rad(135)
  
  if is_ver2 is False:
    dh_params[1, -1] = np.deg2rad(th1)
    dh_params[3, -1] = np.deg2rad(th2) + offset_th2
    dh_params[4, -1] = np.deg2rad(th3) + offset_th3
  else:
    dh_params[1, -1] = th1
    dh_params[3, -1] = th2 + offset_th2
    dh_params[4, -1] = th3 + offset_th3

  T, p = forward_kinematics(dh_params, pose)
  line.set_xdata(p[:, 0])
  line.set_ydata(p[:, 1])
  line.set_3d_properties(p[:, 2])

  point.set_xdata(p[:, 0])
  point.set_ydata(p[:, 1])
  point.set_3d_properties(p[:, 2])

  if is_ver2 is False:
    text[0].set_text(f'{np.deg2rad(th1):.4f} rad')
    text[1].set_text(f'{np.deg2rad(th2):.4f} rad')
    text[2].set_text(f'{np.deg2rad(th3):.4f} rad')
  else:
    text[0].set_text(f'{th1:.4f} rad')
    text[1].set_text(f'{th2:.4f} rad')
    text[2].set_text(f'{th3:.4f} rad')
  
  end_point = np.append(p[-1], 1).reshape(4, 1)
  T01 = T[0]
  R01 = T01[:3, :3]
  p01 = T01[:3, -1].reshape(3, 1)
  T10 = np.r_[np.c_[R01.T, -R01.T@p01], [[0, 0, 0, 1]]]
  end_point = T10 @ end_point
  end_point = end_point[:3, 0]

  text2[0].set_text(f'{end_point[0]:.4f} m')
  text2[1].set_text(f'{end_point[1]:.4f} m')
  text2[2].set_text(f'{end_point[2]:.4f} m')

def update_com(pose, dh_params_list, origin_list, mass_list, theta_list, com_line, com_point):
  for i, thetas in enumerate(theta_list):
    th1, th2, th3 = thetas
    offset_th2 = -np.deg2rad(45) if i >= 3 else np.deg2rad(45)
    offset_th3 = np.deg2rad(135) if i >= 3 else -np.deg2rad(135)

    dh_params_list[i][1, -1] = np.deg2rad(th1)
    dh_params_list[i][3, -1] = np.deg2rad(th2) + offset_th2
    dh_params_list[i][4, -1] = np.deg2rad(th3) + offset_th3

  mass = 10.3552543883176
  com = get_world_to_base(pose[0], pose[1])[:, -1] * mass
  com = com.reshape(4, 1)
  for dh_params, origins, masses in zip(dh_params_list, origin_list, mass_list):
    T_list, p_list = forward_kinematics(dh_params, pose)
    leg_com, leg_mass = leg_center_of_mass(T_list, p_list, origins, masses)
    com += leg_com
    mass += leg_mass
  com /= mass
  
  com_line.set_xdata([com[0, 0], com[0, 0]])
  com_line.set_ydata([com[1, 0], com[1, 0]])
  com_line.set_3d_properties([0, com[2, 0]])
  
  com_point.set_xdata([com[0, 0], com[0, 0]])
  com_point.set_ydata([com[1, 0], com[1, 0]])
  com_point.set_3d_properties([0, com[2, 0]])

def plot_robot_ver2(dh_params_list, colors, origin_list, mass_list):
  l1, l2, l3, l4 = 0.35, 0.25, 0.7, 1.2
  
  fig = plt.figure(figsize=(20, 10))
  fig.subplots_adjust(left=0.55, right=0.85, top=0.85, bottom=0.15)
  ax = fig.add_subplot(111, projection='3d')
  
  axes_sld_1 = fig.add_axes([0.15, 0.65, 0.3, 0.03])
  axes_sld_2 = fig.add_axes([0.15, 0.60, 0.3, 0.03])
  axes_sld_3 = fig.add_axes([0.15, 0.55, 0.3, 0.03])
  
  axes_btn_1 = fig.add_axes([0.15, 0.7, 0.03, 0.03])
  axes_btn_2 = fig.add_axes([0.20, 0.7, 0.03, 0.03])
  axes_btn_3 = fig.add_axes([0.25, 0.7, 0.03, 0.03])
  axes_btn_4 = fig.add_axes([0.30, 0.7, 0.03, 0.03])
  axes_btn_5 = fig.add_axes([0.35, 0.7, 0.03, 0.03])
  axes_btn_6 = fig.add_axes([0.40, 0.7, 0.03, 0.03])
  
  selected_leg = {'index':0}
  theta_list = np.zeros((6, 3))
  end_points = np.zeros((6, 3))
  end_points[:, 0] = l3*np.cos(np.deg2rad(45)) + l1
  end_points[:, 2] = l3*np.sin(np.deg2rad(45)) - l4 + l2
  pose = [[0, 0, 0], [0, 0, 0.455025253]]
  
  mass = 10.3552543883176
  com = get_world_to_base(pose[0], pose[1])[:, -1] * mass
  com = com.reshape(4, 1)
  
  T_array, p_array = [], []
  line_list, point_list = [], []
  base_list = []
  for dh_params, color, origins, masses in zip(dh_params_list, colors, origin_list, mass_list):
    T_list, p_list = forward_kinematics(dh_params, pose)
    leg_com, leg_mass = leg_center_of_mass(T_list, p_list, origins, masses)
    com += leg_com
    mass += leg_mass

    T_array.append(T_list)
    p_array.append(p_list)
    base_list.append(p_list[1])
    
    line, = ax.plot3D(p_list[:, 0], p_list[:, 1], p_list[:, 2], color=color)
    point, = ax.plot3D(p_list[:, 0], p_list[:, 1], p_list[:, 2], 'o', color=color)
    
    line_list.append(line)
    point_list.append(point)
  
  com /= mass
  com_line, = ax.plot3D([com[0, 0], com[0, 0]], [com[1, 0], com[1, 0]], [0, com[2, 0]], 'k--')
  com_point, = ax.plot3D([com[0, 0], com[0, 0]], [com[1, 0], com[1, 0]], [0, com[2, 0]], 'ko')

  # Text
  box_kwargs = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  text_list = []
  for i in range(len(theta_list)):
      plt.text(-9.2, -8.0-i*1.0, f'Leg {i}: ')
      text_1 = plt.text(-8.2, -8.0-i*1.0, f'{theta_list[i, 0]:.4f} rad', bbox=box_kwargs)
      text_2 = plt.text(-6.2, -8.0-i*1.0, f'{theta_list[i, 0]:.4f} rad', bbox=box_kwargs)
      text_3 = plt.text(-4.2, -8.0-i*1.0, f'{theta_list[i, 0]:.4f} rad', bbox=box_kwargs)
      text_list.append([text_1, text_2, text_3])

  # Text
  box_kwargs = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
  for i in range(len(theta_list)):
      plt.text(-9.2, -15.0-i*1.0, f'Leg {i}: ')
      text_1 = plt.text(-8.2, -15.0-i*1.0, f'{theta_list[i, 0]:.4f} m', bbox=box_kwargs)
      text_2 = plt.text(-6.2, -15.0-i*1.0, f'{theta_list[i, 0]:.4f} m', bbox=box_kwargs)
      text_3 = plt.text(-4.2, -15.0-i*1.0, f'{theta_list[i, 0]:.4f} m', bbox=box_kwargs)
      text_list.append([text_1, text_2, text_3])
  
  axes_btn_reset = fig.add_axes([0.35, 0.38, 0.1, 0.1])
  
  # Slider
  slider_1 = Slider(ax=axes_sld_1, label='X', valmin=-2.5, valmax=2.5, valinit=l3*np.cos(np.deg2rad(45)) + l1)
  slider_2 = Slider(ax=axes_sld_2, label='Y', valmin=-2.5, valmax=2.5, valinit=0.0)
  slider_3 = Slider(ax=axes_sld_3, label='Z', valmin=-2.5, valmax=2.5, valinit=l3*np.sin(np.deg2rad(45)) - l4 + l2)
  
  def slider_callback_1(val):
    idx = selected_leg['index']
    point = [val, end_points[idx, 1], end_points[idx, 2]]
    th1, th2, th3 = compute_inverse_kinematics(idx, point)
    dh_params = dh_params_list[idx]
    theta_list[idx, 0] = th1
    theta_list[idx, 1] = th2
    theta_list[idx, 2] = th3
    end_points[idx, 0] = val
    slider_callback(idx, th1, th2, th3, pose, dh_params, line_list[idx], point_list[idx], text_list[idx], text_list[idx+6], is_ver2=True)
    update_com(pose, dh_params_list.copy(), origin_list, mass_list, theta_list, com_line, com_point)
    
  def slider_callback_2(val):
    idx = selected_leg['index']
    point = [end_points[idx, 0], val, end_points[idx, 2]]
    th1, th2, th3 = compute_inverse_kinematics(idx, point)
    dh_params = dh_params_list[idx]
    theta_list[idx, 0] = th1
    theta_list[idx, 1] = th2
    theta_list[idx, 2] = th3
    end_points[idx, 1] = val
    slider_callback(idx, th1, th2, th3, pose, dh_params, line_list[idx], point_list[idx], text_list[idx], text_list[idx+6], is_ver2=True)
    update_com(pose, dh_params_list.copy(), origin_list, mass_list, theta_list, com_line, com_point)
  
  def slider_callback_3(val):
    idx = selected_leg['index']
    point = [end_points[idx, 0], end_points[idx, 1], val]
    th1, th2, th3 = compute_inverse_kinematics(idx, point)
    dh_params = dh_params_list[idx]
    theta_list[idx, 0] = th1
    theta_list[idx, 1] = th2
    theta_list[idx, 2] = th3
    end_points[idx, 2] = val
    slider_callback(idx, th1, th2, th3, pose, dh_params, line_list[idx], point_list[idx], text_list[idx], text_list[idx+6], is_ver2=True)
    update_com(pose, dh_params_list.copy(), origin_list, mass_list, theta_list, com_line, com_point)

  slider_1.on_changed(slider_callback_1)
  slider_2.on_changed(slider_callback_2)
  slider_3.on_changed(slider_callback_3)
  
  # Button 
  btn_1 = Button(axes_btn_1, 'Leg 1')
  btn_2 = Button(axes_btn_2, 'Leg 2')
  btn_3 = Button(axes_btn_3, 'Leg 3')
  btn_4 = Button(axes_btn_4, 'Leg 4')
  btn_5 = Button(axes_btn_5, 'Leg 5')
  btn_6 = Button(axes_btn_6, 'Leg 6')
  btn_reset = Button(axes_btn_reset, 'Reset')

  def button_callback(i):
    def callback(event):
      selected_leg['index'] = i
      slider_1.set_val(end_points[i, 0])
      slider_2.set_val(end_points[i, 1])
      slider_3.set_val(end_points[i, 2])
    return callback
  
  def button_reset_callback(event):
    theta_list = np.zeros((6, 3))
    end_points = np.zeros((6, 3))
    end_points[:, 0] = l3*np.cos(np.deg2rad(45)) + l1
    end_points[:, 2] = l3*np.sin(np.deg2rad(45)) - l4 + l2
    for idx in range(6):
      selected_leg['index'] = idx
      slider_1.set_val(l3*np.cos(np.deg2rad(45)) + l1)
      slider_2.set_val(0.0)
      slider_3.set_val(l3*np.sin(np.deg2rad(45)) - l4 + l2)
    selected_leg['index'] = 0
  
  btn_1.on_clicked(button_callback(0))
  btn_2.on_clicked(button_callback(1))
  btn_3.on_clicked(button_callback(2))
  btn_4.on_clicked(button_callback(3))
  btn_5.on_clicked(button_callback(4))
  btn_6.on_clicked(button_callback(5))
  btn_reset.on_clicked(button_reset_callback)
  
  base_list.append(base_list[0])
  base_list = np.array(base_list)
  base_line, = ax.plot3D(base_list[:, 0], base_list[:, 1], base_list[:, 2], color='grey')
  
  Twb = get_world_to_base(pose[0], pose[1])
  x_axis = Twb @ np.array([0.2, 0, 0, 1]).reshape(4, 1)
  y_axis = Twb @ np.array([0, 0.2, 0, 1]).reshape(4, 1)
  z_axis = Twb @ np.array([0, 0, 0.2, 1]).reshape(4, 1)
  base_x_axis, = ax.plot3D([Twb[0, -1], x_axis[0, 0]], [Twb[1, -1], x_axis[1, 0]], [Twb[2, -1], x_axis[2, 0]], color='red', linewidth=2.5)
  base_y_axis, = ax.plot3D([Twb[0, -1], y_axis[0, 0]], [Twb[1, -1], y_axis[1, 0]], [Twb[2, -1], y_axis[2, 0]], color='green', linewidth=2.5)
  base_z_axis, = ax.plot3D([Twb[0, -1], z_axis[0, 0]], [Twb[1, -1], z_axis[1, 0]], [Twb[2, -1], z_axis[2, 0]], color='blue', linewidth=2.5)

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])
  ax.set_zlim([-0.1, 2.9])
  plt.grid()
  plt.show()

def plot_robot(dh_params_list, colors, origin_list, mass_list):
  fig = plt.figure(figsize=(20, 10))
  fig.subplots_adjust(left=0.55, right=0.85, top=0.85, bottom=0.15)
  ax = fig.add_subplot(111, projection='3d')

  axes_sld_1 = fig.add_axes([0.15, 0.65, 0.3, 0.03])
  axes_sld_2 = fig.add_axes([0.15, 0.60, 0.3, 0.03])
  axes_sld_3 = fig.add_axes([0.15, 0.55, 0.3, 0.03])
  
  axes_btn_1 = fig.add_axes([0.15, 0.7, 0.03, 0.03])
  axes_btn_2 = fig.add_axes([0.20, 0.7, 0.03, 0.03])
  axes_btn_3 = fig.add_axes([0.25, 0.7, 0.03, 0.03])
  axes_btn_4 = fig.add_axes([0.30, 0.7, 0.03, 0.03])
  axes_btn_5 = fig.add_axes([0.35, 0.7, 0.03, 0.03])
  axes_btn_6 = fig.add_axes([0.40, 0.7, 0.03, 0.03])
  
  selected_leg = {'index':0}
  theta_list = np.zeros((6, 3))
  pose = [[0, 0, 0], [0, 0, 0.455025253]]
  
  mass = 10.3552543883176
  com = get_world_to_base(pose[0], pose[1])[:, -1] * mass
  com = com.reshape(4, 1)
  
  T_array, p_array = [], []
  line_list, point_list = [], []
  base_list = []
  for dh_params, color, origins, masses in zip(dh_params_list, colors, origin_list, mass_list):
    T_list, p_list = forward_kinematics(dh_params, pose)
    leg_com, leg_mass = leg_center_of_mass(T_list, p_list, origins, masses)
    com += leg_com
    mass += leg_mass

    T_array.append(T_list)
    p_array.append(p_list)
    base_list.append(p_list[1])
    
    line, = ax.plot3D(p_list[:, 0], p_list[:, 1], p_list[:, 2], color=color)
    point, = ax.plot3D(p_list[:, 0], p_list[:, 1], p_list[:, 2], 'o', color=color)
    
    line_list.append(line)
    point_list.append(point)
  
  com /= mass
  com_line, = ax.plot3D([com[0, 0], com[0, 0]], [com[1, 0], com[1, 0]], [0, com[2, 0]], 'k--')
  com_point, = ax.plot3D([com[0, 0], com[0, 0]], [com[1, 0], com[1, 0]], [0, com[2, 0]], 'ko')

  # Text
  box_kwargs = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  text_list = []
  for i in range(len(theta_list)):
      plt.text(-9.2, -8.0-i*1.0, f'Leg {i}: ')
      text_1 = plt.text(-8.2, -8.0-i*1.0, f'{theta_list[i, 0]:.4f} rad', bbox=box_kwargs)
      text_2 = plt.text(-6.2, -8.0-i*1.0, f'{theta_list[i, 0]:.4f} rad', bbox=box_kwargs)
      text_3 = plt.text(-4.2, -8.0-i*1.0, f'{theta_list[i, 0]:.4f} rad', bbox=box_kwargs)
      text_list.append([text_1, text_2, text_3])

  # Text
  box_kwargs = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
  for i in range(len(theta_list)):
      plt.text(-9.2, -15.0-i*1.0, f'Leg {i}: ')
      text_1 = plt.text(-8.2, -15.0-i*1.0, f'{theta_list[i, 0]:.4f} m', bbox=box_kwargs)
      text_2 = plt.text(-6.2, -15.0-i*1.0, f'{theta_list[i, 0]:.4f} m', bbox=box_kwargs)
      text_3 = plt.text(-4.2, -15.0-i*1.0, f'{theta_list[i, 0]:.4f} m', bbox=box_kwargs)
      text_list.append([text_1, text_2, text_3])
  
  axes_btn_reset = fig.add_axes([0.35, 0.38, 0.1, 0.1])
  
  # Slider
  slider_1 = Slider(ax=axes_sld_1, label='theta 1', valmin=-180.0, valmax=180.0, valinit=0.0)
  slider_2 = Slider(ax=axes_sld_2, label='theta 2', valmin=-180.0, valmax=180.0, valinit=0.0)
  slider_3 = Slider(ax=axes_sld_3, label='theta 3', valmin=-180.0, valmax=180.0, valinit=0.0)
  
  def slider_callback_1(val):
    idx = selected_leg['index']
    th1 = slider_1.val
    th2 = theta_list[idx, 1]
    th3 = theta_list[idx, 2]
    dh_params = dh_params_list[idx]
    theta_list[idx, 0] = th1
    slider_callback(idx, th1, th2, th3, pose, dh_params, line_list[idx], point_list[idx], text_list[idx], text_list[idx+6])
    update_com(pose, dh_params_list.copy(), origin_list, mass_list, theta_list, com_line, com_point)
    
  def slider_callback_2(val):
    idx = selected_leg['index']
    th1 = theta_list[idx, 0]
    th2 = slider_2.val
    th3 = theta_list[idx, 2]
    dh_params = dh_params_list[idx]
    theta_list[idx, 1] = th2
    slider_callback(idx, th1, th2, th3, pose, dh_params, line_list[idx], point_list[idx], text_list[idx], text_list[idx+6])
    update_com(pose, dh_params_list.copy(), origin_list, mass_list, theta_list, com_line, com_point)

  def slider_callback_3(val):
    idx = selected_leg['index']
    th1 = theta_list[idx, 0]
    th2 = theta_list[idx, 1]
    th3 = slider_3.val
    dh_params = dh_params_list[idx]
    theta_list[idx, 2] = th3
    slider_callback(idx, th1, th2, th3, pose, dh_params, line_list[idx], point_list[idx], text_list[idx], text_list[idx+6])
    update_com(pose, dh_params_list.copy(), origin_list, mass_list, theta_list, com_line, com_point)

  slider_1.on_changed(slider_callback_1)
  slider_2.on_changed(slider_callback_2)
  slider_3.on_changed(slider_callback_3)
  
  # Button 
  btn_1 = Button(axes_btn_1, 'Leg 1')
  btn_2 = Button(axes_btn_2, 'Leg 2')
  btn_3 = Button(axes_btn_3, 'Leg 3')
  btn_4 = Button(axes_btn_4, 'Leg 4')
  btn_5 = Button(axes_btn_5, 'Leg 5')
  btn_6 = Button(axes_btn_6, 'Leg 6')
  btn_reset = Button(axes_btn_reset, 'Reset')

  def button_callback(i):
    def callback(event):
      selected_leg['index'] = i
      slider_1.set_val(theta_list[i, 0])
      slider_2.set_val(theta_list[i, 1])
      slider_3.set_val(theta_list[i, 2])
    return callback
  
  def button_reset_callback(event):
    theta_list = np.zeros((6, 3))
    for idx in range(6):
      selected_leg['index'] = idx
      slider_1.set_val(theta_list[idx, 0])
      slider_2.set_val(theta_list[idx, 1])
      slider_3.set_val(theta_list[idx, 2])
  
  btn_1.on_clicked(button_callback(0))
  btn_2.on_clicked(button_callback(1))
  btn_3.on_clicked(button_callback(2))
  btn_4.on_clicked(button_callback(3))
  btn_5.on_clicked(button_callback(4))
  btn_6.on_clicked(button_callback(5))
  btn_reset.on_clicked(button_reset_callback)
  
  base_list.append(base_list[0])
  base_list = np.array(base_list)
  base_line, = ax.plot3D(base_list[:, 0], base_list[:, 1], base_list[:, 2], color='grey')
  
  Twb = get_world_to_base(pose[0], pose[1])
  x_axis = Twb @ np.array([0.2, 0, 0, 1]).reshape(4, 1)
  y_axis = Twb @ np.array([0, 0.2, 0, 1]).reshape(4, 1)
  z_axis = Twb @ np.array([0, 0, 0.2, 1]).reshape(4, 1)
  base_x_axis, = ax.plot3D([Twb[0, -1], x_axis[0, 0]], [Twb[1, -1], x_axis[1, 0]], [Twb[2, -1], x_axis[2, 0]], color='red', linewidth=2.5)
  base_y_axis, = ax.plot3D([Twb[0, -1], y_axis[0, 0]], [Twb[1, -1], y_axis[1, 0]], [Twb[2, -1], y_axis[2, 0]], color='green', linewidth=2.5)
  base_z_axis, = ax.plot3D([Twb[0, -1], z_axis[0, 0]], [Twb[1, -1], z_axis[1, 0]], [Twb[2, -1], z_axis[2, 0]], color='blue', linewidth=2.5)

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])
  ax.set_zlim([-0.1, 2.9])
  plt.grid()
  plt.show()

def main():
  l1, l2, l3, l4 = 0.35, 0.25, 0.7, 1.2
  dh_params_1 = np.array([[ 0,        0,  0, np.deg2rad(30)+np.deg2rad(60)*0],
                          [l1,        0,  0,                               0],
                          [ 0,        0, l2,                               0],
                          [ 0,  np.pi/2,  0,                  np.deg2rad(45)],
                          [l3,        0,  0,                -np.deg2rad(135)],
                          [l4,        0,  0,                               0]])
  
  dh_params_2 = np.array([[ 0,        0,  0, np.deg2rad(30)+np.deg2rad(60)*1],
                          [l1,        0,  0,                               0],
                          [ 0,        0, l2,                               0],
                          [ 0,  np.pi/2,  0,                  np.deg2rad(45)],
                          [l3,        0,  0,                -np.deg2rad(135)],
                          [l4,        0,  0,                               0]])
  
  dh_params_3 = np.array([[ 0,        0,  0, np.deg2rad(30)+np.deg2rad(60)*2],
                          [l1,        0,  0,                               0],
                          [ 0,        0, l2,                               0],
                          [ 0,  np.pi/2,  0,                  np.deg2rad(45)],
                          [l3,        0,  0,                -np.deg2rad(135)],
                          [l4,        0,  0,                               0]])
  
  dh_params_4 = np.array([[ 0,        0,  0, -np.deg2rad(30)-np.deg2rad(60)*2],
                          [l1,        0,  0,                                0],
                          [ 0,        0, l2,                                0],
                          [ 0, -np.pi/2,  0,                  -np.deg2rad(45)],
                          [l3,        0,  0,                  np.deg2rad(135)],
                          [l4,        0,  0,                                0]])
  
  dh_params_5 = np.array([[ 0,        0,  0, -np.deg2rad(30)-np.deg2rad(60)*1],
                          [l1,        0,  0,                                0],
                          [ 0,        0, l2,                                0],
                          [ 0, -np.pi/2,  0,                  -np.deg2rad(45)],
                          [l3,        0,  0,                  np.deg2rad(135)],
                          [l4,        0,  0,                                0]])
  
  dh_params_6 = np.array([[ 0,        0,  0, -np.deg2rad(30)-np.deg2rad(60)*0],
                          [l1,        0,  0,                                0],
                          [ 0,        0, l2,                                0],
                          [ 0, -np.pi/2,  0,                  -np.deg2rad(45)],
                          [l3,        0,  0,                  np.deg2rad(135)],
                          [l4,        0,  0,                                0]])
  
  origin_list_1 = np.array([[-0.000496539974013388, 3.88355542169094E-10,    -0.164643358154885, 1],
                            [    0.247487373415291,    0.247487373415292,  -0.00206006865928844, 1],
                            [                0.675,                    0, -9.80960396928888E-17, 1]])
  
  origin_list_2 = np.array([[-0.000496540203313467, 4.22521449926359E-10,    -0.164643358346364, 1],
                            [                 0.35,                    0,  -0.00206006865928853, 1],
                            [                0.675,                    0, -9.44121348005163E-17, 1]])
  
  origin_list_3 = np.array([[-0.000496539975061716, 3.72729247794499E-10,    -0.164643358158046, 1],
                            [                 0.35,                    0,  -0.00206006865928842, 1],
                            [                0.675,                    0, -9.80960396928884E-17, 1]])
  
  origin_list_4 = np.array([[-0.000496539974013444, 3.88355537599323E-10,   -0.164643358154885, 1],
                            [                 0.35, 5.55111512312578E-17, -0.00206006865928859, 1],
                            [                0.675, 1.11022302462516E-16, 1.33925684875783E-16, 1]])
  
  origin_list_5 = np.array([[-0.000496540203313411,  4.22521395908064E-10,   -0.164643358346364, 1],
                            [                 0.35, -1.66533453693773E-16, -0.00206006865928855, 1],
                            [                0.675,  1.11022302462516E-16, 1.01113686938468E-16, 1]])
  
  origin_list_6 = np.array([[-0.000496539975061827,  3.72729398327905E-10,   -0.164643358158046, 1],
                            [                 0.35, -1.11022302462516E-16, -0.00206006865928856, 1],
                            [                0.675, -1.11022302462516E-16, 2.29033824132673E-17, 1]])

  mass_list_1 = np.array([1.92360326221169, 9.70992368814396, 1.97482300164693])
  mass_list_2 = np.array([1.92360326221169, 9.70992368814396, 1.97482300164693])
  mass_list_3 = np.array([1.92360326221169, 9.70992368814396, 1.97482300164693])
  mass_list_4 = np.array([1.92360326221169, 9.70992368814396, 1.97482300164693])
  mass_list_5 = np.array([1.92360326221169, 9.70992368814396, 1.97482300164693])
  mass_list_6 = np.array([1.92360326221169, 9.70992368814396, 1.97482300164693])

  dh_params_list = [dh_params_1, dh_params_2, dh_params_3,
                    dh_params_4, dh_params_5, dh_params_6]
  colors = ['grey', 'grey', 'grey', 'grey', 'grey', 'grey']
  origin_list = [origin_list_1, origin_list_2, origin_list_3,
                 origin_list_4, origin_list_5, origin_list_6]
  mass_list = [mass_list_1, mass_list_2, mass_list_3,
               mass_list_4, mass_list_5, mass_list_6]
  plot_robot(dh_params_list, colors, origin_list, mass_list)
  plot_robot_ver2(dh_params_list, colors, origin_list, mass_list)

if __name__ == "__main__":
    main()
