# Husain, Rabib
# 1002_053_770
# 2025_11_02
# Assignment_04_01

import numpy as np
import tkinter as tk
from tkinter import filedialog

class cl_camera:
    # Class to represent a camera with its parameter
    def __init__(self):
        self.name = ""
        self.type = "parallel"  # "parallel" or "perspective"
        self.vrp = np.array([0.0, 0.0, 0.0])
        self.vpn = np.array([0.0, 0.0, 1.0])
        self.vup = np.array([0.0, 1.0, 0.0])
        self.prp = np.array([0.0, 0.0, 1.0])
        self.view_volume = [-1, 1, -1, 1, -1, 1]  # umin, umax, vmin, vmax, nmin, nmax
        self.viewport = (0.1, 0.1, 0.4, 0.4)
        self.matrix = None
    
    def compute_projection_matrix(self):
        # Compute the composite projection matrix (parallel or perspective)
        # Step 1: Translate VRP to origin
        T_vrp = np.eye(4)
        T_vrp[:3, 3] = -self.vrp
        
        # Step 2: Rotate to align VPN with Z-axis and VUP with Y-axis
        vpn = self.vpn / np.linalg.norm(self.vpn)
        vup = self.vup / np.linalg.norm(self.vup)
        
        # Compute u, v, n vectors
        n = vpn
        u = np.cross(vup, n)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        
        R = np.eye(4)
        R[0, :3] = u
        R[1, :3] = v
        R[2, :3] = n
        
        # Step 3: Translate PRP to origin
        T_prp = np.eye(4)
        T_prp[:3, 3] = -self.prp
        
        # Step 4: Calculate view volume parameters
        umin, umax, vmin, vmax, nmin, nmax = self.view_volume
        CW = np.array([(umin + umax) / 2, (vmin + vmax) / 2, 0])
        DOP = CW - self.prp
        
        if self.type.lower() == "perspective":
            # Perspective projection
            # Step 4: Shear to make center line parallel to z-axis
            SH_per = np.eye(4)
            SH_per[0, 2] = -DOP[0] / DOP[2]
            SH_per[1, 2] = -DOP[1] / DOP[2]
            
            # Step 5: Scale to canonical perspective view volume
            # The canonical perspective volume has the form:
            # -z <= x <= z, -z <= y <= z, z_min <= z <= z_max
            # After scaling, we want: z_prp + n_min -> some z_min (we use view window)
            #                         z_prp + n_max -> z_max = 1
            
            S_per = np.eye(4)
            S_per[0, 0] = (2 * self.prp[2]) / ((umax - umin) * (self.prp[2] + nmax))
            S_per[1, 1] = (2 * self.prp[2]) / ((vmax - vmin) * (self.prp[2] + nmax))
            S_per[2, 2] = 1 / (self.prp[2] + nmax)
            
            # Composite matrix for perspective
            self.matrix = S_per @ SH_per @ T_prp @ R @ T_vrp
        else:
            # Parallel projection
            # Step 4: Shear to make center line parallel to z-axis
            SH_par = np.eye(4)
            SH_par[0, 2] = -DOP[0] / DOP[2]
            SH_par[1, 2] = -DOP[1] / DOP[2]
            
            # Step 5: Scale to canonical view volume
            S_par = np.eye(4)
            S_par[0, 0] = 2 / (umax - umin)
            S_par[1, 1] = 2 / (vmax - vmin)
            S_par[2, 2] = 1 / (nmax - nmin)
            
            # Step 6: Translate to center
            T_par = np.eye(4)
            T_par[0, 3] = -(umin + umax) / 2
            T_par[1, 3] = -(vmin + vmax) / 2
            T_par[2, 3] = -nmin
            
            # Composite matrix for parallel
            self.matrix = S_par @ T_par @ SH_par @ T_prp @ R @ T_vrp

class cl_world:
    def __init__(self):
        # Main Window
        self.root = tk.Tk()
        self.root.title("Assignment 04 - Parallel and Perspective Projection")
        self.root.geometry("640x800")
        self.root.resizable(True, True)

        # Top Panel - light gray background
        self.top_frame = tk.Frame(self.root, bg="#e0e0e0")
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Filename section
        tk.Label(self.top_frame, text="Filename:", bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        self.filename_entry = tk.Entry(self.top_frame, width=30)
        self.filename_entry.pack(side=tk.LEFT, padx=5, pady=5)
        self.browse_button = tk.Button(self.top_frame, text="Browse", command=self.browse_file_clicked)
        self.browse_button.pack(side=tk.LEFT, padx=2, pady=5)
        self.load_button = tk.Button(self.top_frame, text="Load", bg="#ff6666", command=self.load_button_clicked)
        self.load_button.pack(side=tk.LEFT, padx=2, pady=5)

        # Rotation Panel
        self.rot_frame = tk.Frame(self.root, bg="#e0e0e0")
        self.rot_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Label(self.rot_frame, text="Rotation Axis:", bg="#e0e0e0").pack(side=tk.LEFT)
        self.rot_axis = tk.StringVar(value="x")
        tk.Radiobutton(self.rot_frame, text="X", variable=self.rot_axis, value="x", bg="#e0e0e0", command=self.update_rotation_state).pack(side=tk.LEFT)
        tk.Radiobutton(self.rot_frame, text="Y", variable=self.rot_axis, value="y", bg="#e0e0e0", command=self.update_rotation_state).pack(side=tk.LEFT)
        tk.Radiobutton(self.rot_frame, text="Z", variable=self.rot_axis, value="z", bg="#e0e0e0", command=self.update_rotation_state).pack(side=tk.LEFT)
        tk.Radiobutton(self.rot_frame, text="Line AB", variable=self.rot_axis, value="line", bg="#e0e0e0", command=self.update_rotation_state).pack(side=tk.LEFT)
        
        tk.Label(self.rot_frame, text="A:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(5,1))
        self.ax = tk.Entry(self.rot_frame, width=10)
        self.ax.insert(0, "[0.0,0.0,0.0]")
        self.ax.pack(side=tk.LEFT)
        
        tk.Label(self.rot_frame, text="B:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(3,1))
        self.bx = tk.Entry(self.rot_frame, width=10)
        self.bx.insert(0, "[1.0,1.0,1.0]")
        self.bx.pack(side=tk.LEFT)
        
        tk.Label(self.rot_frame, text="Degree:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(5,1))
        self.rot_angle = self.create_spinner_entry(self.rot_frame, "90", 4)
        
        tk.Label(self.rot_frame, text="Steps:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(3,1))
        self.rot_steps = self.create_spinner_entry(self.rot_frame, "5", 3)
        
        self.rotate_button = tk.Button(self.rot_frame, text="Rotate", command=self.rotate_clicked)
        self.rotate_button.pack(side=tk.LEFT, padx=3)

        # Scale Panel
        self.scale_frame = tk.Frame(self.root, bg="#e0e0e0")
        self.scale_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Label(self.scale_frame, text="Scale about point:", bg="#e0e0e0").pack(side=tk.LEFT)
        self.scale_point = tk.Entry(self.scale_frame, width=10)
        self.scale_point.insert(0, "[0.0,0.0,0.0]")
        self.scale_point.pack(side=tk.LEFT, padx=3)
        
        tk.Label(self.scale_frame, text="Scale Ratio:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(5,1))
        self.scale_ratio_type = tk.StringVar(value="all")
        tk.Radiobutton(self.scale_frame, text="All", variable=self.scale_ratio_type, value="all", bg="#e0e0e0", command=self.update_scale_state).pack(side=tk.LEFT)
        
        # Scale All with spinner
        self.scale_all = self.create_spinner_entry(self.scale_frame, "1", 3, step=0.25)
        
        tk.Radiobutton(self.scale_frame, text="[Sx,Sy,Sz]", variable=self.scale_ratio_type, value="xyz", bg="#e0e0e0", command=self.update_scale_state).pack(side=tk.LEFT, padx=(3,1))
        self.scale_xyz = tk.Entry(self.scale_frame, width=8)
        self.scale_xyz.insert(0, "[1,1,1]")
        self.scale_xyz.pack(side=tk.LEFT)
        
        tk.Label(self.scale_frame, text="Steps:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(3,1))
        self.scale_steps = self.create_spinner_entry(self.scale_frame, "4", 3)
        
        self.scale_button = tk.Button(self.scale_frame, text="Scale", command=self.scale_clicked)
        self.scale_button.pack(side=tk.LEFT, padx=3)

        # Translation Panel
        self.trans_frame = tk.Frame(self.root, bg="#e0e0e0")
        self.trans_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Label(self.trans_frame, text="Translation ([dx, dy, dz]):", bg="#e0e0e0").pack(side=tk.LEFT)
        self.trans_vec = tk.Entry(self.trans_frame, width=12)
        self.trans_vec.insert(0, "[-.5,-.6,0.5]")
        self.trans_vec.pack(side=tk.LEFT, padx=3)
        
        tk.Label(self.trans_frame, text="Steps:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(3,1))
        self.trans_steps = self.create_spinner_entry(self.trans_frame, "4", 3)
        
        self.translate_button = tk.Button(self.trans_frame, text="Translate", command=self.translate_clicked)
        self.translate_button.pack(side=tk.LEFT, padx=3)

        # Fly Camera Panel
        self.fly_frame = tk.Frame(self.root, bg="#e0e0e0")
        self.fly_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        self.fly_button = tk.Button(self.fly_frame, text="Fly Camera...", command=self.fly_camera_clicked)
        self.fly_button.pack(side=tk.LEFT, padx=3)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="yellow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.canvas_resized)
        self.canvas.bind("<ButtonPress-1>", self.left_mouse_pressed)
        self.canvas.bind("<B1-Motion>", self.left_mouse_dragged)
        self.canvas.bind("<ButtonPress-3>", self.right_mouse_pressed)
        self.canvas.bind("<B3-Motion>", self.right_mouse_dragged)

        # Data
        self.vertices = []
        self.faces = []
        self.file_path = None
        self.original_vertices = []
        self.animation_step = 0
        self.animation_total_steps = 0
        self.animation_transform = None
        
        # Multiple cameras
        self.cameras = []
        
        # Fly camera animation
        self.fly_step = 0
        self.fly_start = None
        self.fly_end = None
        self.fly_camera_index = 0  # Which camera to fly
        self.fly_steps = 10
        self.fly_dialog = None  # Reference to fly dialog window
        self.fly_vrp1_stored = {}  # Store VRP1 for each camera
        self.fly_vrp2_stored = {}  # Store VRP2 for each camera
        
        # Mouse interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.vrp_sensitivity = 0.01
        self.prp_sensitivity = 0.01
        self.active_camera_index = None  # Track which camera is being controlled
        
        # Initialize field states
        self.update_rotation_state()
        self.update_scale_state()
        
        # Load cameras.txt on startup
        self.load_cameras()

    def create_spinner_entry(self, parent, default_value, width, step=1):
        frame = tk.Frame(parent, bg="#e0e0e0")
        frame.pack(side=tk.LEFT)
        
        entry = tk.Entry(frame, width=width)
        entry.insert(0, default_value)
        entry.pack(side=tk.LEFT)
        
        spinner_frame = tk.Frame(frame, bg="#e0e0e0")
        spinner_frame.pack(side=tk.LEFT)
        tk.Button(spinner_frame, text="▲", width=1, height=0, font=("Arial", 6), 
                 command=lambda: self.adjust_spinner(entry, step), pady=0).pack(side=tk.TOP)
        tk.Button(spinner_frame, text="▼", width=1, height=0, font=("Arial", 6), 
                 command=lambda: self.adjust_spinner(entry, -step), pady=0).pack(side=tk.TOP)
        
        return entry

    def adjust_spinner(self, entry, delta):
        try:
            current_value = float(entry.get())
            new_value = current_value + delta
            entry.delete(0, tk.END)
            entry.insert(0, str(new_value))
        except ValueError:
            pass

    def load_cameras(self):
        # Load camera parameters from cameras.txt
        self.cameras.clear()
        try:
            with open("cameras.txt", "r") as f:
                current_camera = None
                
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0].startswith('//'):
                        continue
                    
                    if parts[0] == 'c':  # Start new camera
                        if current_camera is not None:
                            current_camera.compute_projection_matrix()
                            self.cameras.append(current_camera)
                        current_camera = cl_camera()
                    
                    elif current_camera is not None:
                        if parts[0] == 'i':  # Camera name
                            current_camera.name = ' '.join(parts[1:])
                        elif parts[0] == 't':  # Camera type
                            current_camera.type = parts[1].lower()
                        elif parts[0] == 'r':  # VRP
                            current_camera.vrp = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif parts[0] == 'n':  # VPN
                            current_camera.vpn = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif parts[0] == 'u':  # VUP
                            current_camera.vup = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif parts[0] == 'p':  # PRP
                            current_camera.prp = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        elif parts[0] == 'w':  # View volume
                            current_camera.view_volume = [float(parts[i]) for i in range(1, 7)]
                        elif parts[0] == 's':  # Viewport
                            current_camera.viewport = tuple([float(parts[i]) for i in range(1, 5)])
                
                # Add the last camera
                if current_camera is not None:
                    current_camera.compute_projection_matrix()
                    self.cameras.append(current_camera)
            
            # If no cameras loaded, create a default one
            if not self.cameras:
                default_cam = cl_camera()
                default_cam.compute_projection_matrix()
                self.cameras.append(default_cam)
            
            print(f"Loaded {len(self.cameras)} camera(s) from cameras.txt")
            for i, cam in enumerate(self.cameras):
                print(f"  Camera {i+1}: {cam.name if cam.name else 'Unnamed'} ({cam.type})")
        
        except FileNotFoundError:
            print("cameras.txt not found, using default camera")
            default_cam = cl_camera()
            default_cam.compute_projection_matrix()
            self.cameras.append(default_cam)

    def browse_file_clicked(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("all files", "*"), ("text files", "*.txt")])
        if self.file_path:
            self.filename_entry.delete(0, tk.END)
            self.filename_entry.insert(0, self.file_path)

    def load_button_clicked(self):
        filepath = self.filename_entry.get()
        if filepath:
            self.load_file(filepath)

    def get_viewport_at_position(self, x, y):
        # Determine which camera's viewport contains the given canvas coordinates
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        
        # Convert canvas coordinates to normalized coordinates [0,1]
        norm_x = x / c_width
        norm_y = 1 - (y / c_height)  # Flip y-axis
        
        # Check each camera's viewport
        for i, camera in enumerate(self.cameras):
            vxmin, vymin, vxmax, vymax = camera.viewport
            if vxmin <= norm_x <= vxmax and vymin <= norm_y <= vymax:
                return i
        
        return None

    def left_mouse_pressed(self, event):
        # Record starting mouse position and determine active camera for left-click drag (VRP control)
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Determine which viewport was clicked
        self.active_camera_index = self.get_viewport_at_position(event.x, event.y)
        
        if self.active_camera_index is not None:
            print(f"Controlling camera {self.active_camera_index + 1}: {self.cameras[self.active_camera_index].name}")

    def left_mouse_dragged(self, event):
        # Change VRP of active camera based on left-click mouse drag
        if not self.vertices or not self.cameras:
            return
        
        # Don't allow mouse interaction during animation
        if self.animation_transform is not None or self.fly_start is not None:
            return
        
        # Only update if we have an active camera
        if self.active_camera_index is None or self.active_camera_index >= len(self.cameras):
            return
        
        # Calculate mouse movement
        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y
        
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Get the active camera
        camera = self.cameras[self.active_camera_index]
        
        # Compute the u, v, n basis vectors for the camera
        vpn = camera.vpn / np.linalg.norm(camera.vpn)
        vup = camera.vup / np.linalg.norm(camera.vup)
        
        n = vpn
        u = np.cross(vup, n)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        
        # Transform mouse movement into world coordinates
        # dx corresponds to movement along u axis (horizontal in view)
        # dy corresponds to movement along v axis (vertical in view)
        # Moving the VRP in the same direction as the mouse makes the object appear to move that way
        # dy is negated because screen y increases downward but v increases upward
        delta_vrp = -dx * self.vrp_sensitivity * u + dy * self.vrp_sensitivity * v
        
        # Update VRP in world coordinates
        camera.vrp += delta_vrp
        
        # Recompute projection matrix and redraw
        camera.compute_projection_matrix()
        self.draw_objects()

    def right_mouse_pressed(self, event):
        # Record starting mouse position and determine active camera for right-click drag (PRP z control)
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Determine which viewport was clicked
        self.active_camera_index = self.get_viewport_at_position(event.x, event.y)
        
        if self.active_camera_index is not None:
            print(f"Controlling PRP of camera {self.active_camera_index + 1}: {self.cameras[self.active_camera_index].name}")

    def right_mouse_dragged(self, event):
        # Change PRP z-coordinate of active camera based on right-click mouse drag
        if not self.vertices or not self.cameras:
            return
        
        # Don't allow mouse interaction during animation
        if self.animation_transform is not None or self.fly_start is not None:
            return
        
        # Only update if we have an active camera
        if self.active_camera_index is None or self.active_camera_index >= len(self.cameras):
            return
        
        # Calculate mouse movement (use vertical movement for z)
        dy = event.y - self.mouse_y
        
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Update active camera's PRP z
        self.cameras[self.active_camera_index].prp[2] += dy * self.prp_sensitivity
        
        # Recompute projection matrix and redraw
        self.cameras[self.active_camera_index].compute_projection_matrix()
        self.draw_objects()

    def canvas_resized(self, event=None):
        if self.canvas.find_all():
            self.draw_objects()

    def load_file(self, filename):
        self.vertices.clear()
        self.faces.clear()
        try:
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == "v":
                        x, y, z = map(float, parts[1:4])
                        self.vertices.append(np.array([x, y, z, 1.0]))
                    elif parts[0] == "f":
                        indices = list(map(int, parts[1:]))
                        self.faces.append(indices)
            
            self.original_vertices = [v.copy() for v in self.vertices]
            self.draw_objects()
        except Exception as e:
            print(f"Error loading file: {e}")

    def parse_vector(self, text):
        text = text.strip().replace('[', '').replace(']', '')
        parts = text.split(',')
        return [float(p.strip()) for p in parts]

    def update_rotation_state(self):
        if self.rot_axis.get() == "line":
            self.ax.config(state="normal", fg="black")
            self.bx.config(state="normal", fg="black")
        else:
            self.ax.config(state="readonly", fg="gray")
            self.bx.config(state="readonly", fg="gray")

    def update_scale_state(self):
        if self.scale_ratio_type.get() == "all":
            self.scale_all.config(state="normal", fg="black")
            self.scale_xyz.config(state="readonly", fg="gray")
        else:
            self.scale_all.config(state="readonly", fg="gray")
            self.scale_xyz.config(state="normal", fg="black")

    def clip_line_parallel_3d(self, p1, p2):
        # Clip a line segment against parallel view volume [-1,1] x [-1,1] x [0,1]
        # Cohen-Sutherland 3D clipping
        def compute_code(p):
            code = 0
            if p[0] < -1: code |= 1  # left
            if p[0] > 1: code |= 2   # right
            if p[1] < -1: code |= 4  # bottom
            if p[1] > 1: code |= 8   # top
            if p[2] < 0: code |= 16  # near
            if p[2] > 1: code |= 32  # far
            return code
        
        x1, y1, z1 = p1[0], p1[1], p1[2]
        x2, y2, z2 = p2[0], p2[1], p2[2]
        
        code1 = compute_code(p1)
        code2 = compute_code(p2)
        
        accept = False
        
        while True:
            if code1 == 0 and code2 == 0:  # Both inside
                accept = True
                break
            elif (code1 & code2) != 0:  # Both outside same region
                break
            else:
                # Pick a point outside
                code_out = code1 if code1 != 0 else code2
                
                # Find intersection
                if code_out & 1:  # left
                    y = y1 + (y2 - y1) * (-1 - x1) / (x2 - x1)
                    z = z1 + (z2 - z1) * (-1 - x1) / (x2 - x1)
                    x = -1
                elif code_out & 2:  # right
                    y = y1 + (y2 - y1) * (1 - x1) / (x2 - x1)
                    z = z1 + (z2 - z1) * (1 - x1) / (x2 - x1)
                    x = 1
                elif code_out & 4:  # bottom
                    x = x1 + (x2 - x1) * (-1 - y1) / (y2 - y1)
                    z = z1 + (z2 - z1) * (-1 - y1) / (y2 - y1)
                    y = -1
                elif code_out & 8:  # top
                    x = x1 + (x2 - x1) * (1 - y1) / (y2 - y1)
                    z = z1 + (z2 - z1) * (1 - y1) / (y2 - y1)
                    y = 1
                elif code_out & 16:  # near
                    x = x1 + (x2 - x1) * (0 - z1) / (z2 - z1)
                    y = y1 + (y2 - y1) * (0 - z1) / (z2 - z1)
                    z = 0
                elif code_out & 32:  # far
                    x = x1 + (x2 - x1) * (1 - z1) / (z2 - z1)
                    y = y1 + (y2 - y1) * (1 - z1) / (z2 - z1)
                    z = 1
                
                # Update point and code
                if code_out == code1:
                    x1, y1, z1 = x, y, z
                    code1 = compute_code(np.array([x, y, z]))
                else:
                    x2, y2, z2 = x, y, z
                    code2 = compute_code(np.array([x, y, z]))
        
        if accept:
            return np.array([x1, y1, z1, 1]), np.array([x2, y2, z2, 1])
        return None, None

    def clip_line_perspective_3d(self, p1, p2, camera):
        # Clip a line segment against perspective view volume
        # After perspective transformation, the canonical view volume is:
        # -z <= x <= z, -z <= y <= z, 0 < z <= 1
        
        # Use small epsilon for near plane
        zmin = 0.0001  # Very small positive value to avoid division by zero
        zmax = 1.0     # Far plane at z=1
        
        def compute_code(p):
            code = 0
            # Check z first to avoid issues
            if p[2] <= zmin:
                code |= 16  # Behind near plane
                return code
            if p[2] > zmax:
                code |= 32  # Beyond far plane
            
            if p[0] < -p[2]: code |= 1   # left: x < -z
            if p[0] > p[2]: code |= 2    # right: x > z
            if p[1] < -p[2]: code |= 4   # bottom: y < -z
            if p[1] > p[2]: code |= 8    # top: y > z
            
            return code
        
        x1, y1, z1 = p1[0], p1[1], p1[2]
        x2, y2, z2 = p2[0], p2[1], p2[2]
        
        code1 = compute_code(p1)
        code2 = compute_code(p2)
        
        accept = False
        
        max_iterations = 20
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            if code1 == 0 and code2 == 0:
                accept = True
                break
            elif (code1 & code2) != 0:
                break
            else:
                code_out = code1 if code1 != 0 else code2
                
                # Find intersection
                x, y, z = x1, y1, z1  # Initialize to avoid uninitialized variable
                
                if code_out & 1:  # left: x = -z
                    denom = (x2 - x1) + (z2 - z1)
                    if abs(denom) < 1e-10:
                        break
                    t = (-z1 - x1) / denom
                    if t < 0 or t > 1:
                        break
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    z = z1 + t * (z2 - z1)
                elif code_out & 2:  # right: x = z
                    denom = (x2 - x1) - (z2 - z1)
                    if abs(denom) < 1e-10:
                        break
                    t = (z1 - x1) / denom
                    if t < 0 or t > 1:
                        break
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    z = z1 + t * (z2 - z1)
                elif code_out & 4:  # bottom: y = -z
                    denom = (y2 - y1) + (z2 - z1)
                    if abs(denom) < 1e-10:
                        break
                    t = (-z1 - y1) / denom
                    if t < 0 or t > 1:
                        break
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    z = z1 + t * (z2 - z1)
                elif code_out & 8:  # top: y = z
                    denom = (y2 - y1) - (z2 - z1)
                    if abs(denom) < 1e-10:
                        break
                    t = (z1 - y1) / denom
                    if t < 0 or t > 1:
                        break
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    z = z1 + t * (z2 - z1)
                elif code_out & 16:  # near: z = zmin
                    if abs(z2 - z1) < 1e-10:
                        break
                    t = (zmin - z1) / (z2 - z1)
                    if t < 0 or t > 1:
                        break
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    z = zmin
                elif code_out & 32:  # far: z = zmax
                    if abs(z2 - z1) < 1e-10:
                        break
                    t = (zmax - z1) / (z2 - z1)
                    if t < 0 or t > 1:
                        break
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    z = zmax
                
                # Ensure z is positive
                if z <= zmin:
                    break
                
                # Update point and code
                if code_out == code1:
                    x1, y1, z1 = x, y, z
                    code1 = compute_code(np.array([x, y, z]))
                else:
                    x2, y2, z2 = x, y, z
                    code2 = compute_code(np.array([x, y, z]))
        
        if accept:
            return np.array([x1, y1, z1, 1]), np.array([x2, y2, z2, 1])
        return None, None

    def map_to_viewport(self, xw, yw, viewport):
        # Map from normalized window coords [-1,1] to viewport
        vxmin, vymin, vxmax, vymax = viewport
        
        # Map from [-1,1] to [0,1]
        u = (xw + 1) / 2
        v = (yw + 1) / 2
        
        # Map to viewport
        xv = vxmin + u * (vxmax - vxmin)
        yv = vymin + v * (vymax - vymin)
        
        # Map to canvas
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        xc = xv * c_width
        yc = (1 - yv) * c_height
        return xc, yc

    def draw_objects(self):
        if not (self.vertices and self.faces and self.cameras):
            return
        
        self.canvas.delete("all")
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        
        # Draw each camera's viewport
        for camera in self.cameras:
            vxmin, vymin, vxmax, vymax = camera.viewport
            
            # Calculate viewport rectangle in canvas coordinates
            x0, y0 = vxmin * c_width, (1 - vymax) * c_height
            x1, y1 = vxmax * c_width, (1 - vymin) * c_height
            
            # Draw viewport boundary
            self.canvas.create_rectangle(x0 - 2, y0 - 2, x1 + 2, y1 + 2, 
                                        outline="red", width=3, dash=(5, 3))
            self.canvas.create_rectangle(x0, y0, x1, y1, 
                                        outline="black", width=2)
            
            # Add corner markers
            marker_size = 8
            self.canvas.create_line(x0, y0, x0 + marker_size, y0, fill="red", width=2)
            self.canvas.create_line(x0, y0, x0, y0 + marker_size, fill="red", width=2)
            self.canvas.create_line(x1, y0, x1 - marker_size, y0, fill="red", width=2)
            self.canvas.create_line(x1, y0, x1, y0 + marker_size, fill="red", width=2)
            self.canvas.create_line(x0, y1, x0 + marker_size, y1, fill="red", width=2)
            self.canvas.create_line(x0, y1, x0, y1 - marker_size, fill="red", width=2)
            self.canvas.create_line(x1, y1, x1 - marker_size, y1, fill="red", width=2)
            self.canvas.create_line(x1, y1, x1, y1 - marker_size, fill="red", width=2)
            
            # Add label
            text_bg_x0 = x0 + 2
            text_bg_y0 = y0 + 2
            text_bg_x1 = x0 + 180
            text_bg_y1 = y0 + 35
            self.canvas.create_rectangle(text_bg_x0, text_bg_y0, text_bg_x1, text_bg_y1, 
                                        fill="white", outline="black", width=1)
            
            label = camera.name if camera.name else f"{camera.type.capitalize()}"
            self.canvas.create_text(x0 + 5, y0 + 5, 
                                text=f"{label}", 
                                anchor="nw", 
                                font=("Arial", 9, "bold"),
                                fill="blue")
            
            coord_text = f"[{vxmin:.2f}, {vymin:.2f}, {vxmax:.2f}, {vymax:.2f}]"
            self.canvas.create_text(x0 + 5, y0 + 20, 
                                text=coord_text, 
                                anchor="nw", 
                                font=("Arial", 7),
                                fill="darkgreen")
            
            # Process each face for this camera
            for face in self.faces:
                # Get vertices and apply camera projection
                projected_verts = []
                for idx in face:
                    v = self.vertices[idx - 1]
                    v_proj = camera.matrix @ v
                    projected_verts.append(v_proj)
                
                # Draw edges of the face
                num_verts = len(projected_verts)
                for i in range(num_verts):
                    p1 = projected_verts[i]
                    p2 = projected_verts[(i + 1) % num_verts]
                    
                    # Clip based on camera type
                    if camera.type.lower() == "perspective":
                        clipped_p1, clipped_p2 = self.clip_line_perspective_3d(p1, p2, camera)
                        
                        if clipped_p1 is not None and clipped_p2 is not None:
                            # Check if z values are valid for perspective division
                            if clipped_p1[2] > 0.0001 and clipped_p2[2] > 0.0001:
                                # Perspective division
                                x1_proj = clipped_p1[0] / clipped_p1[2]
                                y1_proj = clipped_p1[1] / clipped_p1[2]
                                x2_proj = clipped_p2[0] / clipped_p2[2]
                                y2_proj = clipped_p2[1] / clipped_p2[2]
                                
                                # Map to viewport and draw
                                xc1, yc1 = self.map_to_viewport(x1_proj, y1_proj, camera.viewport)
                                xc2, yc2 = self.map_to_viewport(x2_proj, y2_proj, camera.viewport)
                                self.canvas.create_line(xc1, yc1, xc2, yc2, fill="black", width=1)
                    else:  # Parallel
                        clipped_p1, clipped_p2 = self.clip_line_parallel_3d(p1, p2)
                        
                        if clipped_p1 is not None and clipped_p2 is not None:
                            # Drop z and map to viewport
                            xc1, yc1 = self.map_to_viewport(clipped_p1[0], clipped_p1[1], camera.viewport)
                            xc2, yc2 = self.map_to_viewport(clipped_p2[0], clipped_p2[1], camera.viewport)
                            self.canvas.create_line(xc1, yc1, xc2, yc2, fill="black", width=1)

    def get_rotation_matrix(self, angle_deg, axis, a=None, b=None):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        if axis == "x":
            return np.array([
                [1, 0, 0, 0],
                [0, c, -s, 0],
                [0, s, c, 0],
                [0, 0, 0, 1]
            ])
        elif axis == "y":
            return np.array([
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1]
            ])
        elif axis == "z":
            return np.array([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            a = np.array(a[:3])
            b = np.array(b[:3])
            d = b - a
            d_norm = np.linalg.norm(d)
            if d_norm == 0:
                return np.eye(4)
            k = d / d_norm
            kx, ky, kz = k
            R = np.array([
                [c + kx**2*(1-c), kx*ky*(1-c)-kz*s, kx*kz*(1-c)+ky*s, 0],
                [ky*kx*(1-c)+kz*s, c + ky**2*(1-c), ky*kz*(1-c)-kx*s, 0],
                [kz*kx*(1-c)-ky*s, kz*ky*(1-c)+kx*s, c + kz**2*(1-c), 0],
                [0,0,0,1]
            ])
            T1 = np.eye(4); T1[:3,3] = -a
            T2 = np.eye(4); T2[:3,3] = a
            return T2 @ R @ T1

    def get_scale_matrix(self, sx, sy, sz, pivot):
        px, py, pz = pivot
        T1 = np.array([
            [1, 0, 0, -px],
            [0, 1, 0, -py],
            [0, 0, 1, -pz],
            [0, 0, 0, 1]
        ])
        S = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])
        T2 = np.array([
            [1, 0, 0, px],
            [0, 1, 0, py],
            [0, 0, 1, pz],
            [0, 0, 0, 1]
        ])
        return T2 @ S @ T1

    def get_translation_matrix(self, tx, ty, tz):
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])

    def animate(self):
        if self.animation_step >= self.animation_total_steps:
            self.original_vertices = [v.copy() for v in self.vertices]
            self.animation_step = 0
            self.animation_transform = None
            return
        
        self.animation_step += 1
        fraction = self.animation_step / self.animation_total_steps
        transform = self.animation_transform(fraction)
        
        for i in range(len(self.vertices)):
            self.vertices[i] = transform @ self.original_vertices[i]
        
        self.draw_objects()
        self.root.after(50, self.animate)

    def animate_fly(self):
        if self.fly_step >= self.fly_steps:
            self.fly_step = 0
            
            # Swap VRP1 and VRP2 after animation completes
            if self.fly_camera_index in self.fly_vrp1_stored:
                temp = self.fly_vrp1_stored[self.fly_camera_index].copy()
                self.fly_vrp1_stored[self.fly_camera_index] = self.fly_vrp2_stored[self.fly_camera_index].copy()
                self.fly_vrp2_stored[self.fly_camera_index] = temp
                
                # Update dialog if it's still open
                if self.fly_dialog and self.fly_dialog.winfo_exists():
                    vrp1 = self.fly_vrp1_stored[self.fly_camera_index]
                    vrp2 = self.fly_vrp2_stored[self.fly_camera_index]
                    
                    self.fly_current_vrp_entry.delete(0, tk.END)
                    self.fly_current_vrp_entry.insert(0, f"[{vrp1[0]:.1f},{vrp1[1]:.1f},{vrp1[2]:.1f}]")
                    
                    self.fly_vrp2_entry.delete(0, tk.END)
                    self.fly_vrp2_entry.insert(0, f"[{vrp2[0]:.1f},{vrp2[1]:.1f},{vrp2[2]:.1f}]")
            
            self.fly_start = None
            self.fly_end = None
            return
        
        self.fly_step += 1
        t = self.fly_step / self.fly_steps
        
        # Interpolate VRP position of selected camera
        if self.fly_start is not None and self.fly_end is not None and self.cameras:
            self.cameras[self.fly_camera_index].vrp = self.fly_start + t * (self.fly_end - self.fly_start)
            self.cameras[self.fly_camera_index].compute_projection_matrix()
            self.draw_objects()
        
        self.root.after(30, self.animate_fly)

    def rotate_clicked(self):
        try:
            angle = float(self.rot_angle.get())
            axis = self.rot_axis.get()
            steps = max(1, int(self.rot_steps.get()))
            
            a = None
            b = None
            if axis == "line":
                a = self.parse_vector(self.ax.get())
                b = self.parse_vector(self.bx.get())
            
            self.animation_step = 0
            self.animation_total_steps = steps
            self.animation_transform = lambda t: self.get_rotation_matrix(t * angle, axis, a, b)
            self.animate()
        except Exception as e:
            print(f"Rotation error: {e}")

    def scale_clicked(self):
        try:
            pivot = self.parse_vector(self.scale_point.get())
            steps = max(1, int(self.scale_steps.get()))
            
            if self.scale_ratio_type.get() == "all":
                s = float(self.scale_all.get())
                sx, sy, sz = s, s, s
            else:
                scales = self.parse_vector(self.scale_xyz.get())
                sx, sy, sz = scales[0], scales[1], scales[2]
            
            self.animation_step = 0
            self.animation_total_steps = steps
            self.animation_transform = lambda t: self.get_scale_matrix(
                1 + t * (sx - 1), 1 + t * (sy - 1), 1 + t * (sz - 1), pivot
            )
            self.animate()
        except Exception as e:
            print(f"Scale error: {e}")

    def translate_clicked(self):
        try:
            trans = self.parse_vector(self.trans_vec.get())
            tx, ty, tz = trans[0], trans[1], trans[2]
            steps = max(1, int(self.trans_steps.get()))
            
            self.animation_step = 0
            self.animation_total_steps = steps
            self.animation_transform = lambda t: self.get_translation_matrix(t * tx, t * ty, t * tz)
            self.animate()
        except Exception as e:
            print(f"Translation error: {e}")

    def fly_camera_clicked(self):
        # Open a dialog to configure and execute fly camera animation
        # Create dialog window
        self.fly_dialog = tk.Toplevel(self.root)
        self.fly_dialog.title("Select Camera to Fly")
        self.fly_dialog.geometry("500x120")
        self.fly_dialog.resizable(False, False)
        
        # Make it modal
        self.fly_dialog.transient(self.root)
        self.fly_dialog.grab_set()
        
        # Main frame
        main_frame = tk.Frame(self.fly_dialog, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Row 1: Camera selection
        row1 = tk.Frame(main_frame, bg="#f0f0f0")
        row1.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(row1, text="Select Camera to Fly:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 5))
        
        # Create camera options list
        camera_names = []
        for i, cam in enumerate(self.cameras):
            name = cam.name if cam.name else f"Camera {i+1}"
            camera_names.append(name)
        
        self.fly_camera_var = tk.StringVar(value=camera_names[0] if camera_names else "")
        
        # Camera dropdown with arrow
        cam_select_frame = tk.Frame(row1, bg="#f0f0f0")
        cam_select_frame.pack(side=tk.LEFT)
        
        camera_dropdown = tk.OptionMenu(cam_select_frame, self.fly_camera_var, *camera_names)
        camera_dropdown.config(width=15)
        camera_dropdown.pack(side=tk.LEFT)
        
        tk.Label(cam_select_frame, text="→", bg="#f0f0f0", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        
        # Row 2: VRP inputs and Steps
        row2 = tk.Frame(main_frame, bg="#f0f0f0")
        row2.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(row2, text="Current VRP(x,y,z):", bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 5))
        self.fly_current_vrp_entry = tk.Entry(row2, width=12)
        self.fly_current_vrp_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(row2, text="VRP 2(x,y,z):", bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 5))
        self.fly_vrp2_entry = tk.Entry(row2, width=12)
        self.fly_vrp2_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        tk.Label(row2, text="Steps:", bg="#f0f0f0").pack(side=tk.LEFT, padx=(0, 5))
        self.fly_steps_entry = tk.Entry(row2, width=5)
        self.fly_steps_entry.insert(0, "10")
        self.fly_steps_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        # Fly button
        fly_execute_button = tk.Button(row2, text="Fly", bg="#66b3ff", width=8,
                                       command=self.execute_fly_camera)
        fly_execute_button.pack(side=tk.LEFT, padx=5)
        
        # Update VRP displays when camera selection changes
        def update_vrp_display(*args):
            selected_name = self.fly_camera_var.get()
            for i, cam in enumerate(self.cameras):
                name = cam.name if cam.name else f"Camera {i+1}"
                if name == selected_name:
                    # Initialize stored VRPs if not present
                    if i not in self.fly_vrp1_stored:
                        self.fly_vrp1_stored[i] = cam.vrp.copy()
                        self.fly_vrp2_stored[i] = np.array([1.0, 1.0, 1.0])
                    
                    # Update entry fields
                    vrp1 = self.fly_vrp1_stored[i]
                    vrp2 = self.fly_vrp2_stored[i]
                    
                    self.fly_current_vrp_entry.delete(0, tk.END)
                    self.fly_current_vrp_entry.insert(0, f"[{vrp1[0]:.1f},{vrp1[1]:.1f},{vrp1[2]:.1f}]")
                    
                    self.fly_vrp2_entry.delete(0, tk.END)
                    self.fly_vrp2_entry.insert(0, f"[{vrp2[0]:.1f},{vrp2[1]:.1f},{vrp2[2]:.1f}]")
                    break
        
        self.fly_camera_var.trace('w', update_vrp_display)
        update_vrp_display()  # Initial update
    
    def execute_fly_camera(self):
        # Execute the fly camera animation with selected parameters
        try:
            # Get selected camera index
            selected_name = self.fly_camera_var.get()
            camera_index = None
            for i, cam in enumerate(self.cameras):
                name = cam.name if cam.name else f"Camera {i+1}"
                if name == selected_name:
                    camera_index = i
                    break
            
            if camera_index is None:
                print("No camera selected")
                return
            
            # Get current values from entries
            vrp1_text = self.fly_current_vrp_entry.get()
            vrp2_text = self.fly_vrp2_entry.get()
            vrp1 = np.array(self.parse_vector(vrp1_text))
            vrp2 = np.array(self.parse_vector(vrp2_text))
            
            # Store the values for swapping after animation
            self.fly_vrp1_stored[camera_index] = vrp1.copy()
            self.fly_vrp2_stored[camera_index] = vrp2.copy()
            
            # Get steps
            steps = int(self.fly_steps_entry.get())
            
            # Set up animation (fly from current VRP1 to VRP2)
            self.fly_start = vrp1.copy()
            self.fly_end = vrp2.copy()
            self.fly_camera_index = camera_index
            self.fly_steps = steps
            self.fly_step = 0
            
            # Don't close dialog - keep it open for repeated clicks
            
            # Start animation
            self.animate_fly()
            
        except Exception as e:
            print(f"Fly camera error: {e}")

world = cl_world()
world.root.mainloop()