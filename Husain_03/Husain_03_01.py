# Husain, Rabib
# 1002_053_770
# 2025_10_19
# Assignment_03_01

import numpy as np
import tkinter as tk
from tkinter import filedialog

class cl_world:
    def __init__(self):
        # Main Window
        self.root = tk.Tk()
        self.root.title("Assignment 03 - Parallel Projection")
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
        tk.Label(self.fly_frame, text="Fly Camera from A:", bg="#e0e0e0").pack(side=tk.LEFT)
        self.fly_a = tk.Entry(self.fly_frame, width=12)
        self.fly_a.insert(0, "[0.0,0.0,0.0]")
        self.fly_a.pack(side=tk.LEFT, padx=3)
        
        tk.Label(self.fly_frame, text="to B:", bg="#e0e0e0").pack(side=tk.LEFT, padx=(3,1))
        self.fly_b = tk.Entry(self.fly_frame, width=12)
        self.fly_b.insert(0, "[1.0,1.0,1.0]")
        self.fly_b.pack(side=tk.LEFT, padx=3)
        
        self.fly_button = tk.Button(self.fly_frame, text="Fly Camera", command=self.fly_camera_clicked)
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
        self.viewport = None
        self.file_path = None
        self.original_vertices = []
        self.animation_step = 0
        self.animation_total_steps = 0
        self.animation_transform = None
        
        # Camera parameters (from cameras.txt)
        self.vrp = np.array([0.0, 0.0, 0.0])
        self.vpn = np.array([0.0, 0.0, 1.0])
        self.vup = np.array([0.0, 1.0, 0.0])
        self.prp = np.array([0.0, 0.0, 1.0])
        self.view_volume = [-1, 1, -1, 1, -1, 1]  # umin, umax, vmin, vmax, nmin, nmax
        self.parallel_matrix = None
        
        # Fly camera animation
        self.fly_step = 0
        self.fly_start = None
        self.fly_end = None
        
        # Mouse interaction
        self.mouse_x = 0
        self.mouse_y = 0
        self.vrp_sensitivity = 0.01  # Sensitivity for VRP movement
        self.prp_sensitivity = 0.01  # Sensitivity for PRP z movement
        
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
        """Load camera parameters from cameras.txt"""
        try:
            with open("cameras.txt", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    if parts[0] == 'r':  # VRP
                        self.vrp = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'n':  # VPN
                        self.vpn = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'u':  # VUP
                        self.vup = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'p':  # PRP
                        self.prp = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == 'w':  # View volume
                        self.view_volume = [float(parts[i]) for i in range(1, 7)]
                    elif parts[0] == 's':  # Viewport
                        self.viewport = tuple([float(parts[i]) for i in range(1, 5)])
            
            # If no viewport was specified, use default
            if self.viewport is None:
                self.viewport = (0.1, 0.1, 0.4, 0.4)
            
            self.compute_parallel_matrix()
            print("Cameras.txt loaded successfully")
        except FileNotFoundError:
            print("cameras.txt not found, using default camera parameters")
            self.viewport = (0.1, 0.1, 0.4, 0.4)
            self.compute_parallel_matrix()

    def compute_parallel_matrix(self):
        """Compute the composite parallel projection matrix"""
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
        
        # Step 4: Shear to make center line parallel to z-axis
        umin, umax, vmin, vmax, nmin, nmax = self.view_volume
        CW = np.array([(umin + umax) / 2, (vmin + vmax) / 2, 0])
        DOP = CW - self.prp
        
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
        
        # Composite matrix
        self.parallel_matrix = S_par @ T_par @ SH_par @ T_prp @ R @ T_vrp

    def browse_file_clicked(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("all files", "*"), ("text files", "*.txt")])
        if self.file_path:
            self.filename_entry.delete(0, tk.END)
            self.filename_entry.insert(0, self.file_path)

    def load_button_clicked(self):
        filepath = self.filename_entry.get()
        if filepath:
            self.load_file(filepath)

    def left_mouse_pressed(self, event):
        """Record starting mouse position for left-click drag (VRP control)"""
        self.mouse_x = event.x
        self.mouse_y = event.y

    def left_mouse_dragged(self, event):
        """Change VRP based on left-click mouse drag"""
        if not self.vertices:
            return
        
        # Don't allow mouse interaction during animation
        if self.animation_transform is not None or self.fly_start is not None:
            return
        
        # Calculate mouse movement
        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y
        
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Update VRP: horizontal drag affects X, vertical drag affects Y
        self.vrp[0] += dx * self.vrp_sensitivity
        self.vrp[1] += dy * self.vrp_sensitivity  # Positive dy = drag down = move down
        
        # Recompute projection matrix and redraw
        self.compute_parallel_matrix()
        self.draw_objects()

    def right_mouse_pressed(self, event):
        """Record starting mouse position for right-click drag (PRP z control)"""
        self.mouse_x = event.x
        self.mouse_y = event.y

    def right_mouse_dragged(self, event):
        """Change PRP z-coordinate based on right-click mouse drag"""
        if not self.vertices:
            return
        
        # Don't allow mouse interaction during animation
        if self.animation_transform is not None or self.fly_start is not None:
            return
        
        # Calculate mouse movement (use vertical movement for z)
        dy = event.y - self.mouse_y
        
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Update PRP z: drag down increases z, drag up decreases z
        self.prp[2] += dy * self.prp_sensitivity
        
        # Recompute projection matrix and redraw
        self.compute_parallel_matrix()
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

    def clip_3d(self, vertices):
        """Clip a polygon against the canonical view volume [-1,1] x [-1,1] x [0,1]"""
        # Clip against 6 planes
        planes = [
            (0, -1),  # x = -1
            (0, 1),   # x = 1
            (1, -1),  # y = -1
            (1, 1),   # y = 1
            (2, 0),   # z = 0
            (2, 1)    # z = 1
        ]
        
        poly = vertices
        for axis, boundary in planes:
            if len(poly) == 0:
                return []
            
            clipped = []
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                
                if axis == 2 and boundary == 0:
                    inside1 = v1[axis] >= boundary
                    inside2 = v2[axis] >= boundary
                elif axis == 2 and boundary == 1:
                    inside1 = v1[axis] <= boundary
                    inside2 = v2[axis] <= boundary
                else:
                    if boundary < 0:
                        inside1 = v1[axis] >= boundary
                        inside2 = v2[axis] >= boundary
                    else:
                        inside1 = v1[axis] <= boundary
                        inside2 = v2[axis] <= boundary
                
                if inside1 and inside2:
                    clipped.append(v2)
                elif inside1 and not inside2:
                    t = (boundary - v1[axis]) / (v2[axis] - v1[axis] + 1e-10)
                    intersect = v1 + t * (v2 - v1)
                    clipped.append(intersect)
                elif not inside1 and inside2:
                    t = (boundary - v1[axis]) / (v2[axis] - v1[axis] + 1e-10)
                    intersect = v1 + t * (v2 - v1)
                    clipped.append(intersect)
                    clipped.append(v2)
            
            poly = clipped
        
        return poly

    def map_to_viewport(self, xw, yw):
        """Map from normalized window coords [-1,1] to viewport"""
        vxmin, vymin, vxmax, vymax = self.viewport
        
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
        if not (self.vertices and self.faces and self.viewport and self.parallel_matrix is not None):
            return
        
        self.canvas.delete("all")
        
        # Draw viewport boundary with enhanced visibility
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        vxmin, vymin, vxmax, vymax = self.viewport
        
        # Calculate viewport rectangle in canvas coordinates
        x0, y0 = vxmin * c_width, (1 - vymax) * c_height
        x1, y1 = vxmax * c_width, (1 - vymin) * c_height
        
        # Draw a prominent viewport boundary
        # Outer border (thicker, darker)
        self.canvas.create_rectangle(x0 - 2, y0 - 2, x1 + 2, y1 + 2, 
                                    outline="red", width=3, dash=(5, 3))
        
        # Inner border (standard black)
        self.canvas.create_rectangle(x0, y0, x1, y1, 
                                    outline="black", width=2)
        
        # Add corner markers for better visibility
        marker_size = 8
        # Top-left corner
        self.canvas.create_line(x0, y0, x0 + marker_size, y0, fill="red", width=2)
        self.canvas.create_line(x0, y0, x0, y0 + marker_size, fill="red", width=2)
        # Top-right corner
        self.canvas.create_line(x1, y0, x1 - marker_size, y0, fill="red", width=2)
        self.canvas.create_line(x1, y0, x1, y0 + marker_size, fill="red", width=2)
        # Bottom-left corner
        self.canvas.create_line(x0, y1, x0 + marker_size, y1, fill="red", width=2)
        self.canvas.create_line(x0, y1, x0, y1 - marker_size, fill="red", width=2)
        # Bottom-right corner
        self.canvas.create_line(x1, y1, x1 - marker_size, y1, fill="red", width=2)
        self.canvas.create_line(x1, y1, x1, y1 - marker_size, fill="red", width=2)
        
        # Add informative labels
        # Background for text (white rectangle for readability)
        text_bg_x0 = x0 + 2
        text_bg_y0 = y0 + 2
        text_bg_x1 = x0 + 120
        text_bg_y1 = y0 + 20
        self.canvas.create_rectangle(text_bg_x0, text_bg_y0, text_bg_x1, text_bg_y1, 
                                    fill="white", outline="black", width=1)
        
        # Viewport label
        self.canvas.create_text(x0 + 5, y0 + 5, 
                            text="Viewport Boundary", 
                            anchor="nw", 
                            font=("Arial", 9, "bold"),
                            fill="blue")
        
        # Display viewport coordinates (optional but helpful)
        coord_text = f"[{vxmin:.2f}, {vymin:.2f}, {vxmax:.2f}, {vymax:.2f}]"
        self.canvas.create_text(x0 + 5, y1 - 5, 
                            text=coord_text, 
                            anchor="sw", 
                            font=("Arial", 7),
                            fill="darkgreen")

        # Process each face
        for face in self.faces:
            # Get vertices in world coordinates and apply parallel projection
            projected_verts = []
            for idx in face:
                v = self.vertices[idx - 1]
                v_proj = self.parallel_matrix @ v
                projected_verts.append(v_proj)
            
            # 3D Clipping
            clipped = self.clip_3d(projected_verts)
            
            if len(clipped) >= 3:
                # Drop z coordinate and map to viewport
                points = []
                for v in clipped:
                    xc, yc = self.map_to_viewport(v[0], v[1])
                    points.append((xc, yc))
                
                if len(points) >= 2:
                    self.canvas.create_polygon(points, outline="black", fill="", width=1)

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
        if self.fly_step >= 100:
            self.fly_step = 0
            
            # Swap A and B values after animation completes
            if self.fly_start is not None and self.fly_end is not None:
                # Update entry fields with swapped values
                self.fly_a.delete(0, tk.END)
                self.fly_a.insert(0, f"[{self.fly_end[0]},{self.fly_end[1]},{self.fly_end[2]}]")
                
                self.fly_b.delete(0, tk.END)
                self.fly_b.insert(0, f"[{self.fly_start[0]},{self.fly_start[1]},{self.fly_start[2]}]")
            
            self.fly_start = None
            self.fly_end = None
            return
        
        self.fly_step += 1
        t = self.fly_step / 100.0
        
        # Interpolate VRP position
        if self.fly_start is not None and self.fly_end is not None:
            self.vrp = self.fly_start + t * (self.fly_end - self.fly_start)
            self.compute_parallel_matrix()
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
        try:
            a = self.parse_vector(self.fly_a.get())
            b = self.parse_vector(self.fly_b.get())
            
            self.fly_start = np.array(a)
            self.fly_end = np.array(b)
            self.fly_step = 0
            self.animate_fly()
        except Exception as e:
            print(f"Fly camera error: {e}")

world = cl_world()
world.root.mainloop()