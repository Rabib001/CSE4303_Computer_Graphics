# Husain, Rabib
# 1002_053_770
# 2025_10_5
# Assignment_02_01


import numpy as np
import tkinter as tk
from tkinter import filedialog

class cl_world:
    def __init__(self):
        # Main Window
        self.root = tk.Tk()
        self.root.title("tk")
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

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="yellow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.canvas_resized)
        self.canvas.bind("<ButtonPress-1>", self.mouse_pressed)
        self.canvas.bind("<B1-Motion>", self.mouse_dragged)

        # Data
        self.vertices = []
        self.faces = []
        self.window = None
        self.viewport = None
        self.file_path = None
        self.original_vertices = []
        self.animation_step = 0
        self.animation_total_steps = 0
        self.animation_transform = None
        
        # Mouse rotation
        self.mouse_x = 0
        self.mouse_y = 0
        self.rotation_x = 0
        self.rotation_y = 0
        
        # Initialize field states
        self.update_rotation_state()
        self.update_scale_state()

    def create_spinner_entry(self, parent, default_value, width, step=1):
        # Create an entry with spinner buttons
        frame = tk.Frame(parent, bg="#e0e0e0")
        frame.pack(side=tk.LEFT)
        
        entry = tk.Entry(frame, width=width)
        entry.insert(0, default_value)
        entry.pack(side=tk.LEFT)
        
        # Spinner buttons in vertical frame
        spinner_frame = tk.Frame(frame, bg="#e0e0e0")
        spinner_frame.pack(side=tk.LEFT)
        tk.Button(spinner_frame, text="▲", width=1, height=0, font=("Arial", 6), 
                 command=lambda: self.adjust_spinner(entry, step), pady=0).pack(side=tk.TOP)
        tk.Button(spinner_frame, text="▼", width=1, height=0, font=("Arial", 6), 
                 command=lambda: self.adjust_spinner(entry, -step), pady=0).pack(side=tk.TOP)
        
        return entry

    def adjust_spinner(self, entry, delta):
        # Adjust spinner value by delta
        try:
            current_value = float(entry.get())
            new_value = current_value + delta
            entry.delete(0, tk.END)
            entry.insert(0, str(new_value))
        except ValueError:
            pass

    def browse_file_clicked(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("all files", "*"), ("text files", "*.txt")])
        if self.file_path:
            self.filename_entry.delete(0, tk.END)
            self.filename_entry.insert(0, self.file_path)

    def load_button_clicked(self):
        filepath = self.filename_entry.get()
        if filepath:
            self.load_file(filepath)
            self.rotation_x = 0
            self.rotation_y = 0

    def mouse_pressed(self, event):
        # Record starting mouse position
        self.mouse_x = event.x
        self.mouse_y = event.y

    def mouse_dragged(self, event):
        # Rotate model based on mouse drag
        if not self.original_vertices:
            return
        
        # Don't allow mouse rotation during animation
        if self.animation_transform is not None:
            return
        
        # Calculate rotation based on mouse movement
        dx = event.x - self.mouse_x
        dy = event.y - self.mouse_y
        
        self.mouse_x = event.x
        self.mouse_y = event.y
        
        # Update cumulative rotation
        self.rotation_y += dx * 0.5  # Horizontal drag rotates around Y axis
        self.rotation_x += dy * 0.5  # Vertical drag rotates around X axis
        
        # Apply rotation incrementally from current state
        rot_y = self.get_rotation_matrix(dx * 0.5, "y")
        rot_x = self.get_rotation_matrix(dy * 0.5, "x")
        combined_rotation = rot_y @ rot_x
        
        for i in range(len(self.vertices)):
            self.vertices[i] = combined_rotation @ self.vertices[i]
        
        # Update original vertices to maintain cumulative transformations
        self.original_vertices = [v.copy() for v in self.vertices]
        
        self.draw_objects()

    def canvas_resized(self, event=None):
        if self.canvas.find_all():
            self.draw_objects(event)

    def load_file(self, filename):
        self.vertices.clear()
        self.faces.clear()
        self.window = None
        self.viewport = None
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
                    elif parts[0] == "w":
                        self.window = tuple(map(float, parts[1:]))
                    elif parts[0] == "s":
                        self.viewport = tuple(map(float, parts[1:]))
            self.original_vertices = [v.copy() for v in self.vertices]
            self.rotation_x = 0
            self.rotation_y = 0
            self.draw_objects()
        except Exception as e:
            print(f"Error loading file: {e}")

    def parse_vector(self, text):
        # Parse [x,y,z] format
        text = text.strip().replace('[', '').replace(']', '')
        parts = text.split(',')
        return [float(p.strip()) for p in parts]

    def update_rotation_state(self):
        # Enable/disable A and B entry fields based on rotation axis selection
        if self.rot_axis.get() == "line":
            self.ax.config(state="normal", fg="black")
            self.bx.config(state="normal", fg="black")
        else:
            self.ax.config(state="readonly", fg="gray")
            self.bx.config(state="readonly", fg="gray")

    def update_scale_state(self):
        # Enable/disable scale entry fields based on scale type selection
        if self.scale_ratio_type.get() == "all":
            self.scale_all.config(state="normal", fg="black")
            self.scale_xyz.config(state="readonly", fg="gray")
        else:
            self.scale_all.config(state="readonly", fg="gray")
            self.scale_xyz.config(state="normal", fg="black")

    def map_to_viewport(self, xw, yw):
        wxmin, wymin, wxmax, wymax = self.window
        vxmin, vymin, vxmax, vymax = self.viewport
        xv = vxmin + ((xw - wxmin) / (wxmax - wxmin)) * (vxmax - vxmin)
        yv = vymin + ((yw - wymin) / (wymax - wymin)) * (vymax - vymin)
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        xc = xv * c_width
        yc = (1 - yv) * c_height
        return xc, yc

    def draw_objects(self):
        if not (self.vertices and self.faces and self.window and self.viewport):
            return
        self.canvas.delete("all")
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        vxmin, vymin, vxmax, vymax = self.viewport
        x0, y0 = vxmin * c_width, (1 - vymax) * c_height
        x1, y1 = vxmax * c_width, (1 - vymin) * c_height
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

        for face in self.faces:
            points = []
            for idx in face:
                xw, yw, _ = self.vertices[idx - 1][:3]
                xc, yc = self.map_to_viewport(xw, yw)
                points.append((xc, yc))
            if len(points) >= 2:
                self.canvas.create_polygon(points, outline="black", fill="", width=2)

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
        else:  # Arbitrary line AB
            a = np.array(a[:3])
            b = np.array(b[:3])
            d = b - a
            d_norm = np.linalg.norm(d)
            if d_norm == 0:
                return np.eye(4)
            d = d / d_norm
            dx, dy, dz = d
            ax, ay, az = a
            
            l = np.sqrt(dy**2 + dz**2)
            if l < 1e-10:
                return self.get_rotation_matrix(angle_deg, "x")
            
            T1 = np.array([
                [1, 0, 0, -ax],
                [0, 1, 0, -ay],
                [0, 0, 1, -az],
                [0, 0, 0, 1]
            ])
            Rx = np.array([
                [1, 0, 0, 0],
                [0, dz/l, -dy/l, 0],
                [0, dy/l, dz/l, 0],
                [0, 0, 0, 1]
            ])
            Ry = np.array([
                [l, 0, -dx, 0],
                [0, 1, 0, 0],
                [dx, 0, l, 0],
                [0, 0, 0, 1]
            ])
            Rz = np.array([
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            T2 = np.array([
                [1, 0, 0, ax],
                [0, 1, 0, ay],
                [0, 0, 1, az],
                [0, 0, 0, 1]
            ])
            return T2 @ np.linalg.inv(Ry) @ np.linalg.inv(Rx) @ Rz @ Rx @ Ry @ T1

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
            # Animation complete - update original_vertices to current state
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

world = cl_world()
world.root.mainloop()