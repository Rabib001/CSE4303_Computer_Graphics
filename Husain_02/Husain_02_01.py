# Husain, Rabib
# 1002_053_770
# 2025_10_5
# Assignment_02_01

import numpy as np
import tkinter as tk
from tkinter import simpledialog, filedialog

class cl_world:
    def __init__(self):
        ################## Main Window ########################
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Resizable Window")
        # Set the window gemetry (Size) and Make it resizable
        self.root.geometry("400x600")
        self.root.resizable(True, True)
        ################### Top Pnael ##########################
        # Create a top frame for the button
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        # Create a button in the top panel
        self.brwose_button = tk.Button(self.top_frame, text="Browse", fg="blue", command=self.browse_file_clicked)
        self.brwose_button.pack(side=tk.LEFT)
        self.draw_button = tk.Button(self.top_frame, text="Draw", command=self.draw_button_clicked)
        self.draw_button.pack(side=tk.LEFT, padx=10, pady=10)
        ################### Canvas #############################
        # Create a canvas to draw on
        self.canvas = tk.Canvas(self.root, bg="light goldenrod")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # Bind the resize event to redraw the canvas when window is resized
        self.canvas.bind("<Configure>", self.canvas_resized)
        #################### Bottom Panel #######################
        # Create a bottom frame for displaying messages
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        # Create a lebel for showing messages
        self.message_label = tk.Label(self.bottom_frame, text="")
        self.message_label.pack(padx=10, pady=10)

        #Store Data
        self.vertices = []
        self.faces = []
        self.window = None
        self.viewport = None
        self.file_path = None

    def browse_file_clicked(self):
        self.file_path = tk.filedialog.askopenfilename(filetypes=[("allfiles", "*"), ("pythonfiles", "*.txt")])
        self.message_label.config(text=self.file_path)
        self.load_file(self.file_path)

    def draw_button_clicked(self):
        self.draw_objects()

    def canvas_resized(self,event=None):
        if self.canvas.find_all():
            self.draw_objects(event)

    def load_file(self,filename):
        # Modify this file to complete your assignment\
        #Remove all previously stored data
        self.vertices.clear()
        self.faces.clear()
        self.window = None
        self.viewport = None
        
        with open(filename, "r") as f:
            for line in f:
                #remove leading or tailing whitespace
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] =="v":  #vertex
                    x, y = float(parts[1]), float(parts[2])
                    self.vertices.append((x, y)) #ignore z
                elif parts[0] =="f":    #face
                    indices = list(map(int, parts[1:]))
                    self.faces.append(indices)
                elif parts[0] =="w":    #window
                    self.window = tuple(map(float, parts[1:]))
                elif parts[0] =="s":    #viewport
                    self.viewport = tuple(map(float, parts[1:]))    
    
    #Mapping
    def map_to_viewport(self, xw, yw):
        wxmin, wymin, wxmax, wymax = self.window
        vxmin, vymin, vxmax, vymax = self.viewport
        #Map world to normalized viewport
        xv = vxmin + ((xw - wxmin) / (wxmax - wxmin)) * (vxmax - vxmin)
        yv = vymin + ((yw - wymin) / (wymax - wymin)) * (vymax - vymin)

        #Convert normalized viewport to canvas cordinates
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        
        #Conevrt normal coordinates to pixel coordinates
        xc = xv * c_width
        yc = (1 - yv) * c_height

        return xc, yc

    def draw_objects(self,event=None):
        # Modify this file to complete your assignment
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
                xw, yw = self.vertices[idx - 1]
                xc, yc = self.map_to_viewport(xw, yw)
                points.append((xc, yc))
            self.canvas.create_polygon(points, outline="black", fill="", width=2)


# Run the tkinter main loop
world=cl_world()
world.root.mainloop()
