from __future__ import division

import os

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.colors import hsv_to_rgb

from contour_detection_blue import filter_blue

root = "defective"
paths = []
for set in range(5):
    for i in range(16):
        paths.append([
            os.path.join(root, str(set+1), "Side1", f'{i+1}.jpg'),
            os.path.join(root, str(set+1), "Side2", f'{i+1}.jpg')
        ])

index = 0

# Setup window
window=Tk()
window.title("HSV mask lower and upper boundary values")
# window.geometry('350x200')

current_row = 1
# Instruction
Label(window, text="Move the sliders to adjust the HSV mask lower and upper boundary values", font=("Arial Bold", 20), wraplength=600).grid(row=current_row, columnspan=3)
current_row += 1

# Boundary labels
Label(window, text="Lower bound", font=("Arial Bold", 20), wraplength=600).grid(column=1, row=current_row)
Label(window, text="Upper bound", font=("Arial Bold", 20), wraplength=600).grid(column=2, row=current_row)
current_row += 1

# HSV class storing the HSV values
class HSV():
    def __init__(self, lower=[100, 100, 100], upper=[255, 255, 255], kernel_size=10):
        self.values = {
            'lower': lower,
            'upper': upper,
            'kernel_size': kernel_size,
            'image_index': 0
        }
        
        # Figure
        self.fig1, self.fig2, self.fig3 = Figure(figsize=(1, 1)), Figure(figsize=(1, 1)), Figure(figsize=(6, 6))
        self.canvas1, self.canvas2, self.canvas3 = FigureCanvasTkAgg(self.fig1, master=window), FigureCanvasTkAgg(self.fig2, master=window), FigureCanvasTkAgg(self.fig3, master=window)
        self.canvas1.get_tk_widget().grid(column=1, row=current_row)
        self.canvas2.get_tk_widget().grid(column=2, row=current_row)
        self.canvas3.get_tk_widget().grid(row=0, columnspan=3)

        self.ax1 = self.fig1.add_subplot(1,1,1)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax3_1 = self.fig3.add_subplot(221)
        self.ax3_2 = self.fig3.add_subplot(222)
        self.ax3_3 = self.fig3.add_subplot(223)
        self.ax3_4 = self.fig3.add_subplot(224)

        self.ax3_1.set_title('Original')
        self.ax3_2.set_title('Filtered')
        
        self.update_figure()

    def update_factory(self, type, hsv_index=0):
        def update(val):
            if type == 'kernel_size':
                self.values[type] = int(val)
            else:
                self.values[type][hsv_index] = int(val)
            self.update_figure()
        return update

    def next_image(self):
        self.values['image_index'] += 1
        self.update_figure()

    def previous_image(self):
        if hsv.values['image_index'] > 0:
            self.values['image_index'] -= 1
            self.update_figure()

    def update_figure(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3_1.clear()
        self.ax3_2.clear()
        self.ax3_3.clear()
        self.ax3_4.clear()

        img_side1, img_filtered_side1 = filter_blue(
            paths[self.values['image_index']][0], lower=self.values['lower'], upper=self.values['upper'], kernel_size=2*self.values['kernel_size']+1)
        img_side2, img_filtered_side2 = filter_blue(
            paths[self.values['image_index']][1], lower=self.values['lower'], upper=self.values['upper'], kernel_size=2*self.values['kernel_size']+1)
        self.ax1.imshow(hsv_to_rgb((np.array(self.values['lower']).reshape(1,1,3)) / 255))
        self.ax2.imshow(hsv_to_rgb((np.array(self.values['upper']).reshape(1,1,3)) / 255))
        self.ax3_1.imshow(img_side1)
        self.ax3_2.imshow(img_filtered_side1)
        self.ax3_3.imshow(img_side2)
        self.ax3_4.imshow(img_filtered_side2)

        self.ax1.axis('off')
        self.ax2.axis('off')
        self.ax3_1.axis('off')
        self.ax3_2.axis('off')
        self.ax3_3.axis('off')
        self.ax3_4.axis('off')

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        
hsv = HSV()

# HSV sliders
sliders = {}
current_row += 1
init_col = 0
types = ["lower", "upper"]
hsv_labels = ["Hue", "Saturation", "Value"]

for t in range(len(types)):
    sliders[types[t]] = []
    for i in range(3):
        if t == 0:
            Label(window, text=hsv_labels[i], font=("Arial Bold", 20)).grid(column=init_col, row=current_row+i)

        slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, command=hsv.update_factory(types[t], i), cursor = 'hand2')
        slider.set(int(hsv.values[types[t]][i]))
        sliders[types[t]].append(slider)
        sliders[types[t]][i].grid(column=t+init_col+1, row=current_row+i)

current_row += 3

# Kernel slider
Label(window, text="Kernel size", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)
slider = Scale(window, from_=1, to=99, orient=HORIZONTAL, command=hsv.update_factory("kernel_size"), cursor = 'hand2')
slider.set(int(hsv.values["kernel_size"]))
sliders["kernel_size"] = slider
sliders["kernel_size"].grid(column=init_col+1, row=current_row)
current_row += 1

# Image index
Label(window, text="Image", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)    
Button(window, text="Previous", command=hsv.previous_image).grid(column=1, row=current_row)
Button(window, text="Next", command=hsv.next_image).grid(column=2, row=current_row)
current_row +=1

# Open/Save label
Label(window, text="Open/Save values", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)

# Open function
def open_callback():
    path = filedialog.askopenfilename(initialdir = "./boundaries",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
    with open(path, 'r') as dic_txt:
        dic = eval(dic_txt.read())
        
    for boundary in ['lower', 'upper']:
        for i in range(3):
            sliders[boundary][i].set(int(dic[boundary][i]))

Button(window, text="OPEN", command=open_callback).grid(column=1, row=current_row)
        
# Save function
def save_callback():
    f = filedialog.asksaveasfile(initialdir = "./boundaries", defaultextension=".txt")
    if f is None:
        return
    f.write(str(hsv.values))
    f.close()
        
Button(window, text="SAVE", command=save_callback).grid(column=2, row=current_row)

window.mainloop()
