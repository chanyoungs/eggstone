from __future__ import division

import os

from tkinter import Label, Button, ttk, Tk, Scale, HORIZONTAL, messagebox, filedialog

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.colors import hsv_to_rgb

import contour_detection_blue
import preprocess

import importlib
importlib.reload(contour_detection_blue)
importlib.reload(preprocess)

from contour_detection_blue import filter_blue
from preprocess import preprocess

root = "defective"
paths = []
for set in range(5):
    for i in range(16):
        paths.append([
            os.path.join(root, str(set+1), "Side1", f'{i+1}.jpg'),
            os.path.join(root, str(set+1), "Side2", f'{i+1}.jpg')
        ])

# Setup window
window=Tk()
window.title("HSV mask hsv_lower and hsv_upper boundary values")
# window.geometry('350x200')

current_row = 1
# Instruction
Label(window, text="Move the sliders to adjust the HSV mask hsv_lower and hsv_upper boundary values", font=("Arial Bold", 20), wraplength=600).grid(row=current_row, columnspan=3)
current_row += 1
 
# Boundary labels
Label(window, text="Lower bound", font=("Arial Bold", 20), wraplength=600).grid(column=1, row=current_row)
Label(window, text="Upper bound", font=("Arial Bold", 20), wraplength=600).grid(column=2, row=current_row)
current_row += 1

# HSV class storing the HSV values
class GUI():
    def __init__(self, hsv_lower=[100, 100, 100], hsv_upper=[255, 255, 255], lum_lower=0, lum_upper=255, kernel_size=10):
        self.values = {
            'hsv_lower': hsv_lower,
            'hsv_upper': hsv_upper,
            'lum_lower': lum_lower,
            'lum_upper': lum_upper,
            'kernel_size': kernel_size
        }
        self.image_index = 0
        self.no_images = 10
        
        # Figure
        self.fig1, self.fig2, self.fig3 = Figure(figsize=(1, 1)), Figure(figsize=(1, 1)), Figure(figsize=(self.no_images, 3))
        self.canvas1, self.canvas2, self.canvas3 = FigureCanvasTkAgg(self.fig1, master=window), FigureCanvasTkAgg(self.fig2, master=window), FigureCanvasTkAgg(self.fig3, master=window)
        self.canvas1.get_tk_widget().grid(column=1, row=current_row)
        self.canvas2.get_tk_widget().grid(column=2, row=current_row)
        self.canvas3.get_tk_widget().grid(row=0, columnspan=3)

        self.ax1 = self.fig1.add_subplot(1, 1, 1)
        self.ax2 = self.fig2.add_subplot(1, 1, 1)

        self.ax3s = []
        for i in range(self.no_images):
            self.ax3s.append([
                self.fig3.add_subplot(2, self.no_images, i+1),
                self.fig3.add_subplot(2, self.no_images, i+self.no_images+1)
            ])

        self.ax3s[0][0].set_title('Original')
        self.ax3s[0][1].set_title('Filtered')
        
        self.update_figure()

    def update_factory(self, type, hsv_index=0):
        def update(val):
            if type in ['lum_lower', 'lum_upper', 'kernel_size']:
                self.values[type] = int(val)
            else:
                self.values[type][hsv_index] = int(val)
            self.update_figure()
        return update

    def next_image(self):
        if self.image_index + self.no_images < len(paths):
            self.image_index += self.no_images
            self.update_figure()

    def previous_image(self):
        if self.image_index - self.no_images > 0:
            self.image_index -= self.no_images
            self.update_figure()

    def update_figure(self):
        self.ax1.clear()
        self.ax2.clear()

        for i in range(self.no_images):
            for n in range(2):
                self.ax3s[i][n].clear()

        imgs = []
        for i in range(self.image_index, self.image_index + self.no_images):
            for n in range(2):
                imgs.append(filter_blue(
                    paths[i][n],
                    hsv_lower=self.values['hsv_lower'],
                    hsv_upper=self.values['hsv_upper'],
                    lum_lower=self.values['lum_lower'],
                    lum_upper=self.values['lum_upper'],
                    kernel_size=2*self.values['kernel_size']+1))
        
        self.ax1.imshow(hsv_to_rgb((np.array(self.values['hsv_lower']).reshape(1,1,3)) / 255))
        self.ax2.imshow(hsv_to_rgb((np.array(self.values['hsv_upper']).reshape(1,1,3)) / 255))

        for i in range(self.no_images):
            for n in range(2):
                self.ax3s[i][n].imshow(imgs[i][n])
                self.ax3s[i][n].axis('off')

        self.ax1.axis('off')
        self.ax2.axis('off')

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        
gui = GUI()

# HSV sliders
sliders = {}
current_row += 1
init_col = 0
hsv_types = ["hsv_lower", "hsv_upper"]
hsv_labels = ["Hue", "Saturation", "Value"]

for t in range(len(hsv_types)):
    sliders[hsv_types[t]] = []
    for i in range(3):
        if t == 0:
            Label(window, text=hsv_labels[i], font=("Arial Bold", 20)).grid(column=init_col, row=current_row+i)

        slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, command=gui.update_factory(hsv_types[t], i), cursor = 'hand2')
        slider.set(gui.values[hsv_types[t]][i])
        slider.grid(column=t+init_col+1, row=current_row+i)
        sliders[hsv_types[t]].append(slider)
current_row += 3

# Luminosity sliders
lum_types = ["lum_lower", "lum_upper"]
Label(window, text="Luminosity", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)
for t in range(len(lum_types)):
    slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, command=gui.update_factory(lum_types[t]), cursor = 'hand2')
    slider.set(gui.values[lum_types[t]])
    slider.grid(column=t+init_col+1, row=current_row)
    sliders[lum_types[t]] = slider
current_row += 2

# Kernel slider
Label(window, text="Kernel size", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)
slider = Scale(window, from_=1, to=99, orient=HORIZONTAL, command=gui.update_factory("kernel_size"), cursor = 'hand2')
slider.set(gui.values["kernel_size"])
slider.grid(column=init_col+1, row=current_row)
sliders["kernel_size"] = slider
current_row += 1

# Image index
Label(window, text="Image", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)    
Button(window, text="Previous", command=gui.previous_image).grid(column=1, row=current_row)
Button(window, text="Next", command=gui.next_image).grid(column=2, row=current_row)
current_row += 1

# Open/Save label
Label(window, text="Open/Save values", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)

# Open function
def open_callback():
    path = filedialog.askopenfilename(initialdir = "./boundaries",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
    with open(path, 'r') as dic_txt:
        dic = eval(dic_txt.read())
        
    for boundary in ['hsv_lower', 'hsv_upper']:
        for i in range(3):
            sliders[boundary][i].set(int(dic[boundary][i]))
    
    sliders['kernel_size'].set(int(dic['kernel_size']))

Button(window, text="OPEN", command=open_callback).grid(column=1, row=current_row)
        
# Save function
def save_callback():
    f = filedialog.asksaveasfile(initialdir = "./boundaries", defaultextension=".txt")
    if f is None:
        return
    f.write(str(gui.values))
    f.close()
        
Button(window, text="SAVE", command=save_callback).grid(column=2, row=current_row)
current_row += 1

# Define progressbar
progressbar=ttk.Progressbar(window, orient="horizontal", mode="determinate")
progressbar.grid(column=2, row=current_row)


# Run all function
def run_all_callback():
    preprocess(
        progressbar=progressbar,
        paths=paths,
        hsv_lower=gui.values['hsv_lower'],
        hsv_upper=gui.values['hsv_upper'],
        lum_lower=gui.values['lum_lower'],
        lum_upper=gui.values['lum_upper'],
        kernel_size=2*gui.values['kernel_size']+1)

Label(window, text="Preprocess all", font=("Arial Bold", 20)).grid(column=init_col, row=current_row)
Button(window, text="Run!", command=run_all_callback).grid(column=1, row=current_row)
current_row += 1

window.mainloop()
