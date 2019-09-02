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

# Setup window
window=Tk()
window.title("HSV mask lower and upper boundary values")
# window.geometry('350x200')

# Instruction
instruction = Label(window, text="Move the sliders to adjust the HSV mask lower and upper boundary values", font=("Arial Bold", 20), wraplength=600)
instruction.grid(row=0, columnspan=3)

# HSV class storing the HSV values
class HSV():
    def __init__(self, lower=[100, 100, 100], upper=[255, 255, 255]):
        self.values = {
            'lower': np.array(lower),
            'upper': np.array(upper)
        }
        
        # Figure
        self.fig1, self.fig2 = Figure(figsize=(5, 4), dpi=100), Figure(figsize=(5, 4), dpi=100)
        self.canvas1, self.canvas2 = FigureCanvasTkAgg(self.fig1, master=window), FigureCanvasTkAgg(self.fig2, master=window)
        self.canvas1.get_tk_widget().grid(column=1, row=8)
        self.canvas2.get_tk_widget().grid(column=2, row=8)
        # t = np.arange(0, 3, .01)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)

        self.update_figure()

    def update_factory(self, lower_upper, hsv_index):
        def update(val):
            self.values[lower_upper][hsv_index] = val
            print(self.values)
            
            self.update_figure()
        return update
    
    def update_figure(self):
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.imshow(hsv_to_rgb(
            np.zeros((10, 10, 3)) + (self.values['lower'] / 255)))
        self.ax2.imshow(hsv_to_rgb(
            np.zeros((10, 10, 3)) + (self.values['upper'] / 255)))

        self.canvas1.draw()
        self.canvas2.draw()

        
hsv = HSV()

# HSV sliders
hsv_sliders = {}
hsv_col = 0
boundaries = ["lower", "upper"]
hsv_labels = ["Hue", "Saturation", "Value"]
for b in range(len(boundaries)):
    hsv_sliders[boundaries[b]] = []
    for i in range(3):
        if b == 0:
            Label(window, text=hsv_labels[i], font=("Arial Bold", 20)).grid(column=hsv_col, row=i+2)

        slider = Scale(window, from_=0, to=255, orient=HORIZONTAL, command=hsv.update_factory(boundaries[b], i))
        slider.set(hsv.values[boundaries[b]][i])
        hsv_sliders[boundaries[b]].append(slider)
        hsv_sliders[boundaries[b]][i].grid(column=b+hsv_col+1, row=i+2)
        
# Open function
def open_callback():
    path = filedialog.askopenfilename(initialdir = "./boundaries",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
    with open(path, 'r') as dic_txt:
        dic = eval(dic_txt.read())
        
    for boundary in ['lower', 'upper']:
        for i in range(3):
            hsv_sliders[boundary][i].set(dic[boundary][i])

open_button = Button(window, text="OPEN", command=open_callback)
open_button.grid(column=0, row=1)
        
# Save function
def save_callback():
    f = filedialog.asksaveasfile(initialdir = "./boundaries", defaultextension=".txt")
    if f is None:
        return
    f.write(str(hsv.values))
    f.close()
    
    messagebox.showinfo('Save', 'Values saved!')
        
save_button = Button(window, text="SAVE", command=save_callback)
save_button.grid(column=1, row=1)

window.mainloop()
