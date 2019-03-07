import tkinter as tk

# canvas size
HEIGHT = 1000
WIDTH = 1800
# slider range
LOW = 0
HIGH = 255
# slider length
LENGTH = 400

# root window
root = tk.Tk()

# setting the size of the canvas
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

# camera frame
cameraFrame = tk.Frame(root, bg='#80c1ff')
cameraFrame.place(relwidth=0.4, relheight=0.6, relx=0.05, rely=0.05)
# label
cameraLabel = tk.Label(cameraFrame, text='Camera Frame', bg='black', fg='white')
cameraLabel.pack()

# result frame
resultFrame = tk.Frame(root, bg='#80c1ff')
resultFrame.place(relwidth=0.4, relheight=0.6, relx=0.55, rely=0.05)
# label
resultLabel = tk.Label(resultFrame, text='Result Frame', bg='black', fg='white')
resultLabel.pack()

# HSV sliders frame
slidersFrame = tk.Frame(root, bg='#80c1ff')
slidersFrame.place(relwidth=0.3, relheight=0.2, relx=0.60, rely=0.75)
# label
slidersLabel = tk.Label(slidersFrame, text='HSV Sliders', bg='black', fg='white')
slidersLabel.pack()
# H, S and V sliders
hVar = tk.DoubleVar()
scaleH = tk.Scale(slidersFrame, length=LENGTH, variable=hVar, orient=tk.HORIZONTAL, from_=LOW, to=HIGH)
scaleH.pack(anchor=tk.CENTER, pady=6)
sVar = tk.DoubleVar()
scaleS = tk.Scale(slidersFrame, length=LENGTH, variable=sVar, orient=tk.HORIZONTAL, from_=LOW, to=HIGH)
scaleS.pack(anchor=tk.CENTER, pady=6)
vVar = tk.DoubleVar()
scaleV = tk.Scale(slidersFrame, length=LENGTH, variable=vVar, orient=tk.HORIZONTAL, from_=LOW, to=HIGH)
scaleV.pack(anchor=tk.CENTER, pady=6)


button = tk.Button(root, text="Button")
button.pack()



# start the GUI main loop
root.mainloop()