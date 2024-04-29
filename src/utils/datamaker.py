import os
import sys
import numpy as np
from tkinter import *

class DataMaker:
    def __init__(self, path=None):
        self.on_flag = False
        self.name = "S001C001P001R001A001"
        self.count = 0
        if path is None:
            self.path = "./"
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            self.path = path
        self.property = ["001", "001", "001", "001", "001"]
        
    def inputbox(self):
        gui = Tk()
        scene = StringVar(gui, value=self.property[0])
        camera = StringVar(gui, value=self.property[1])
        person = StringVar(gui, value=self.property[2])
        repeat = StringVar(gui, value=self.property[3])
        action = StringVar(gui, value=self.property[4])
        gui.title('dataset maker')
        gui.geometry('350x350')
        text1 = Label(gui, text="Scene")
        text1.grid(row=1, column=1)
        box1 = Entry(gui, textvariable=scene)
        box1.grid(row=1, column=2)
        text2 = Label(gui, text="Camera")
        text2.grid(row=2, column=1)
        box2 = Entry(gui, textvariable=camera)
        box2.grid(row=2, column=2)
        text3 = Label(gui, text="Person")
        text3.grid(row=3, column=1)
        box3 = Entry(gui, textvariable=person)
        box3.grid(row=3, column=2)
        text4 = Label(gui, text="Repeat")
        text4.grid(row=4, column=1)
        box4 = Entry(gui, textvariable=repeat)
        box4.grid(row=4, column=2)
        text5 = Label(gui, text="Action")
        text5.grid(row=5, column=1)
        box5 = Entry(gui, textvariable=action)
        box5.grid(row=5, column=2)
        def submit():
            self.property = [scene.get(), camera.get(), person.get(), repeat.get(), action.get()]
            gui.destroy()

        submit_button = Button(gui, text="submit", command=submit)
        submit_button.grid(row=6, column=1)
        gui.mainloop()

    def start(self):
        self.inputbox()
        
        self.name = 'S{}C{}P{}R{}A{}'.format(self.property[0],
                                             self.property[1],
                                             self.property[2],
                                             self.property[3],
                                             self.property[4])
        if not os.path.exists(self.path+self.name):
            os.makedirs(self.path+self.name)
        self.on_flag = True
        self.count = 0
        print("Data clip on.")
        pass

    def end(self):
        self.on_flag = False
        self.count = 0
        print("Data clip off.")
        pass

    def put(self, frame):
        if self.on_flag:
            np.save(self.path+self.name+'/raw_data_{}.npy'.format(self.count), frame)
            self.count += 1
