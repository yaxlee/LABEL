import math
import warnings
import collections
import tkinter as tk
from tkinter import *
import numpy as np
from tkinter import ttk,filedialog
from ttkbootstrap import Style
from tkinter.messagebox import showinfo
from PIL import Image,ImageDraw, ImageTk
from shapely.geometry import LineString
import argparse
import os
import cv2
import json
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_folder', default=None, required=True,
                        help='mask folder')
    parser.add_argument('--image_folder', default=None, required=True,
                        help='image folder')
    parser.add_argument('--feature_folder', default=None, required=True,
                        help='feature folder')
    parser.add_argument('--outf', default=None, required=True,
                        help='output folder')
    opt = parser.parse_args()
    return opt

def load_data(path):
    if os.path.exists(os.path.join(path, 'slam_data.json')):
        with open(os.path.join(path, 'slam_data.json'), 'r') as f:
            data=json.load(f)
            keyframes = data['keyframes']
            pts = np.array(
                [keyframes[k]['trans'] for k in list(keyframes.keys())], dtype=int)
            knames = [k.split('_')[0] for k in list(keyframes.keys())]
            # plan = data['floorplan']
    else:
        print('Map doesn\'t exist!')
        exit()
    return knames,pts

class Loader():
    def __init__(self,opt):
        self.image_folder=opt.image_folder
        self.mask_folder=opt.mask_folder
        self.feature_folder=opt.feature_folder
    def getitem(self,key):
        image=cv2.imread(os.path.join(self.image_folder,key+'.png'))
        mask=cv2.imread(os.path.join(self.mask_folder,key+'.png'))
        
        feature=np.load(os.path.join(self.feature_folder,key+'.npz'))['pts']
        return image,mask,feature

class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class Button(ttk.Button):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class CanvasImage:
    """ Display and zoom image """

    def __init__(self, placeholder, parent,image,key):
        """ Initialize the ImageFrame """
        self.parent = parent
        self.placeholder = placeholder
        self.key=key
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.1  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.text = None
        self.r = 3
        self.chose = {}
        self.l=[]
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        self.coords = {"x": 0, "y": 0, "x2": 0, "y2": 0}
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<Button-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<Button-3>', self.__coordinates)
        self.canvas.bind('<B1-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        self.canvas.bind('<ButtonPress-2>', self.__click)
        self.canvas.bind('<B2-Motion>', self.__drag)
        self.canvas.bind('<ButtonRelease-2>', self.__release)
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = image  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [image]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)

        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.coords(self.container)
        scale = (bbox[2] - bbox[0]) / self.imwidth
        x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale

        if (x_ < self.imwidth) and (x_ > 0) and (y_ < self.imheight) and (y_ > 0):
            x0l, x0r, y0l, y0r = (x_ - self.r) * scale + bbox[0], (x_ + self.r) * scale + bbox[0], (
                    y_ - self.r) * scale + bbox[1], (y_ + self.r) * scale + bbox[1]
            if len(self.chose.keys()) == 1:
                if (str(x_) + '-' + str(y_)) == list(self.chose.keys())[0]:
                    self.canvas.delete(self.chose[str(x_) + '-' + str(y_)]['index'])
                    self.chose.pop(str(x_) + '-' + str(y_))
            if self.text:
                self.canvas.delete(self.text)
            self.text = self.canvas.create_text(x_ * scale + bbox[0], (y_ + self.r * 2) * scale + bbox[1],
                                                fill="darkblue", font="Times 10 italic bold",
                                                text="Right click to pick")
            self.chose.update({str(x_) + '-' + str(y_): {
                'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='red', activefill='green')}})
        if len(self.chose) > 1:
            for i in list(self.chose.keys())[:-1]:
                self.canvas.delete(self.chose[i]['index'])
                self.chose.pop(i)

        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __coordinates(self, event):
        if len(self.chose) == 1:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            bbox = self.canvas.coords(self.container)
            scale = (bbox[2] - bbox[0]) / self.imwidth
            x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale
            for s in self.chose:
                list = s.split('-')
                x0, y0 = float(list[0]), float(list[1])
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r):
                    if self.key in self.parent.points:
                        self.parent.points[self.key].append([x0, y0])
                    else:
                        self.parent.points.update({self.key: [[x0, y0]]})
                    self.parent.arrange()
                    self.placeholder.destroy()

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def __release(self,event):
        bbox = self.canvas.coords(self.container)
        scale = (bbox[2] - bbox[0]) / self.imwidth
        x00, y00, x10, y10 = (self.coords["x"] - bbox[0]) / scale, (self.coords["y"] - bbox[1]) / scale, (
                    self.coords["x2"] - bbox[0]) / scale, (self.coords["y2"] - bbox[1]) / scale
        if self.key in self.parent.lines:
            self.parent.lines[self.key].append([x00, y00, x10, y10])
        else:
            self.parent.lines.update({self.key:[[x00, y00, x10, y10]]})
        self.parent.arrange()
        self.placeholder.destroy()

    def __drag(self,event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.coords["x2"] = x
        self.coords["y2"] = y
        self.canvas.coords(self.l[-1], self.coords["x"], self.coords["y"], self.coords["x2"], self.coords["y2"])

    def __click(self,event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.coords["x"] = x
        self.coords["y"] = y
        self.l.append(self.canvas.create_line(self.coords["x"], self.coords["y"], self.coords["x"], self.coords["y"],
                                              width=self.r * 2, fill='red', activefill='green'))

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll', 1, 'unit', event=event)

class Operation_window(ttk.Frame):
    def __init__(self, mainframe, parent, name,image):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Pick a point on dynamic objects on (current frame: {})'.format(name))
        self.master.geometry('1200x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        canvas = CanvasImage(self.master, parent,image,name)  # create widget
        canvas.grid(row=0, column=0)  # show widget

class Main_window(ttk.Frame):
    def __init__(self, master,opt,loader):
        ttk.Frame.__init__(self, master=master)
        self.style = ttk.Style()
        self.image_folder = opt.image_folder
        self.mask_folder = opt.mask_folder
        self.feature_folder = opt.feature_folder
        self.outf=opt.outf
        self.loader=loader
        self.keys=sorted([i.replace('.png','') for i in os.listdir(opt.image_folder)])
        self.static_num, self.quasi_num, self.dynamic_num = 0, 0, 0
        if os.path.exists(os.path.join(self.outf,'save.json')):
            with open(os.path.join(self.outf,'save.json'),'r') as f:
                save_data=json.load(f)
                self.points = save_data['points']
                self.lines = save_data['lines']
        else:
            self.points={}
            self.lines={}
        self.color=[(255,0,0),(0,255,0),(255,208,63)]
        windowWidth = self.master.winfo_reqwidth()
        windowHeight = self.master.winfo_reqheight()
        self.positionRight = int(self.master.winfo_screenwidth() / 2 - windowWidth / 2)
        self.positionDown = int(self.master.winfo_screenheight() / 2 - windowHeight / 2)
        self.master.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.master.title('Dynamic Feature Labeling GUI')
        self.pack(side="left", fill="both", expand=False)
        self.master.geometry('2000x1200')
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(3, pad=7)
        self.master.rowconfigure(3, weight=1)
        self.master.rowconfigure(6, pad=7)
        # --------------------------------------------------
        lbl = Label(self, text="Frame list:")
        lbl.grid(row=0, column=0, sticky=W, pady=4, ipadx=2)

        var2 = tk.StringVar()
        self.lb = tk.Listbox(self, listvariable=var2)

        self.scrollbar = Scrollbar(self, orient=VERTICAL)
        self.lb.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.lb.yview)

        self.lb.bind('<Double-Button-1>', lambda event, action='double':
        self.show_frame(action))
        self.lb.bind('<Return>', self.dynamic_select_window)
        self.lb.bind('<Up>', lambda event, action='up':
        self.show_frame(action))
        self.lb.bind('<Down>', lambda event, action='down':
        self.show_frame(action))

        for i in self.keys:
            self.lb.insert('end', i)

        self.scrollbar.grid(row=1, column=0, columnspan=2, rowspan=9, padx=2, sticky='sn')
        self.lb.grid(row=1, column=0, columnspan=1, rowspan=9, padx=2,
                     sticky=E + W + S + N)

        # ---------------------------------------------------
        separatorh = ttk.Separator(self, orient='horizontal')
        separatorh.grid(row=20, column=0, pady=10, ipadx=1, columnspan=40, rowspan=1, sticky="ew")
        separatorv1 = ttk.Separator(self, orient='vertical')
        separatorv1.grid(row=0, column=3, padx=10, columnspan=1, rowspan=70, sticky="sn")
        separatorv2 = ttk.Separator(self, orient='vertical')
        separatorv2.grid(row=0, column=40, ipadx=1, columnspan=1, rowspan=70, sticky="sn")
        ebtn = tk.Button(self, text='Reset Dynamic', width=16, command=self.reset_dynamic)
        ebtn.grid(row=20, column=0, padx=10, pady=18, columnspan=2, rowspan=1)
        ebtn = tk.Button(self, text='Reset Segments', width=16, command=self.reset_segment)
        ebtn.grid(row=40, column=0, padx=10, pady=18, columnspan=2, rowspan=1)
        ebtn = tk.Button(self, text='Help', width=16, command=self.help)
        ebtn.grid(row=60, column=0, padx=10, pady=18, columnspan=2, rowspan=1)
        # ---------------------------------------------------------
        label1 = Label(self, text='If you meet any bugs please contact:\nay1620@nyu.edu')
        label1.grid(row=60, column=42, padx=20, rowspan=1)

    def help(self):
        self.info = tk.Toplevel(self.master)
        self.info.geometry('800x650')
        self.info.title('Instruction')
        self.info.geometry("+{}+{}".format(self.positionRight - 300, self.positionDown - 200))
        # This will create a LabelFrame
        label_frame1 = LabelFrame(self.info, height=100, text='Steps')
        label_frame1.pack(expand='yes', fill='both')

        label1 = Label(label_frame1, text='1. Choose a frame in frame list.')
        label1.place(x=0, y=5)

        label2 = Label(label_frame1, text='2. Double click or press <Enter> to open frame and pick a feature point.')
        label2.place(x=0, y=35)

        label3 = Label(label_frame1,
                       text='3. Double click floor plan and choose corresponding point of previous feature point.')
        label3.place(x=0, y=65)

        label4 = Label(label_frame1, text='4. Repeat above until get good matching, save the slam data.')
        label4.place(x=0, y=95)

        label_frame1 = LabelFrame(self.info, height=400, text='Buttons')
        label_frame1.pack(expand='yes', fill='both', side='bottom')

        label_1 = LabelFrame(label_frame1, height=60, text='Save Animation')
        label_1.place(x=5, y=23)
        label1 = Label(label_1, text='Save a Gif animation of the whole mapping trajectory')
        label1.pack()

        label_2 = LabelFrame(label_frame1, height=40, text='Select Floor Plan')
        label_2.place(x=5, y=70)
        label2 = Label(label_2, text='Select a floor plan of your project.')
        label2.pack()

        label_3 = LabelFrame(label_frame1, height=40, text='Delete last pair')
        label_3.place(x=5, y=117)
        label3 = Label(label_3,
                       text='Permanently Delete last pairs that used to calculate transformation matrix.')
        label3.pack()

        label_4 = LabelFrame(label_frame1, height=60, text='Left\Right button')
        label_4.place(x=5, y=164)
        label4 = Label(label_4,
                       text='Left button moves the last current pair which used to compute matrix to trash,\nright button recovers fisrt pair in trash.')
        label4.pack()

        label_5 = LabelFrame(label_frame1, height=40, text='Delete all pairs')
        label_5.place(x=5, y=229)
        label5 = Label(label_5,
                       text='Permanently Delete all pairs in current and in trash.')
        label5.pack()

        label_6 = LabelFrame(label_frame1, height=40, text='Delete trash pairs')
        label_6.place(x=5, y=276)
        label6 = Label(label_6,
                       text='Permanently Delete all pairs in trash.')
        label6.pack()

        label_7 = LabelFrame(label_frame1, height=60, text='Save SLAM\Matrix')
        label_7.place(x=5, y=323)
        label7 = Label(label_7,
                       text='Save all current pairs, transformation matrix which will be load and reuse when reopen\nthe project. Save SLAM data which is used in colmap.')
        label7.pack()

        label_8 = LabelFrame(label_frame1, height=40, text='Animation')
        label_8.place(x=5, y=388)
        label8 = Label(label_8,
                       text='Create an animation gif of mapping.')
        label8.pack()

    def clear_plots(self):
        try:
            for widget in self.li.winfo_children():
                widget.destroy()
            self.li.destroy()
            self.sl7.destroy()
        except:
            pass
        try:
            self.l.destroy()
        except:
            pass

    def plot_current(self,static,quasi,dynamic):
        self.value = self.lb.get(self.lb.curselection())
        self.clear_plots()
        self.static_num+=static
        self.quasi_num+=quasi
        self.dynamic_num +=dynamic
        total=self.static_num+self.quasi_num+self.dynamic_num
        self.li = ttk.LabelFrame(self, text=f'Current frame: {self.value}')
        self.li1 = ttk.LabelFrame(self.li, text='Index')
        self.li2 = ttk.LabelFrame(self.li, text='World\ncoordinates')
        self.li3 = ttk.LabelFrame(self.li, text='Floor Plan\ncoordinates')
        if self.linear_matching:
            self.li4 = ttk.LabelFrame(self.li, text='Error')
        self.li2x = ttk.LabelFrame(self.li2, text='x')
        self.li2y = ttk.LabelFrame(self.li2, text='y')
        self.li2z = ttk.LabelFrame(self.li2, text='z')
        self.li3x = ttk.LabelFrame(self.li3, text='X')
        self.li3y = ttk.LabelFrame(self.li3, text='Y')
        self.li2x.pack(side=LEFT)
        self.li2y.pack(side=LEFT)
        self.li2z.pack(side=LEFT)
        self.li3x.pack(side=LEFT)
        self.li3y.pack(side=LEFT)
        self.li1.pack(side=LEFT)
        self.li2.pack(side=LEFT)
        self.li3.pack(side=LEFT)
        if self.linear_matching:
            self.li4.pack(side=LEFT)
        self.sl1 = Label(self.li1, text='\n')
        self.sl1.pack()
        if self.linear_matching:
            self.sl6 = Label(self.li4, text='\n')
            self.sl6.pack()
        first_pair = len(points2D) - 20
        for i, points in enumerate(points2D):
            if (len(points) > 0) and (i >= first_pair):
                dict = list(points3D.keys())[i]
                self.sl1 = Label(self.li1, text='%-s' % (i))
                self.sl1.pack()
                self.sl2 = Label(self.li2x, text='%4.1f' % points3D[dict]['lm'][0])
                self.sl2.pack()
                self.sl2 = Label(self.li2y, text='%4.1f' % points3D[dict]['lm'][1])
                self.sl2.pack()
                self.sl2 = Label(self.li2z, text='%4.1f' % points3D[dict]['lm'][2])
                self.sl2.pack()
                self.sl3 = Label(self.li3x, text='%4d' % points[0])
                self.sl3.pack()
                self.sl3 = Label(self.li3y, text='%4d' % points[1])
                self.sl3.pack()
                if (num_pairs > 2) and self.linear_matching:
                    self.sl6 = Label(self.li4, text='%2d\'%2d\'\'' % (
                    int(self.error[i]), round((self.error[i] - int(self.error[i])) * 12)))
                    self.sl6.pack()
        if (num_pairs > 2) and self.linear_matching:
            err = np.mean(self.error)
            err_feet = int(err)
            err_inch = round((err - int(err)) * 12)
            err_cm = round(err * 30.48)
            self.sl7 = Label(self, text='Mean Error:{}\'{}\'\' ({}cm)'.format(err_feet, err_inch, err_cm))
            self.sl7.grid(row=51, column=42, rowspan=1, sticky="ew")
        self.li.grid(row=1, column=42, rowspan=50, sticky="ew")

    def arrange(self):
        self.value = self.lb.get(self.lb.curselection())
        mask = self.mask_image.copy()
        draw = ImageDraw.Draw(mask)
        if self.value in self.lines:
            lines=self.lines[self.value]
            for x0,y0,x1,y1 in lines:
                draw.line([(x0,y0),(x1,y1)],fill=(0,0,0),width=3)
        mask_arr=np.array(mask)
        gray = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 70, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dyna_contours=[]
        quasi_contours=[]
        for contour in contours:
            find=False
            if self.value in self.points:
                for x, y in self.points[self.value]:
                    point = (x, y)
                    exist = cv2.pointPolygonTest(np.array(np.array(contour)), point, False)
                    if exist>0:
                        contour = np.asarray(contour)
                        dyna_contours.append(contour)
                        cv2.fillPoly(mask_arr, [contour], (0, 255, 0))
                        find=True
                        break
            if not find:
                contour = np.asarray(contour)
                quasi_contours.append(contour)
        mask=Image.fromarray(mask_arr)
        width, height = mask.size
        scale = 1226 / width
        newsize = (1226, int(height * scale))
        mask_image = mask.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(mask_image)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.bind('<Double-Button-1>', self.dynamic_select_window)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=21, column=4, columnspan=1, rowspan=40, sticky="snew")
        frame=self.frame.copy()
        draw=ImageDraw.Draw(frame)
        new_feature=[]
        static_num,quasi_num,dynamic_num=0,0,0
        for feature in self.features:
            x,y,s=feature[:3]
            x = 2*x
            y = 2*y
            point=(x,y)
            find=False
            for contour in dyna_contours:
                exist = cv2.pointPolygonTest(np.array(np.array(contour)), point, False)
                if exist > 0:
                    new_feature.append([x,y,s,2])
                    dynamic_num+=1
                    find=True
                    break
            if not find:
                for contour in quasi_contours:
                    exist = cv2.pointPolygonTest(np.array(np.array(contour)), point, False)
                    if exist > 0:
                        new_feature.append([x, y, s, 3])
                        quasi_num+=1
                        find = True
                        break
                if not find:
                    new_feature.append([x, y, s, 1])
                    static_num+=1
            draw.ellipse((x - 15 * s, y - 15 * s, x + 15 * s, y + 15 * s),
                               fill=self.color[new_feature[-1][-1]-1])
        path=os.path.join(self.outf,self.value+'.npz')
        new_feature=np.array(new_feature)
        np.savez(path,pts=new_feature)
        framew, frameh = frame.size
        scale = 1226 / framew
        newsize = (1226, int(frameh * scale))
        frame = frame.resize(newsize)
        tkimage = ImageTk.PhotoImage(frame)
        self.myvar = Label(self, image=tkimage)
        self.myvar.image = tkimage
        self.myvar.grid(row=1, column=4, columnspan=1, rowspan=10, sticky="snew")
        path=os.path.join(self.outf,'save.json')
        save_data={'lines':self.lines,'points':self.points}
        with open(path,'w') as f:
            json.dump(save_data,f)
        # self.plot_current(static_num,quasi_num,dynamic_num)

    def delete_lines(self):
        self.value = self.lb.get(self.lb.curselection())
        if self.value in self.lines:
            self.lines.pop(self.value)
        self.arrange()
        self.win.destroy()

    def reset_segment(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('380x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Do you confirm delete all segments?")
        l.grid(row=1, column=0, columnspan=2, rowspan=2,
               padx=40, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.delete_lines)
        b1.grid(row=3, column=0, columnspan=1, rowspan=3)

        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, rowspan=3)

    def delete_points(self):
        self.value = self.lb.get(self.lb.curselection())
        if self.value in self.points:
            self.points.pop(self.value)
        self.arrange()
        self.win.destroy()

    def reset_dynamic(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('380x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Do you confirm delete all dynamics?")
        l.grid(row=1, column=0, columnspan=2, rowspan=2,
               padx=40, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.delete_points)
        b1.grid(row=3, column=0, columnspan=1, rowspan=3)

        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, rowspan=3)

    def show_frame(self, action):
        self.value = self.lb.get(self.lb.curselection())
        if action == 'up':
            i = self.keys.index(self.value)
            if i > 0:
                self.value = self.keys[i - 1]
        if action == 'down':
            i = self.keys.index(self.value)
            if i < (len(self.keys) - 1):
                self.value = self.keys[i + 1]
        with Image.open(os.path.join(self.image_folder, self.value + '.png')) as frame:
            rgbimg = Image.new("RGBA", frame.size)
            rgbimg.paste(frame)
            self.frame=rgbimg
            self.features=np.load(os.path.join(self.feature_folder, self.value + '.npz'))['pts']
        mask = cv2.imread(os.path.join(self.mask_folder, self.value + '.png'))
        self.mask_image =Image.fromarray(mask*255)
        self.arrange()

    def dynamic_select_window(self, w):
        self.value = self.lb.get(self.lb.curselection())
        self.newWindow = tk.Toplevel(self.master)
        self.app1 = Operation_window(self.newWindow, parent=self, name=self.value,image=self.mask_image)

def main(opt):
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    style = Style(theme='superhero')
    root = style.master
    data_loader=Loader(opt)
    Main_window(root,opt,data_loader)
    root.mainloop()

if __name__ == '__main__':
    opt = options()
    main(opt)
