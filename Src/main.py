import io
from modelController import *
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from tkinter.colorchooser import askcolor
from PIL import ImageTk, Image, ImageChops
from tkinter import ttk
import tkinter as tk
from PIL import ImageGrab
root = Tk()
root.title("White Board")
root.geometry ( "750x350+150+50" )
root.configure (bg = "#f2f3f5")
root. resizable(False,False)

current_x = 0
current_y = 0
color = 'black'

def locate_xy(work):
    global current_x, current_y
    current_x = work.x
    current_y = work.y

def addLine(work):
    global current_x, current_y

    canvas.create_line((current_x,current_y,work.x,work.y),width = 5, fill = color, capstyle=ROUND, smooth=True)
    current_x, current_y = work.x, work.y

def show_color(new_color):
    global color
    color = new_color

def new_canvas():

    canvas.delete('all')
    display_pallete()

#icon
image_icon = ImageTk.PhotoImage(file = "images/GUI images/logo.png")
root.iconphoto(False, image_icon)

# color_box = ImageTk.PhotoImage(file = "boardColor.png")
# Label(root, image = color_box, bg = "#f2f3f5").place(x = 10, y = 20)

color_box = PhotoImage(file = "images/GUI images/color_section.png")
Label(root,image=color_box,bg="#f2f3f5").place(x=150, y= 10)

eraser = ImageTk.PhotoImage(file = "images/GUI images/eraser.png")
Button(root, image = eraser, bg = "#f2f3f5", command = new_canvas).place(x = 500, y = 28)

colors = Canvas(root, bg = "#ffffff", width = 300, height = 37, bd = 0)
colors.place(x = 180, y = 28)

def display_pallete():
    id = colors.create_rectangle((10,10,30,30),fill = "black")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('black'))

    id = colors.create_rectangle((40,10,60,30),fill = "gray")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('gray'))

    id = colors.create_rectangle((70,10,90,30),fill = "brown4")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('brown4'))

    id = colors.create_rectangle((100,10,120,30),fill = "red")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('red'))

    id = colors.create_rectangle((130,10,150,30),fill = "orange")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('orange'))

    id = colors.create_rectangle((160,10,180,30),fill = "green")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('green'))

    id = colors.create_rectangle((190,10,210,30),fill = "blue")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('blue'))

    id = colors.create_rectangle((220,10,240,30),fill = "purple")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('purple'))



def upload_file():
    f_types = [('Jpg Files', '.jpg'),
    ('PNG Files','.png')]   # type of files to select 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    col=1 # start from column 1
    row=3 # start from row 3 
    for f in filename:
        img=cv2.imread(f)
        img2 =cv2.resize(img, (500, 500))
        cv2.imshow("Uploaded Image",img2)
        cv2.imwrite("images/Model Temp Images/captured_snapshot_out.jpg",img)
        predictedWord=getPredictedOutsideWord()   # read the image file
        outputText = "The model predicted: " + predictedWord
        label.config(text=outputText)
        # img=ImageTk.PhotoImage(img)
        e1 =tk.Label(root)
        e1.grid(row=row,column=col)
        # e1.image = img # keep a reference! by attaching it to a widget attribute
        # e1['image']=img # Show Image
        if(col==3): # start new line after third column
            row=row+1# start wtih next row
            col=1    # start with first column
        else:       # within the same row 
            col=col+1 # increase to next column
    



display_pallete()

canvas = Canvas(root, width = 550, height = 100, background="white", cursor="hand2")
canvas.pack()
canvas.place(x=100,y=100)



canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', addLine)



root.update()
# Create a new image with the same width and height as the canvas.
# Define the area you want to capture
x = root.winfo_rootx() + canvas.winfo_x()
y = root.winfo_rooty() + canvas.winfo_y()
x1 = x + canvas.winfo_width()
y1 = y + canvas.winfo_height()
capture_region = (x, y, x1, y1)




# Define the function to capture the snapshot and send it to the machine learning model
def capture_and_send():
    # Use the canvas's postscript() method to render the canvas to a new image.
    img = canvas.postscript(width=canvas.winfo_width(), height=canvas.winfo_height())
    
    # Convert the image to a binary image file.
    with io.BytesIO() as f:
        f.write(img.encode())
        img = f.getvalue()
    
    # Convert the image to a PIL image.
    pil_img = Image.open(io.BytesIO(img))
    
    # Crop the image to the size of the canvas.
    pil_img = pil_img.crop((0, 0,410,80))
    
    # Save the image to disk.
    pil_img.save("images/Model Temp Images/captured_snapshot.jpg")
    predictedWord = getPredictedWord()
    outputText = "The model predicted: " + predictedWord
    label.config(text=outputText)
    
    



label = Label(root,text="",font=('Helvetica bold', 22))
label.pack()
label.place(relx = 0.5,
                   rely = 0.8,
                   anchor = 'center')

button = tk.Button(root, text="Predict Word", width=20, command=capture_and_send)
b1 = tk.Button(root, text='Upload Image', 
   width=20,command = lambda:upload_file())

button.pack()
b1.pack()
button.place(x = 400, y = 230)
b1.place(x=200,y=230)



root. mainloop()

