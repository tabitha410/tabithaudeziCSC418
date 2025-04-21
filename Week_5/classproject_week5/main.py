import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Global variables
category_var = None
video_frame = None
video_running = False
cap = None
video_label = None
play_btn = None
pause_btn = None

video_paths = {
    "video1": "video1.mp4",
    "video2": "video2.mp4",
    "video3": "video3.mp4"
}

# Load YOLO
# net = cv2.dnn.readNet('cfg/yolov3.weights', 'cfg/yolov3.cfg')
net = cv2.dnn.readNet('cfg/yolov3-tiny.weights', 'cfg/yolov3-tiny.cfg')   # for faster video display but less accurate
classes = []
with open('cfg/coco.names', 'r') as f:
    classes = f.read().splitlines()
output_layers_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# hover effect for buttons
def add_hover_effect(widget, bg, fg):
    def on_enter(e):
        widget['background'] = fg
        widget['foreground'] = bg

    def on_leave(e):
        widget['background'] = bg
        widget['foreground'] = fg

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)

def play_video():
    global video_running, cap, video_label
    if not video_running:
        return

    ret, frame = cap.read()        # read each video frame
    if ret:
        height, width, _ = frame.shape

        # Object Detection with YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes:
                i = int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i)
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        # Convert to ImageTk for display
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, play_video)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # Loop video if ended (i.e ret=False)
        play_video()

def start_video():
    global video_running, play_btn, pause_btn
    video_running = True
    play_video()
    play_btn.config(state=tk.DISABLED)
    pause_btn.config(state=tk.NORMAL)

def pause_video():
    global video_running, play_btn, pause_btn
    video_running = False
    pause_btn.config(state=tk.DISABLED)
    play_btn.config(state=tk.NORMAL)

def select_category():
    category = category_var.get()
    if not category:
        messagebox.showwarning("No Selection", "Please select a video category.")
        return
    if category not in video_paths:
        messagebox.showerror("Error", f"No path mapped for {category}")
        return
    open_video_frame(video_paths[category])

def open_video_frame(video_path):
    global cap, video_label, video_running, play_btn, pause_btn

    # New window for video display
    video_window = tk.Toplevel()
    video_window.title("Video Frame")
    video_window.geometry("800x800")
    video_window.configure(bg='white')

    # Video display label
    video_label = tk.Label(video_window, bg='black')
    video_label.pack(pady=10)

    # Control buttons
    control_frame = tk.Frame(video_window, bg='white')
    control_frame.pack(pady=10)

    button_style = {"font": ("Arial", 12, "bold"), "bg": "#00008B", "fg": "white", "width": 10}

    play_btn = tk.Button(control_frame, text="Play", command=start_video, state=tk.DISABLED, **button_style)
    play_btn.pack(side=tk.LEFT, padx=10)

    pause_btn = tk.Button(control_frame, text="Pause", command=pause_video, state=tk.NORMAL, **button_style)
    pause_btn.pack(side=tk.LEFT, padx=10)

    add_hover_effect(play_btn, "#00008B", "white")
    add_hover_effect(pause_btn, "#00008B", "white")

    # Load video
    cap = cv2.VideoCapture(video_path)
    video_running = True
    play_video()

def create_home_screen():
    global category_var
    root = tk.Tk()
    root.title("Object Detection Application")
    root.geometry("400x200")
    root.configure(bg='white')

    lbl = tk.Label(root, text="Select Video", font=("Arial", 14, "bold"), bg="white")
    lbl.pack(pady=(30, 10))

    category_var = tk.StringVar()
    combo = ttk.Combobox(root, textvariable=category_var, values=list(video_paths.keys()), state="readonly", font=("Arial", 12))
    combo.current(0)
    combo.pack(pady=5)

    btn = tk.Button(root, text="Select", command=select_category, font=("Arial", 12, "bold"), bg="#00008B", fg="white")
    btn.pack(pady=(15, 10))
    add_hover_effect(btn, "#00008B", "white")

    root.mainloop()

if __name__ == "__main__":
    create_home_screen()
