#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append(r"C:\Users\admin\Anaconda3\envs\east-ocr\Lib\site-packages")
import cv2
import numpy as np
import keras_ocr
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Initialize OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

def extract_text(image):
    predictions = pipeline.recognize([image])
    extracted_text = [text for text, _ in predictions[0]]
    return " ".join(extracted_text)

# Load EAST model
east_model_path = "frozen_east_text_detection.pb"
net = cv2.dnn.readNet(east_model_path)

def detect_text_regions(image):
    orig = image.copy()
    (H, W) = image.shape[:2]
    
    newW, newH = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)
    
    image = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                      "feature_fusion/concat_3"])
    
    rectangles, confidences = decode_predictions(scores, geometry)
    return rectangles, confidences, rW, rH, orig

def decode_predictions(scores, geometry):
    rects = []
    confidences = []
    
    for y in range(scores.shape[2]):
        for x in range(scores.shape[3]):
            if scores[0, 0, y, x] < 0.5:
                continue
            
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos, sin = np.cos(angle), np.sin(angle)
            
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            
            endX = int(offsetX + (cos * w) + (sin * h))
            endY = int(offsetY - (sin * w) + (cos * h))
            startX = int(endX - w)
            startY = int(endY - h)
            
            rects.append((startX, startY, endX, endY))
            confidences.append(scores[0, 0, y, x])
    
    return rects, confidences

def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

def save_as_file():
    text = text_output.get("1.0", tk.END)
    try:
        file_path = filedialog.asksaveasfilename(
            initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Data Folder'),
            title="Save As",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if file_path:
            with open(file_path, "w") as file:
                file.write(text)
            messagebox.showinfo("Save Successful", f"Text saved to {file_path} successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

def load_and_convert_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            process_image(gray_image)
        else:
            messagebox.showerror("Error", "Failed to load image.")
    else:
        messagebox.showerror("Error", "No image selected.")

def capture_image():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        process_image(gray_image)
    else:
        messagebox.showerror("Error", "Failed to capture image from camera.")

def save_image():
    global displayed_image
    if displayed_image:
        try:
            file_path = filedialog.asksaveasfilename(
                initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Images Folder'),
                title="Save Image",
                filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")),
                defaultextension=".png"
            )
            if file_path:
                displayed_image.save(file_path)
                messagebox.showinfo("Save Successful", f"Image saved to {file_path} successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the image: {e}")
    else:
        messagebox.showerror("Error", "No image to save.")

# Initialize GUI
root = tk.Tk()
root.title("OCR Image Processing")

load_button = tk.Button(root, text="Load Image", command=load_and_convert_image)
load_button.pack(pady=10)

capture_button = tk.Button(root, text="Capture Image from Camera", command=capture_image)
capture_button.pack(pady=5)

panel = tk.Label(root)
panel.pack()

text_output = tk.Text(root, height=10, width=50)
text_output.config(state=tk.DISABLED)
text_output.pack(pady=10)

copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.pack(pady=5)

save_text_button = tk.Button(root, text="Save As Text", command=save_as_file)
save_text_button.pack(side=tk.LEFT, padx=5, pady=5)

save_image_button = tk.Button(root, text="Save Image", command=save_image)
save_image_button.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




