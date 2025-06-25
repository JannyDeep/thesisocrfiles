#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytesseract


# In[3]:


pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# In[4]:


import cv2


# In[5]:


imag=cv2.imread(r"C:\Users\janev\Desktop\acad\thesis\SAMPLE 1 REYES, MARIA.png")


# In[ ]:


import cv2

# Load an image from file
img = cv2.imread('SAMPLE 1 REYES, MARIA.png')

# Check if the image was loaded successfully
if img is not None:
    # Display the image in a window named "window"
    cv2.imshow("window", img)
    
    # Wait indefinitely for a key press
    cv2.waitKey(0)
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Error: Failed to load the image.")


# In[ ]:


text=pytesseract.image_to_string(img)


# In[ ]:


import cv2
import pytesseract  # Add this import statement

# Load an image from file
img = cv2.imread('SAMPLE 1 REYES, MARIA.png')

# Check if the image was loaded successfully
if img is not None:
    # Perform OCR on the image
    text = pytesseract.image_to_string(img)
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)
else:
    print("Error: Failed to load the image.")


# In[ ]:


import cv2
import pytesseract
from matplotlib import pyplot as plt  # Import matplotlib for image display

# Load an image from file
img = cv2.imread('SAMPLE 1 REYES, MARIA.png')

# Check if the image was loaded successfully
if img is not None:
    # Perform OCR on the image
    text = pytesseract.image_to_string(img)
    
    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)
else:
    print("Error: Failed to load the image.")


# In[ ]:


import cv2
import pytesseract
from matplotlib import pyplot as plt  # Import matplotlib for image display

# Load an image from file
img = cv2.imread('MicrosoftTeams-image.png')

# Check if the image was loaded successfully
if img is not None:
    # Perform OCR on the image
    text = pytesseract.image_to_string(img)
    
    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)
else:
    print("Error: Failed to load the image.")


# In[ ]:


import cv2
import pytesseract

# Load the image
image = cv2.imread('MicrosoftTeams-image.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set the number of OCR epochs (iterations)
num_epochs = 5

# Iterate over the specified number of epochs
for epoch in range(num_epochs):
    # Perform OCR on the grayscale image
    text = pytesseract.image_to_string(gray)
    
    # Print the extracted text for the current epoch
    print(f"Epoch {epoch + 1} Text:")
    print(text)
    
    # Optionally, you can further process or refine the OCR results here
    
    # Display the image with extracted text
    cv2.imshow('Image with Extracted Text', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:


import cv2
import pytesseract
from matplotlib import pyplot as plt  # Import matplotlib for image display

# Load an image from file
img = cv2.imread('MicrosoftTeams-image.png')

# Check if the image was loaded successfully
if img is not None:
    # Perform OCR on the image
    text = pytesseract.image_to_string(img)
    
    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis
    plt.show()
    
    # Print the extracted text
    print("Extracted Text:")
    print(text)
else:
    print("Error: Failed to load the image.")


# In[ ]:


import cv2
import pytesseract
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('SAMPLE 1 REYES, MARIA.png', cv2.IMREAD_GRAYSCALE)

# Perform OCR on the grayscale image
text = pytesseract.image_to_string(image)

# Calculate the accuracy of the text received
# For demonstration purposes, we'll assume the accuracy is 80% (replace with your actual accuracy calculation)
accuracy_ratio = 0.80

# Plot the image
plt.imshow(image, cmap='gray')
plt.title('Image with OCR Text')
plt.axis('off')
plt.show()

# Plot the accuracy as a bar chart
if accuracy_ratio == -1:
    plt.bar(['Accuracy'], [-1], color='red')
    plt.ylim(-1, 1)  # Set y-axis limits for better visualization
    plt.title('Accuracy of OCR Text')
    plt.ylabel('Accuracy Ratio')
    plt.xticks(rotation=45)
    plt.show()
else:
    plt.bar(['Accuracy'], [accuracy_ratio])
    plt.ylim(0, 1)  # Set y-axis limits for better visualization
    plt.title('Accuracy of OCR Text')
    plt.ylabel('Accuracy Ratio')
    plt.xticks(rotation=45)
    plt.show()

# Print the extracted text below the graph
print("Extracted Text:")
print(text)


# In[ ]:


import cv2
import pytesseract
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('SAMPLE 1 REYES, MARIA.png', cv2.IMREAD_GRAYSCALE)

# Set the number of OCR epochs (iterations)
num_epochs = 3

for epoch in range(num_epochs):
    # Perform OCR on the grayscale image
    text = pytesseract.image_to_string(image)

    # Calculate the accuracy of the text received
    # For demonstration purposes, we'll assume the accuracy is 80% (replace with your actual accuracy calculation)
    accuracy_ratio = 0.80

    # Plot the image
    plt.imshow(image, cmap='gray')
    plt.title(f'Epoch {epoch + 1}: Image with OCR Text')
    plt.axis('off')
    plt.show()

    # Plot the accuracy as a bar chart
    if accuracy_ratio == -1:
        plt.bar(['Accuracy'], [-1], color='red')
        plt.ylim(-1, 1)  # Set y-axis limits for better visualization
        plt.title('Accuracy of OCR Text')
        plt.ylabel('Accuracy Ratio')
        plt.xticks(rotation=45)
        plt.show()
    else:
        plt.bar(['Accuracy'], [accuracy_ratio])
        plt.ylim(0, 1)  # Set y-axis limits for better visualization
        plt.title('Accuracy of OCR Text')
        plt.ylabel('Accuracy Ratio')
        plt.xticks(rotation=45)
        plt.show()

    # Print the extracted text below the graph
    print(f"Epoch {epoch + 1} Extracted Text:")
    print(text)


# In[ ]:


import cv2
import pytesseract
import matplotlib.pyplot as plt
import re

# Load the image
image = cv2.imread('SAMPLE 1 REYES, MARIA.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Set the number of OCR epochs (iterations)
num_epochs = 3

# Initialize lists to store extracted text for each epoch
text_list = []
numerical_values = []

# Initialize previous numerical value
prev_numerical_value = None

# Iterate over the specified number of epochs
for epoch in range(num_epochs):
    # Perform OCR on the grayscale image
    text = pytesseract.image_to_string(gray_image)
    
    # Append the extracted text to the list
    text_list.append(text)
    
    # Calculate the numerical value (for demonstration purposes, we'll just use epoch number)
    numerical_value = epoch * 100  # Convert to percentage
    
    # Determine improvement or disimprovement with previous epoch
    if prev_numerical_value is not None:
        comparison = "Improved" if numerical_value < prev_numerical_value else "Disimproved"
    else:
        comparison = "N/A"

    # Store current numerical value for comparison in the next epoch
    prev_numerical_value = numerical_value

    # Append numerical value to list
    numerical_values.append(numerical_value)

    # Display the numerical value and comparison
    print(f"Epoch {epoch + 1} Numerical Value: {numerical_value}%")
    print(f"Comparison with previous epoch: {comparison}\n")

# Plot the numerical values as a scalar comparison
plt.plot(range(1, num_epochs + 1), numerical_values, marker='o')
plt.title('Scalar Comparison of Numerical Values for Each Epoch')
plt.xlabel('Epoch')
plt.ylabel('Numerical Value (%)')
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)
plt.show()

# Display the extracted text for each epoch
for epoch, text in enumerate(text_list):
    print(f"Epoch {epoch + 1} Extracted Text:")
    print(text)
    print("\n")


# In[ ]:


import cv2
import pytesseract
import matplotlib.pyplot as plt
import re

# Load the image
image = cv2.imread('SAMPLE 1 REYES, MARIA.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Set the number of OCR epochs (iterations)
num_epochs = 3

# Initialize lists to store extracted text for each epoch
text_list = []
numerical_values = []

# Initialize previous numerical value
prev_numerical_value = None

# Iterate over the specified number of epochs
for epoch in range(num_epochs):
    # Perform OCR on the grayscale image
    text = pytesseract.image_to_string(gray_image)
    
    # Append the extracted text to the list
    text_list.append(text)
    
    # Calculate the numerical value (for demonstration purposes, we'll just use epoch number)
    numerical_value = epoch * 100  # Convert to percentage
    
    # Determine improvement or disimprovement with previous epoch
    if prev_numerical_value is not None:
        comparison = "Improved" if numerical_value < prev_numerical_value else "Disimproved"
    else:
        comparison = "N/A"

    # Store current numerical value for comparison in the next epoch
    prev_numerical_value = numerical_value

    # Append numerical value to list
    numerical_values.append(numerical_value)

    # Display the numerical value and comparison
    print(f"Epoch {epoch + 1} Numerical Value: {numerical_value}%")
    print(f"Comparison with previous epoch: {comparison}\n")

# Plot the numerical values as a scalar comparison
plt.plot(range(1, num_epochs + 1), numerical_values, marker='o')
plt.title('Scalar Comparison of Numerical Values for Each Epoch')
plt.xlabel('Epoch')
plt.ylabel('Numerical Value (%)')
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)
plt.show()

# Display the extracted text for each epoch
for epoch, text in enumerate(text_list):
    print(f"Epoch {epoch + 1} Extracted Text:")
    print(text)
    print("\n")

# Check if the image improved or not
improved = numerical_values[-1] < numerical_values[0]
print(f"Did the image improve? {improved}")


# In[ ]:


import cv2
import pytesseract
import matplotlib.pyplot as plt
import re
from difflib import SequenceMatcher

# Define expected text
expected_text = """
REPUBLIC OF THE PHILIPPINES
DEPARTMENT OF TRANSPORTATION
LAND TRANSPORTATION OFFICE
DRIVER'S LICENSE
Last Name. First Name. Middle Name
REYES, MARIA SOLEDAD
Nationality Sex Date of Birth Weight (kg) Height(m)
PHL F 1985/12/10 55 1.60
Address
456 ESPANA ST., MANILA
License,No, Expiration Date Agency Code
N02-20-654321 2024/10/11 N42
Blood,Type Eyes Color
BLACK
DL Codes Conditions
1,2 NONE
"""

# Load the image
image = cv2.imread('SAMPLE 1 REYES, MARIA.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set the number of OCR epochs (iterations)
num_epochs = 3

# Initialize lists to store extracted text for each epoch
text_list = []
numerical_values = []

# Iterate over the specified number of epochs
for epoch in range(num_epochs):
    # Perform OCR on the grayscale image
    text = pytesseract.image_to_string(gray_image)
    
    # Append the extracted text to the list
    text_list.append(text)

# Function to calculate similarity score between two strings
def similarity_score(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

# Function to extract numerical values from a string
def extract_numerical_values(text):
    # Extract numerical values using regular expressions
    numerical_values = re.findall(r'\d+(\.\d+)?', text)
    return numerical_values

# Initialize similarity score
total_similarity_score = 0
num_characters = len(expected_text)

# Compare each extracted value with the expected value and compute the similarity score
for epoch, extracted_text in enumerate(text_list):
    print(f"Epoch {epoch + 1} Extracted Text:")
    print(extracted_text)
    similarity_score_epoch = 0
    for line in expected_text.split('\n'):
        if line.strip() != "":
            expected_value = " ".join(line.strip().split()[1:])
            if expected_value:  # Check if expected_value is not empty
                if expected_value in extracted_text:
                    # Split the extracted text to get the value
                    split_text = extracted_text.split(expected_value)
                    if len(split_text) > 1:
                        extracted_value = split_text[-1].split("\n")[0].split()
                        if extracted_value:
                            extracted_value = " ".join(extracted_value)
                            similarity = similarity_score(expected_value, extracted_value)
                            print(f"Expected: {expected_value} | Extracted: {extracted_value} | Similarity Score: {similarity}")
                            similarity_score_epoch += similarity
                            # Draw red outline around the detected text
                            h, w, _ = image.shape
                            boxes = pytesseract.image_to_boxes(image)
                            for b in boxes.splitlines():
                                b = b.split()
                                cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 0, 255), 1)
    similarity_score_epoch /= num_characters
    if similarity_score_epoch < 1:  # Ensure the score never goes below 1
        similarity_score_epoch = 1
    numerical_values.extend(extract_numerical_values(extracted_text))
    total_similarity_score += similarity_score_epoch
    print(f"Epoch {epoch + 1} Similarity Score: {similarity_score_epoch}\n")

# Calculate the average similarity score across epochs
average_similarity_score = total_similarity_score / num_epochs

print(f"Average Similarity Score: {average_similarity_score}/{num_characters}")

# Display the image with red outlines around detected text
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Image with Red Outlines around Detected Text')
plt.axis('off')
plt.show()


# In[ ]:


import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Function to process image, perform OCR, and display result in GUI
def process_image():
    # Load the selected image
    filename = filedialog.askopenfilename()
    if filename:
        image = cv2.imread(filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the grayscale image
        extracted_text = pytesseract.image_to_string(gray_image)

        # Display the image with red outlines around detected text
        img = Image.fromarray(gray_image)
        img = img.resize((400, 400), resample=Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)

        panel.config(image=img_tk)
        panel.image = img_tk

        # Display the extracted text in the GUI
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, extracted_text)
        text_output.config(state=tk.DISABLED)

# Function to copy text from textbox to clipboard
def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

# Create Tkinter window
root = tk.Tk()
root.title("OCR Image Processing")

# Create GUI elements
load_button = tk.Button(root, text="Load Image", command=process_image)
load_button.pack(pady=10)

panel = tk.Label(root)
panel.pack()

text_output = tk.Text(root, height=10, width=50)
text_output.config(state=tk.DISABLED)
text_output.pack(pady=10)

copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.pack(pady=5)

root.mainloop()


# In[ ]:


import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Function to process image, perform OCR, and display result in GUI
def process_image():
    # Load the selected image
    filename = filedialog.askopenfilename()
    if filename:
        image = cv2.imread(filename)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the grayscale image
        extracted_text = pytesseract.image_to_string(gray_image)

        # Get image dimensions
        h, w = gray_image.shape

        # Draw red outlines around the detected text
        boxes = pytesseract.image_to_boxes(gray_image)
        for b in boxes.splitlines():
            b = b.split()
            x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)

        # Convert the processed image with red outlines to PIL format for displaying
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image)

        # Resize the image to fit the GUI
        pil_image = pil_image.resize((400, 400), resample=Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=pil_image)

        panel.config(image=img_tk)
        panel.image = img_tk

        # Display the extracted text in the GUI
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, extracted_text)
        text_output.config(state=tk.DISABLED)

# Function to copy text from textbox to clipboard
def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

# Create Tkinter window
root = tk.Tk()
root.title("OCR Image Processing")

# Create GUI elements
load_button = tk.Button(root, text="Load Image", command=process_image)
load_button.pack(pady=10)

panel = tk.Label(root)
panel.pack()

text_output = tk.Text(root, height=10, width=50)
text_output.config(state=tk.DISABLED)
text_output.pack(pady=10)

copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.pack(pady=5)

root.mainloop()


# In[ ]:


import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Function to process image, perform OCR, and display result in GUI
def process_image():
    # Load the selected image
    filename = filedialog.askopenfilename()
    if filename:
        image = cv2.imread(filename)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the grayscale image
        extracted_text = pytesseract.image_to_string(gray_image)

        # Get image dimensions
        h, w = gray_image.shape

        # Draw red outlines around the detected text
        boxes = pytesseract.image_to_boxes(gray_image)
        for b in boxes.splitlines():
            b = b.split()
            x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)

        # Convert the processed image with red outlines to PIL format for displaying
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image)

        # Resize the image to fit the GUI
        pil_image = pil_image.resize((400, 400), resample=Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=pil_image)

        panel.config(image=img_tk)
        panel.image = img_tk

        # Display the extracted text in the GUI
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, extracted_text)
        text_output.config(state=tk.DISABLED)

# Function to copy text from textbox to clipboard
def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

# Function to save text to a file
def save_to_file():
    text = text_output.get("1.0", tk.END)
    try:
        with open("C:\\Users\\janev\\Desktop\\acad\\Trying to run The EastOCR\\DATA_FROM_OCR_EXTRACTION.txt", "w") as file:
            file.write(text)
        messagebox.showinfo("Save Successful", "Text saved to file successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

# Create Tkinter window
root = tk.Tk()
root.title("OCR Image Processing")

# Create GUI elements
load_button = tk.Button(root, text="Load Image", command=process_image)
load_button.pack(pady=10)

panel = tk.Label(root)
panel.pack()

text_output = tk.Text(root, height=10, width=50)
text_output.config(state=tk.DISABLED)
text_output.pack(pady=10)

copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.pack(pady=5)

save_button = tk.Button(root, text="Save to File", command=save_to_file)
save_button.pack(pady=5)

root.mainloop()


# In[13]:


import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Function to process image, perform OCR, and display result in GUI
def process_image():
    # Load the selected image
    filename = filedialog.askopenfilename()
    if filename:
        image = cv2.imread(filename)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the grayscale image
        extracted_text = pytesseract.image_to_string(gray_image)

        # Get image dimensions
        h, w = gray_image.shape

        # Draw red outlines around the detected text
        boxes = pytesseract.image_to_boxes(gray_image)
        for b in boxes.splitlines():
            b = b.split()
            x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)

        # Convert the processed image with red outlines to PIL format for displaying
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image)

        # Resize the image to fit the GUI
        pil_image = pil_image.resize((400, 400), resample=Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=pil_image)

        panel.config(image=img_tk)
        panel.image = img_tk

        # Display the extracted text in the GUI
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, extracted_text)
        text_output.config(state=tk.DISABLED)

# Function to copy text from textbox to clipboard
def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

# Function to save text to a file with user-defined name and location
def save_as_file():
    text = text_output.get("1.0", tk.END)
    try:
        # Ask user for file name and location
        file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Data Folder'), title="Save As", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if file_path:
            with open(file_path, "w") as file:
                file.write(text)
            messagebox.showinfo("Save Successful", f"Text saved to {file_path} successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

# Create Tkinter window
root = tk.Tk()
root.title("OCR Image Processing")

# Create GUI elements
load_button = tk.Button(root, text="Load Image", command=process_image)
load_button.pack(pady=10)

panel = tk.Label(root)
panel.pack()

text_output = tk.Text(root, height=10, width=50)
text_output.config(state=tk.DISABLED)
text_output.pack(pady=10)

copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.pack(pady=5)

save_button = tk.Button(root, text="Save As", command=save_as_file)
save_button.pack(pady=5)

root.mainloop()


# In[15]:


import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import os

# Global variable to store the currently displayed image
displayed_image = None

# Function to process image, perform OCR, and display result in GUI
def process_image(image):
    global displayed_image
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the grayscale image
    extracted_text = pytesseract.image_to_string(gray_image)

    # Get image dimensions
    h, w = gray_image.shape

    # Draw red outlines around the detected text
    boxes = pytesseract.image_to_boxes(gray_image)
    for b in boxes.splitlines():
        b = b.split()
        x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
        cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)

    # Convert the processed image with red outlines to PIL format for displaying
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(processed_image)
    displayed_image = pil_image  # Store the displayed image for saving

    # Resize the image to fit the GUI
    pil_image = pil_image.resize((400, 400), resample=Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(image=pil_image)

    panel.config(image=img_tk)
    panel.image = img_tk

    # Display the extracted text in the GUI
    text_output.config(state=tk.NORMAL)
    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, extracted_text)
    text_output.config(state=tk.DISABLED)

# Function to copy text from textbox to clipboard
def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

# Function to save text to a file with user-defined name and location
def save_as_file():
    text = text_output.get("1.0", tk.END)
    try:
        # Ask user for file name and location
        file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Data Folder'), title="Save As", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if file_path:
            with open(file_path, "w") as file:
                file.write(text)
            messagebox.showinfo("Save Successful", f"Text saved to {file_path} successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

# Function to capture image from camera
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        process_image(frame)
    else:
        messagebox.showerror("Error", "Failed to capture image from camera.")

# Function to save the displayed image
def save_image():
    global displayed_image
    if displayed_image:
        try:
            # Ask user for file name and location
            file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Images Folder'), title="Save Image", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")), defaultextension=".png")
            if file_path:
                displayed_image.save(file_path)
                messagebox.showinfo("Save Successful", f"Image saved to {file_path} successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the image: {e}")
    else:
        messagebox.showerror("Error", "No image to save.")

# Create Tkinter window
root = tk.Tk()
root.title("OCR Image Processing")

# Create GUI elements
load_button = tk.Button(root, text="Load Image", command=lambda: process_image(cv2.imread(filedialog.askopenfilename())))
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


# In[30]:


import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import os

# Global variable to store the currently displayed image
displayed_image = None

# Function to process image, perform OCR, and display result in GUI
def process_image(image):
    global displayed_image
    if image is not None and len(image.shape) == 2:  # Check if image is not None and is grayscale
        # Perform OCR on the grayscale image
        extracted_text = pytesseract.image_to_string(image)

        # Get image dimensions
        h, w = image.shape

        # Draw red outlines around the detected text
        for b in pytesseract.image_to_boxes(image).splitlines():
            b = b.split()
            x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)

        # Convert the processed image with red outlines to PIL format for displaying
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image)
        displayed_image = pil_image  # Store the displayed image for saving

        # Resize the image to fit the GUI
        pil_image = pil_image.resize((400, 400), resample=Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=pil_image)

        panel.config(image=img_tk)
        panel.image = img_tk

        # Display the extracted text in the GUI
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, extracted_text)
        text_output.config(state=tk.DISABLED)
    else:
        messagebox.showerror("Error", "Invalid input image.")

# Function to copy text from textbox to clipboard
def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

# Function to save text to a file with user-defined name and location
def save_as_file():
    text = text_output.get("1.0", tk.END)
    try:
        # Ask user for file name and location
        file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Data Folder'), title="Save As", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if file_path:
            with open(file_path, "w") as file:
                file.write(text)
            messagebox.showinfo("Save Successful", f"Text saved to {file_path} successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while saving the file: {e}")

# Function to load image from file and convert it to grayscale
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

# Function to capture image from camera with highest resolution
def capture_image():
    cap = cv2.VideoCapture(0)
    # Set highest resolution supported by the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # Convert captured image to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        process_image(gray_image)
    else:
        messagebox.showerror("Error", "Failed to capture image from camera.")

# Function to save the displayed image
def save_image():
    global displayed_image
    if displayed_image:
        try:
            # Ask user for file name and location
            file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Images Folder'), title="Save Image", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")), defaultextension=".png")
            if file_path:
                displayed_image.save(file_path)
                messagebox.showinfo("Save Successful", f"Image saved to {file_path} successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the image: {e}")
    else:
        messagebox.showerror("Error", "No image to save.")

# Create Tkinter window
root = tk.Tk()Q
root.title("OCR Image Processing")

# Create GUI elements
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


# In[6]:


import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import os


displayed_image = None

def process_image(image):
    global displayed_image
    if image is not None and len(image.shape) == 2:  
        extracted_text = pytesseract.image_to_string(image)


        h, w = image.shape

        
        for b in pytesseract.image_to_boxes(image).splitlines():
            b = b.split()
            x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)

        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image)
        displayed_image = pil_image  

       
        pil_image = pil_image.resize((1024, 576), resample=Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=pil_image)

        panel.config(image=img_tk)
        panel.image = img_tk

        # Display the extracted text in the GUI
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, extracted_text)
        text_output.config(state=tk.DISABLED)
    else:
        messagebox.showerror("Error", "Invalid input image.")

def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")


def save_as_file():
    text = text_output.get("1.0", tk.END)
    try:
       
        file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Data Folder'), title="Save As", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
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
            
            file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Images Folder'), title="Save Image", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")), defaultextension=".png")
            if file_path:
                displayed_image.save(file_path)
                messagebox.showinfo("Save Successful", f"Image saved to {file_path} successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the image: {e}")
    else:
        messagebox.showerror("Error", "No image to save.")

root = tk.Tk()
root.title("OCR Image Processing")

load_button = tk.Button(root, text="Load Image", command=load_and_convert_image)
load_button.pack(pady=10)

capture_button = tk.Button(root, text="Capture Image from Camera", command=capture_image)
capture_button.pack(pady=5)

copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.pack(pady=5)

save_text_button = tk.Button(root, text="Save As Text", command=save_as_file)
save_text_button.pack(side=tk.LEFT, padx=5, pady=5)

save_image_button = tk.Button(root, text="Save Image", command=save_image)
save_image_button.pack(side=tk.LEFT, padx=5, pady=5)

panel = tk.Label(root)
panel.pack()

text_output = tk.Text(root, height=10, width=50)
text_output.config(state=tk.DISABLED)
text_output.pack(pady=10)


root.mainloop()


# In[1]:


import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import os

# Global variables
displayed_image = None
current_mode = "light"

def process_image(image):
    global displayed_image
    if image is not None and len(image.shape) == 2:  
        extracted_text = pytesseract.image_to_string(image)

        h, w = image.shape
        
        for b in pytesseract.image_to_boxes(image).splitlines():
            b = b.split()
            x, y, x2, y2 = int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4])
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 1)

        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image)
        displayed_image = pil_image  

        pil_image = pil_image.resize((400, 400), resample=Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=pil_image)

        panel.config(image=img_tk)
        panel.image = img_tk

        # Display the extracted text in the GUI
        text_output.config(state=tk.NORMAL)
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, extracted_text)
        text_output.config(state=tk.DISABLED)
    else:
        messagebox.showerror("Error", "Invalid input image.")

def copy_to_clipboard():
    text = text_output.get("1.0", tk.END)
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copy Successful", "Text copied to clipboard.")

def save_as_file():
    text = text_output.get("1.0", tk.END)
    try:
        file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Data Folder'), title="Save As", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
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
    cap = cv2.VideoCapture(0)
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
            file_path = filedialog.asksaveasfilename(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop', 'acad', 'Trying to run The EastOCR', 'Saved Images Folder'), title="Save Image", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")), defaultextension=".png")
            if file_path:
                displayed_image.save(file_path)
                messagebox.showinfo("Save Successful", f"Image saved to {file_path} successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while saving the image: {e}")
    else:
        messagebox.showerror("Error", "No image to save.")

def set_light_mode():
    global current_mode
    current_mode = "light"
    root.config(bg="white")
    panel.config(bg="white")
    text_output.config(bg="white", fg="black")

def set_dark_mode():
    global current_mode
    current_mode = "dark"
    root.config(bg="black")
    panel.config(bg="black")
    text_output.config(bg="black", fg="white")

root = tk.Tk()
root.title("OCR ImageProcessing")
root.config(bg="white")

# Menu bar
menu_bar = tk.Menu(root)

# Mode menu
mode_menu = tk.Menu(menu_bar, tearoff=0)
mode_menu.add_command(label="Light Mode", command=set_light_mode)
mode_menu.add_command(label="Dark Mode", command=set_dark_mode)

# Add mode menu to menu bar
menu_bar.add_cascade(label="Mode", menu=mode_menu)

# Configure menu bar
root.config(menu=menu_bar)

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




