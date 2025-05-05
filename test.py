
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import Tk, Label, Button, filedialog, PhotoImage, Canvas
from PIL import Image, ImageTk

# Load the trained model
model = load_model('trained_model_DNN1.h5')

# Class names (update as per your dataset)
class_names = ['algal leaf', 'Anthracnose', 'bird eye spot', 'brown blight', 'gray light',
               'healthy', 'red leaf spot', 'white spot']

# Preprocess the image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Predict the class
def predict(img_path):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

# Function to handle image selection and display
def select_image():
    file_path = filedialog.askopenfilename(title="Select Image File",
                                           filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        # Load and display the image
        img = Image.open(file_path)
        img_resized = img.resize((300, 300))  
        img_tk = ImageTk.PhotoImage(img_resized)
        canvas.image = img_tk  
        canvas.create_image(0, 0, anchor='nw', image=img_tk)

        # Predict and display result
        predicted_class, confidence = predict(file_path)
        result_label.config(text=f"Predicted: {predicted_class} ({confidence * 100:.2f}%)")
    else:
        result_label.config(text="No image selected.")

# Build the GUI
root = Tk()
root.title("Leaf Disease Classifier")

# Create canvas for image
canvas = Canvas(root, width=300, height=300)
canvas.pack()

# Button to upload image
upload_button = Button(root, text="Upload Image", command=select_image)
upload_button.pack(pady=10)

# Label to show prediction result
result_label = Label(root, text="Prediction result will appear here", font=('Arial', 14))
result_label.pack(pady=10)

# Start the GUI loop
root.mainloop()

