import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import numpy as np

# Train the model using the MNIST dataset (Step 1)
def train_model():
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize the images
    
    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 image
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Output 10 classes (digits 0-9)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    
    # Save the trained model
    model.save('digit_recognition_model.h5')
    print("Model trained and saved as 'digit_recognition_model.h5'")

# Load the trained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Function to preprocess the drawn image
def preprocess_image(img):
    # Convert to grayscale, resize to 28x28, and invert if needed
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels (MNIST size)
    img = ImageOps.invert(img)  # Invert the image colors (white background, black digits)
    
    # Create a numpy array and normalize it
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (1 for grayscale)
    
    return img_array

# Function to handle drawing on the canvas
def draw(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
    draw_img.line([x1, y1, x2, y2], fill='black', width=5)

# Function to predict the digit from the drawn image
def predict_digit():
    global draw_img, img
    
    # Save the canvas drawing as a PNG file
    img.save("drawing.png")  # Save as PNG image
    img = Image.open("drawing.png")  # Open the saved PNG file

    img_array = preprocess_image(img)  # Preprocess the drawn image

    prediction = model.predict(img_array)  # Get the model prediction
    predicted_digit = np.argmax(prediction)  # Get the predicted digit

    result_label.config(text=f"Predicted Digit: {predicted_digit}")  # Display result

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Predicted Digit: ")
    # Re-initialize the image and drawing context
    global img, draw_img
    img = Image.new('RGB', (280, 280), color='white')
    draw_img = ImageDraw.Draw(img)

# Initialize the Tkinter window
root = tk.Tk()
root.title("Draw and Recognize Digit")

# Create a canvas for drawing
canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.pack()

# Initialize the PIL drawing context
img = Image.new('RGB', (280, 280), color='white')
draw_img = ImageDraw.Draw(img)

# Bind the draw function to mouse movement
canvas.bind("<B1-Motion>", draw)

# Button to predict the digit
predict_button = tk.Button(root, text="Predict Digit", command=predict_digit)
predict_button.pack()

# Button to clear the canvas
clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas)
clear_button.pack()

# Label to display the result
result_label = tk.Label(root, text="Predicted Digit: ", font=("Helvetica", 16))
result_label.pack()

# Train the model if not already trained (you can comment out this line once you have the model saved)
# train_model()

# Run the Tkinter event loop
root.mainloop()
