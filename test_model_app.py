import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from keras.models import load_model

# Load your trained model
loaded_model = load_model("my_model.h5")  # Load your trained model here

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black", highlightthickness=0)
        self.canvas.pack()

        self.image = Image.new("L", (280, 280), "black")  # Change size to 280x280
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)

        self.recognize_button = tk.Button(self.root, text="Recognize", command=self.recognize_number)
        self.recognize_button.pack()

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.prediction_label = tk.Label(self.root, text="")
        self.prediction_label.pack()

    def preprocess_input_number(self, input_number):
        input_number = input_number.resize((28, 28))
        input_number = ImageOps.invert(input_number)  # Invert colors (black background, white digit)
        input_number = input_number.convert('L')
        preprocessed_number = np.array(input_number) / 255.0
        return preprocessed_number

    def predict_number(self, input_number):
        input_number = input_number.reshape((1, 28, 28, 1))
        predicted_class = loaded_model.predict(input_number).argmax()
        return predicted_class

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")  # Reset image
        self.draw = ImageDraw.Draw(self.image)

    def recognize_number(self):
        input_image = self.image.copy()
        preprocessed_number = self.preprocess_input_number(input_image)
        predicted_class = self.predict_number(preprocessed_number)
        self.prediction_label.config(text=f"Predicted number: {predicted_class}")

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        r = 10  # Increase the size of the drawn point
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")  # Use create_oval for smoother drawing
        self.draw.rectangle([x - r, y - r, x + r, y + r], fill="white")

def main():
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
