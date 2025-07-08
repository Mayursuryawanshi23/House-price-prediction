
import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib  # to load saved model

# Load your trained model
model = joblib.load(r"C:\Users\acer\OneDrive\Desktop\ML-PROJECTS\House-price-prediction\housepriceprediction.sav")  

# GUI window
root = tk.Tk()
root.title("🏠 House Price Prediction")
root.geometry("600x600")
root.configure(bg="#f5f5f5")

# Labels
tk.Label(root, text="Enter 13 Values:", font=("Arial", 16), bg="#f5f5f5").pack(pady=10)

# Input box
input_entry = tk.Text(root, height=5, width=60, font=("Arial", 12))
input_entry.pack(pady=10)

# Output label
output_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f5f5f5", fg="#007acc")
output_label.pack(pady=20)

# Prediction function
def predict_price():
    input_text = input_entry.get("1.0", tk.END).strip()
    try:
        values = tuple(map(float, input_text.split(",")))
        if len(values) != 13:
            raise ValueError("Must enter exactly 13 values.")

        input_array = np.asarray(values).reshape(1, -1)
        prediction = model.predict(input_array)
        price = round(prediction[0], 2)

        output_label.config(text=f"Predicted House Price: $ {price}")
    except Exception as e:
        messagebox.showerror("Invalid Input", f"Error: {str(e)}")

# Predict Button
tk.Button(root, text="Predict Price", font=("Arial", 14), command=predict_price, bg="#007acc", fg="white").pack(pady=10)

root.mainloop()
