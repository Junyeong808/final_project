import cv2
import tkinter as tk
from tkinter import messagebox
import os

save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Capture Image")

label = tk.Label(root)
label.pack()

counter=0

def capture_image():
    global counter
    ret, frame = cap.read()
    if ret :
        filename = f"captured_image_{counter}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        counter += 1
        messagebox.showinfo(f"Success", "이미지가 저장되었습니다{captured_image.jpg}")
    else:
        messagebox.showerror("Error", "이미지가 저장되지 않았습니다. 코드를 확인해주세요.")

capture_button = tk.Button(root, text = "Capture", command = capture_image)
capture_button.pack(pady=20)

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.imencode('.png', frame_rgb)[1].tobytes()
        img = tk.PhotoImage(data=img)
        label.config(image=img)
        label.img = img
    label.after(10, update_frame)

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()


