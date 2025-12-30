import os
import sys
import shutil
import tkinter as tk
from tkinter import simpledialog, messagebox
import requests
import time
import numpy as np
from keras.preprocessing import image
from PIL import Image, ImageTk
from keras.models import load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------ Configuration ------------------
MODEL_PATH = 'E:\\code\\py\\mL\\cat and dog\\cat_dog_model.keras'
IMG_SIZE = (128, 128)
SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
TEMP_DIR = "testing"

plt.rcParams['toolbar'] = 'None'

# ------------------ Download Functions ------------------
def download_images_from_api(animal, count, save_dir):
    """Download images of 'cat' or 'dog' using public APIs"""
    if animal not in ['cat', 'dog']:
        return []
    
    base_url = "https://api.thecatapi.com/v1/images/search" if animal == 'cat' else "https://api.thedogapi.com/v1/images/search"
    downloaded = 0
    page = 0
    paths = []

    while downloaded < count:
        try:
            limit = min(10, count - downloaded)
            params = {'limit': limit, 'mime_types': 'jpg,png', 'size': 'med'}
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for item in data:
                if downloaded >= count:
                    break
                img_url = item['url']
                ext = os.path.splitext(img_url)[1].split('?')[0] or '.jpg'
                filename = f"{animal}_{downloaded:04d}{ext}"
                filepath = os.path.join(save_dir, filename)

                try:
                    img_data = requests.get(img_url, timeout=10)
                    img_data.raise_for_status()
                    with open(filepath, 'wb') as f:
                        f.write(img_data.content)
                    paths.append(filepath)
                    downloaded += 1
                except Exception as e:
                    print(f" Failed to download {img_url}: {e}")
                time.sleep(0.3)

            page += 1
        except Exception as e:
            print(f" API request failed for {animal}: {e}")
            break

    return paths

def prompt_and_download():
    """Ask user how many cats/dogs to download, then download them"""
    root = tk.Tk()
    root.withdraw() 

    try:
        cat_num = simpledialog.askinteger("Input", "How many CAT images to download?", minvalue=0, maxvalue=100)
        if cat_num is None:
            return None
        dog_num = simpledialog.askinteger("Input", "How many DOG images to download?", minvalue=0, maxvalue=100)
        if dog_num is None:
            return None
    except Exception:
        return None

    if cat_num == 0 and dog_num == 0:
        messagebox.showinfo("Info", "No images selected. Exiting.")
        return None

    # Create temp folder
    os.makedirs(TEMP_DIR, exist_ok=True)

    all_paths = []
    if cat_num > 0:
        print(f"📥 Downloading {cat_num} cat images...")
        all_paths.extend(download_images_from_api('cat', cat_num, TEMP_DIR))
    if dog_num > 0:
        print(f"📥 Downloading {dog_num} dog images...")
        all_paths.extend(download_images_from_api('dog', dog_num, TEMP_DIR))

    root.destroy()
    return sorted(all_paths)

# ------------------ Load Model ------------------
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    messagebox.showerror("Error", f"Failed to load model:\n{e}")
    sys.exit(1)

# ------------------ Prediction Logic ------------------
def predict_image_from_path(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    pred_val = model.predict(img_array, verbose=0)[0][0]
    return {
        'prob_cat': float(1.0 - pred_val),
        'prob_dog': float(pred_val),
        'class': 'DOG' if pred_val > 0.5 else 'CAT'
    }

# ------------------ GUI Class ------------------
class CatDogClassifierApp:
    def __init__(self, root, image_paths):
        self.root = root
        self.image_paths = image_paths
        self.current_index = 0
        self.root.title("AI Image Classifier (Auto-downloaded Test Set)")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) 

        self.setup_ui()
        if self.image_paths:
            self.update_content()

    def on_closing(self):
        """Clean up temp folder and exit"""
        try:
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
                print(f" Temporary folder '{TEMP_DIR}' deleted.")
        except Exception as e:
            print(f" Failed to delete temp folder: {e}")
        self.root.destroy()
        os._exit(0)

    def force_cursor(self, event=None):
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().config(cursor="arrow")

    def setup_ui(self):
        header = tk.Frame(self.root)
        header.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(header, text=f"Testing on {len(self.image_paths)} auto-downloaded images", fg="gray").pack()

        body = tk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=10)

        left = tk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.img_disp = tk.Label(left, bg="#f0f0f0", relief="solid")
        self.img_disp.pack(pady=10, expand=True)
        self.info_lbl = tk.Label(left, text="0 / 0", font=("Arial", 10))
        self.info_lbl.pack()
        self.res_lbl = tk.Label(left, text="Result: ---", font=("Arial", 14, "bold"))
        self.res_lbl.pack(pady=10)

        right = tk.Frame(body)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        cw = self.canvas.get_tk_widget()
        cw.pack(fill=tk.BOTH, expand=True)
        cw.bind("<Motion>", self.force_cursor)

        footer = tk.Frame(self.root)
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        self.btn_prev = tk.Button(footer, text="◀ Previous", width=15, command=self.prev_img,
                                  state=tk.NORMAL if len(self.image_paths) > 1 else tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=100)
        self.btn_next = tk.Button(footer, text="Next ▶", width=15, command=self.next_img,
                                  state=tk.NORMAL if len(self.image_paths) > 1 else tk.DISABLED)
        self.btn_next.pack(side=tk.RIGHT, padx=100)

    def update_content(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        pil_img = Image.open(path)
        pil_img.thumbnail((300, 300))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.img_disp.config(image=tk_img)
        self.img_disp.image = tk_img
        self.info_lbl.config(text=f"{self.current_index + 1} / {len(self.image_paths)}")

        res = predict_image_from_path(path)
        color = "#E67E22" if res['class'] == 'DOG' else "#3498DB"
        self.res_lbl.config(text=f"Prediction: {res['class']}", fg=color)

        self.ax.clear()
        bars = self.ax.bar(['Cat', 'Dog'], [res['prob_cat'], res['prob_dog']], color=['#3498DB', '#E67E22'])
        self.ax.set_ylim(0, 1.1)
        self.ax.set_title("Confidence Level")
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.1%}', ha='center', va='bottom')
        self.fig.tight_layout()
        self.canvas.draw()
        self.force_cursor()

    def next_img(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_content()

    def prev_img(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_content()

# ------------------ Main Execution ------------------
if __name__ == '__main__':
    image_list = prompt_and_download()
    if not image_list:
        print("No images to process. Exiting.")
        sys.exit(0)

    root = tk.Tk()
    app = CatDogClassifierApp(root, image_list)
    root.mainloop()