# Player Re-Identification using YOLOv11 and Deep-Person-ReID

## 🔥 Best Way to Run — Use Google Colab

The easiest and most stable way to run this project is via Google Colab.  
No local setup, no dependency issues — just open and run.

➡️ **[Click here to open the Colab notebook](https://colab.research.google.com/drive/1CIIN2kjhYFTqe3_Js3kgoBfaAfR5gMGw?usp=sharing)**  

---

## ⚠️ Running Locally (Advanced Users)

> **Note:** Running locally can be tricky due to CUDA and PyTorch compatibility issues. Proceed only if you're comfortable debugging environment setups.

If you still prefer or need to run locally, follow these steps carefully:

---

### ✅ Step-by-Step Installation (Local)


1. **Follow their official installation instructions:**  
   📌 Instructions: [https://github.com/KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid?tab=readme-ov-file#installation)

   Make sure to correctly install all dependencies and set up the environment as specified there.

2. **Install YOLOv11 (via ultralytics):**

   ```bash
   pip install ultralytics
   ```

3. **Download `reid.py` into the cloned folder:**

   Place your `reid.py` script inside the `deep-person-reid` folder.

4. **Some requirements (already installed due to above if not then):**
    ```bash
    pip install opencv-python gdown
    ```

5. **Run the script:**

   ```bash
   python reid.py
   ```

---

## 📁 File Overview

- `reid.py` – Main script to run the Re-ID pipeline using YOLO and Deep-Person-ReID.

- `approach.pdf` - Explains the method and maths

---

## 🛠 Notes

- This project combines **object detection (YOLOv11)** with **person re-identification** using Deep-Person-ReID.
- Ensure your GPU drivers and CUDA versions are compatible with your PyTorch installation if running locally.

---

## 🤝 Credits

- [Deep-Person-ReID](https://github.com/KaiyangZhou/deep-person-reid) by Kaiyang Zhou
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
