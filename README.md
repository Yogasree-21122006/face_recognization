# 🎯 Face Recognition System (Streamlit App)

This project is a **Face Recognition Web Application** developed using **Streamlit** and **PyTorch**. It allows users to upload a cropped face image and verifies the identity using a deep learning-based face embedding model.

---

## 🚀 Live Demo

👉 https://facerecognization-eg4w6fnbpw9q4taz8ytvjk.streamlit.app/

---

## 🧠 Project Overview

This system uses a **deep learning Face Embedding Model** to extract facial features and compare them with a stored database of known faces.

Instead of directly comparing images, the model converts faces into **numerical vectors (embeddings)** and uses similarity metrics to identify matches.

---

## 🔄 Workflow

1. User uploads a cropped face image
2. Image is preprocessed (resize, normalization)
3. Deep learning model generates face embedding
4. Embedding is compared with stored database
5. System finds the best match
6. Based on threshold:

   * ✅ VERIFIED (if match is strong)
   * ❌ UNKNOWN (if match is weak)

---

## 🛠️ Tech Stack

* **Python** 🐍
* **Streamlit** 🌐 (Web Interface)
* **PyTorch** 🔥 (Deep Learning)
* **Torchvision** (Image transformations)
* **PIL** (Image processing)

---

## 📁 Project Structure

```
Face_recognization/
│
├── app.py                         # Main Streamlit application (UI + logic)
├── model_helper.py                # FaceEmbeddingNet model architecture
├── face_embedding_model_cpu.pth   # Trained model weights
├── embeddings.pkl                 # Stored face embeddings database
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
```

---

## 🧪 How It Works

* The model extracts **128-dimensional face embeddings**
* Uses **cosine similarity** to compare faces
* A predefined **threshold (0.5)** determines verification
* Highest similarity score is selected as best match

---

## ▶️ How to Run Locally

1. Clone the repository:

```
git clone https://github.com/Yogasree-21122006/face_recognization.git
cd face_recognization
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
streamlit run app.py
```

---

## ⚠️ Limitations

* Requires **cropped face images** for accurate results
* Accuracy depends on training data quality
* Threshold tuning may be required for better performance

---

## 🔮 Future Enhancements

* 🎥 Real-time face recognition using webcam
* 🧠 Improve model accuracy with larger dataset
* ☁️ Cloud database for scalable storage
* 🔐 Add authentication & user management

---

## 👨‍💻Developer

**Yogasree S**

---

## ⭐ Conclusion

This project demonstrates how **deep learning and web technologies** can be combined to build a real-time face verification system. It is suitable for applications like attendance systems, security verification, and identity authentication.

---
