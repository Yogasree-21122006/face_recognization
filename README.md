# 🎯 Face Recognition System (Streamlit App)

This is a **Face Recognition Web App** built using **Streamlit** and **PyTorch**.  
It allows users to upload a cropped face image and verifies the identity using a deep learning model.

---

## 🚀 Live Demo
👉[(https://facerecognization-eg4w6fnbpw9q4taz8ytvjk.streamlit.app/)](https://facerecognization-eg4w6fnbpw9q4taz8ytvjk.streamlit.app/)

---

## 🧠 Project Overview

This project uses a **Face Embedding Model** to compare uploaded face images with a stored database of embeddings.

### Workflow:
1. Upload a face image
2. Image is preprocessed
3. Model generates embedding
4. Compared with stored embeddings
5. Shows:
   - Best Match
   - Similarity Score
   - VERIFIED / UNKNOWN

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🌐
- PyTorch 🔥
- Torchvision
- PIL (Image Processing)

---

## 📁 Project Structure
Face_recognization/
│
├── app.py                         # Main Streamlit application (frontend + logic)
├── model_helper.py                # Neural network architecture (FaceEmbeddingNet)
├── face_embedding_model_cpu.pth   # Trained deep learning model
├── embeddings.pkl                 # Stored face embeddings database
├── requirements.txt               # Required Python libraries
├── README.md                      # Project documentation
