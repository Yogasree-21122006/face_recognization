import streamlit as st
from PIL import Image
import torch
import pickle
from torchvision import transforms
import torch.nn.functional as F
from model_helper import FaceEmbeddingNet
import io

st.title("Face Recognition System")
st.write("Upload a CROPPED face image for verification.")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# Load model
@st.cache_resource
def load_model():
    model = FaceEmbeddingNet(embedding_dim=128)
    model.load_state_dict(
        torch.load("face_embedding_model_cpu.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Load embeddings
@st.cache_data
def load_embeddings():
    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)

embedding_db = load_embeddings()

# Image transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

def cosine_similarity(a, b):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    similarity = F.cosine_similarity(a, b, dim=1)
    return similarity.item()

# File uploader
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare tensor
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Get embedding
    with torch.no_grad():
        test_embedding = model(img_tensor).squeeze(0)

    # Compare with DB
    best_match = None
    best_score = -1
    for name, db_embedding in embedding_db.items():
        db_embedding = torch.tensor(db_embedding, dtype=torch.float32).to(DEVICE)
        score = cosine_similarity(test_embedding, db_embedding)
        if score > best_score:
            best_score = score
            best_match = name

    result = "VERIFIED" if best_score > THRESHOLD else "UNKNOWN"

    st.write(f"### Best Match: {best_match}")
    st.write(f"### Similarity Score: {best_score:.4f}")

    if result == "VERIFIED":
        st.success("VERIFIED")
    else:
        st.error("UNKNOWN PERSON")