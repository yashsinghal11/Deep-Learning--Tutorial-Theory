
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import CNN

st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ CNN Image Classifier")
st.write("Upload an image and let our trained CNN predict its class.")

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load the trained model
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(
        torch.load("../data/cifar10_cnn.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# Prepare uploaded image for the CNN
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Predict Image"):
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_index = probabilities.argmax().item()

        st.success(f"Prediction: {classes[predicted_index]}")
        st.write(
            f"Model confidence: "
            f"{probabilities[predicted_index].item() * 100:.2f}%"
        )

        st.subheader("Class probabilities")
        for i, class_name in enumerate(classes):
            st.write(f"{class_name}: {probabilities[i].item() * 100:.2f}%")
