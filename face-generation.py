import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import streamlit as st
from PIL import Image

st.markdown(
    """
    <style>
    /* Custom styles */
    .main {
        background-color: #0e1117;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    .css-1d391kg {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stSlider .st-bc {
        color: #4CAF50;
    }
    img {
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_stable_diffusion_models():
    """Loads Stable Diffusion models with precision."""
    try:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        txt2img_model = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2", torch_dtype=torch_dtype
        )
        img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2", torch_dtype=torch_dtype
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        txt2img_model.to(device)
        img2img_model.to(device)
        return txt2img_model, img2img_model, device
    except torch.cuda.OutOfMemoryError:
        st.warning("CUDA out of memory. Switching to CPU.")
        torch.cuda.empty_cache()
        txt2img_model = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2", torch_dtype=torch.float32
        )
        img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2", torch_dtype=torch.float32
        )
        return txt2img_model.to("cpu"), img2img_model.to("cpu"), "cpu"

def generate_image_with_sd(prompt, model, device):
    """Generates an image using Stable Diffusion."""
    try:
        model.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else torch.no_grad():
            return model(prompt).images[0]
    except torch.cuda.OutOfMemoryError:
        st.error("CUDA ran out of memory.")
        torch.cuda.empty_cache()
        return None

def edit_image_with_sd(image, prompt, img2img_model, strength=0.75, device="cpu"):
    """Edits an existing image using Stable Diffusion Image-to-Image."""
    try:
        img2img_model.to(device)
        init_image = image.convert("RGB")
        with torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else torch.no_grad():
            edited_image = img2img_model(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=7.5
            ).images[0]
        return edited_image
    except torch.cuda.OutOfMemoryError:
        st.error("CUDA ran out of memory while editing the image.")
        torch.cuda.empty_cache()
        return None

def main():
    st.title("🎨 Face Generation & Editing with Stable Diffusion 👦")

    with st.sidebar:
        st.header("🔧 Image Settings")
        base_prompt = st.text_input("Enter your prompt:", "A photorealistic portrait of a young person")
        st.subheader("👦 Edit Parameters")
        age = st.slider("Age (years)", 10, 80, 25)
        smile_length = st.slider("Smile Intensity", 0, 10, 5)
        eye_color = st.selectbox("Eye Color", ["blue", "green", "brown", "hazel", "grey", "amber"])
        hair_color = st.selectbox("Hair Color", ["black", "brown", "blonde", "red", "grey", "white"])

        edit_feature = st.selectbox("Select Feature to Edit", ["Age", "Smile", "Eye Color", "Hair Color"])

    # Load models
    txt2img_model, img2img_model, device = load_stable_diffusion_models()

    if 'base_image' not in st.session_state:
        st.session_state.base_image = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None

    # Generate Image Button
    st.write("## Generate a Base Image")
    if st.button("🚀 Generate"):
        with st.spinner("Generating your image... 🎨"):
            st.session_state.base_image = generate_image_with_sd(base_prompt, txt2img_model, device)
            st.session_state.original_image = st.session_state.base_image  # Store original image
            if st.session_state.base_image:
                st.image(st.session_state.base_image, caption="🖼 Base Image", use_container_width=True)
            else:
                st.error("Failed to generate the image.")

    # Edit Image Section
    if st.session_state.base_image is not None:
        st.write("## Edit the Generated Image")

        edit_prompt = base_prompt  # Start with the base prompt

        if edit_feature == "Age":
            edit_prompt = f"{base_prompt}, age {age}"
        elif edit_feature == "Smile":
            edit_prompt = f"{base_prompt}, {'with a big smile' if smile_length > 6 else 'with a gentle smile' if smile_length > 3 else 'with a serious expression'}"
        elif edit_feature == "Eye Color":
            edit_prompt = f"{base_prompt}, {eye_color} eyes"
        elif edit_feature == "Hair Color":
            edit_prompt = f"{base_prompt}, {hair_color} hair"

        if st.button("✨ Edit Image"):
            with st.spinner("Editing image... 🔧"):
                edited_image = edit_image_with_sd(
                    st.session_state.base_image,  # Pass the base image
                    edit_prompt,  # Use the generated prompt based on selected feature
                    img2img_model,  # Image-to-image pipeline
                    strength=0.75,  # Default strength for image editing
                    device=device
                )
                if edited_image:
                    st.image(edited_image, caption="👀 Edited Image", use_container_width=True)
                    st.session_state.base_image = edited_image  # Update base image with the edited one
                else:
                    st.error("Failed to edit the image.")

        if st.button("🔙 Revert to Original Image"):
            if st.session_state.original_image:
                st.session_state.base_image = st.session_state.original_image
                st.image(st.session_state.base_image, caption="🖼 Original Image", use_container_width=True)

    st.info("💡 Tip: Use the sidebar to refine parameters for a better result.")

if __name__ == "__main__":
    main()
