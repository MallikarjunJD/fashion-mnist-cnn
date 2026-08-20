"""
Gradio demo for the Fashion-MNIST CNN — primary deployment target.

Local run:      python app.py
HF Spaces:      set this file as the Space's entry point (Gradio SDK),
                 alongside model.py, fashion_cnn.pt, and requirements.txt.

FIX (see DEBUG_GUIDE.md): the previous version used `image.getextrema()[0]`
(the single darkest pixel in the whole image) to decide whether to invert —
which is almost never true for real photos, so inversion never triggered and
every real-world image was fed to the model with inverted polarity relative
to training data. This version detects background brightness from the image
corners, crops tightly to the garment's bounding box (matching Fashion-MNIST's
tight framing), pads to square, then resizes — all before normalizing.
"""

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from torchvision import transforms
from PIL import Image, ImageOps

from model import FashionCNN

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
MEAN, STD = (0.2860,), (0.3530,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FashionCNN()
model.load_state_dict(torch.load("fashion_cnn.pt", map_location=device))
model.to(device)
model.eval()

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


import cv2
import numpy as np
from PIL import Image, ImageOps


def preprocess_for_model(image: Image.Image) -> Image.Image:
    # ---------------------------------------------------------
    # 1. PIL → OpenCV
    # ---------------------------------------------------------
    image = image.convert("RGB")
    img = np.array(image)

    # RGB → BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ---------------------------------------------------------
    # 2. Work at reasonable resolution
    # ---------------------------------------------------------
    h, w = img.shape[:2]

    max_size = 512

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------
    # 3. Denoise slightly
    # ---------------------------------------------------------
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # ---------------------------------------------------------
    # 4. Convert to grayscale
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # 5. Estimate foreground using adaptive threshold
    # ---------------------------------------------------------
    # Otsu gives us an initial separation between
    # foreground and background.
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Try both polarities and choose the one with
    # a reasonable foreground size.
    foreground_ratio = np.mean(mask > 0)

    if foreground_ratio > 0.7:
        mask = cv2.bitwise_not(mask)

    # ---------------------------------------------------------
    # 6. Clean the mask
    # ---------------------------------------------------------
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ---------------------------------------------------------
    # 7. Find largest connected object
    # ---------------------------------------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Ignore tiny objects/noise
        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )

        largest = contours[0]

        area = cv2.contourArea(largest)
        image_area = img.shape[0] * img.shape[1]

        # Only trust segmentation if the object
        # occupies a reasonable amount of the image.
        if area > image_area * 0.02:
            x, y, bw, bh = cv2.boundingRect(largest)

            # Add margin around garment
            margin = int(0.10 * max(bw, bh))

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + bw + margin)
            y2 = min(img.shape[0], y + bh + margin)

            cropped = gray[y1:y2, x1:x2]
        else:
            cropped = gray
    else:
        cropped = gray

    # ---------------------------------------------------------
    # 8. Normalize contrast
    # ---------------------------------------------------------
    cropped = cv2.normalize(
        cropped,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # ---------------------------------------------------------
    # 9. Put object on square canvas
    # ---------------------------------------------------------
    h, w = cropped.shape

    size = max(h, w)

    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    canvas[
        y_offset:y_offset + h,
        x_offset:x_offset + w
    ] = cropped

    # ---------------------------------------------------------
    # 10. Resize to Fashion-MNIST resolution
    # ---------------------------------------------------------
    final = cv2.resize(
        canvas,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    # ---------------------------------------------------------
    # 11. Convert to PIL
    # ---------------------------------------------------------
    result = Image.fromarray(final)

    return result


def predict(image: Image.Image):
    """Takes a PIL image (any size/mode), returns (class -> probability dict, preprocessed thumbnail)."""
    if image is None:
        return None, None

    thumb = preprocess_for_model(image)
    tensor = normalize(thumb).unsqueeze(0).to(device)   # (1, 1, 28, 28)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    label_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    # Upscale the 28x28 preprocessed thumbnail so it's actually visible in the UI
    preview = thumb.resize((196, 196), Image.NEAREST)
    return label_dict, preview


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload or draw a clothing item"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediction"),
        gr.Image(label="What the model actually sees (28x28, preprocessed)"),
    ],
    title="Fashion-MNIST Classifier (CNN, PyTorch)",
    description=(
        "A CNN trained from scratch on Fashion-MNIST (~92-93% validation accuracy). "
        "Upload a photo of a clothing item. The second output shows the exact "
        "28x28 image fed to the model, for transparency. "
        "Model architecture, training code, and math write-up: see the linked repo."
    ),
    examples=None,   # optionally: add paths to a few sample images shipped alongside this app
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
