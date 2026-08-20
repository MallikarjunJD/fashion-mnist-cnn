"""
Standalone preprocessing debugger — verify the fix BEFORE redeploying to Spaces.

Usage:
    python debug_preprocessing.py path/to/your/photo.jpg

Saves `preprocessed_debug.png`: the exact 28x28 image the model will receive,
upscaled 8x so you can actually inspect it. Also runs the model on it and
prints the prediction, so you can confirm the fix end-to-end without touching
the Gradio UI at all.
"""

import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import FashionCNN
from app import preprocess_for_model, CLASS_NAMES, MEAN, STD


def main(image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionCNN()
    model.load_state_dict(torch.load("fashion_cnn.pt", map_location=device))
    model.to(device)
    model.eval()

    original = Image.open(image_path)
    thumb = preprocess_for_model(original)

    # Save an upscaled version so it's actually inspectable by eye
    debug_out = thumb.resize((28 * 8, 28 * 8), Image.NEAREST)
    debug_out.save("preprocessed_debug.png")
    print(f"Saved preprocessed_debug.png — open it and check:")
    print("  - background should be BLACK (dark)")
    print("  - garment should be the BRIGHT region, filling most of the frame")
    print("  - it should look roughly like a Fashion-MNIST sample, not your original photo\n")

    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tensor = normalize(thumb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    ranked = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: -x[1])
    print("Top predictions:")
    for name, p in ranked[:5]:
        print(f"  {name:<14} {p*100:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_preprocessing.py path/to/photo.jpg")
        sys.exit(1)
    main(sys.argv[1])
