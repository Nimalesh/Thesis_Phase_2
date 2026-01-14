import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import os
from models.multitask import MultiTaskLatentAugModel
from config import IMAGE_SIZE

def run_prediction():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to test image (e.g. test1.png)")
    parser.add_argument("--model", type=str, default="checkpoints/model/model_fold_1_best.pth")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    model = MultiTaskLatentAugModel(num_classes=3).to(device)
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
    else:
        print(f"Error: Model weights not found at {args.model}")
        return

    
    raw_img = Image.open(args.image).convert('RGB')
    orig_w, orig_h = raw_img.size
    img = TF.resize(raw_img, [IMAGE_SIZE, IMAGE_SIZE])
    img_t = TF.normalize(TF.to_tensor(img), [0.485,0.456,0.406], [0.229,0.224,0.225]).unsqueeze(0).to(device)

    
    with torch.no_grad():
        seg_logits, cls_logits = model(img_t)
        
    
    probs = F.softmax(cls_logits, dim=1).cpu().numpy()[0]
    mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    classes = ['Normal', 'Benign', 'Malignant']
    pred_idx = np.argmax(probs)
    
    print(f"\n--- Results for {args.image} ---")
    print(f"Predicted Class: {classes[pred_idx]}")
    print(f"Normal: {probs[0]:.4f}, Benign: {probs[1]:.4f}, Malignant: {probs[2]:.4f}")

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(mask, cmap='jet', alpha=0.4)
    plt.title(f"Pred: {classes[pred_idx]}")
    plt.axis('off')
    
    save_path = "prediction_output.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_prediction()