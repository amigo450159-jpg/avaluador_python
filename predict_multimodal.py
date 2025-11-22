import argparse
import os
import pandas as pd
import joblib
import torch
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

def load_image_embedder():
    w = ResNet50_Weights.DEFAULT
    model = resnet50(weights=w)
    model.fc = torch.nn.Identity()
    model.eval()
    preprocess = w.transforms()
    return model, preprocess

def embed_images(paths, model, preprocess):
    embs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img = preprocess(img)
            with torch.no_grad():
                out = model(img.unsqueeze(0))
            embs.append(out.squeeze(0).numpy())
        except Exception:
            embs.append(np.zeros(2048))
    return np.array(embs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_csv", default="predicciones.csv")
    parser.add_argument("--label", default="price")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    bundle = joblib.load(os.path.join(args.model_dir, "model.joblib"))
    ct = bundle["transformer"]
    rf = bundle["model"]
    features = bundle["features"]
    model, preprocess = load_image_embedder()
    embs = embed_images(df["image_path"].tolist(), model, preprocess)
    X_tab = ct.transform(df[features])
    X = np.hstack([X_tab.toarray() if hasattr(X_tab, "toarray") else X_tab, embs])
    preds = rf.predict(X)
    out = df.copy()
    out[args.label + "_pred"] = preds
    out.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()

