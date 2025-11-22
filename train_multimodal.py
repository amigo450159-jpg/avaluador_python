import argparse
import json
import os
import pandas as pd
import joblib
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
            embs.append(np.zeros(model.fc.in_features if hasattr(model.fc, "in_features") else 2048))
    return np.array(embs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label", default="price")
    parser.add_argument("--output_dir", default=os.path.join("artifacts", "sklearn_multimodal"))
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--price_min", type=float, default=None)
    parser.add_argument("--price_max", type=float, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.price_min is not None:
        df = df[df[args.label] >= args.price_min]
    if args.price_max is not None:
        df = df[df[args.label] <= args.price_max]

    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)

    features = [c for c in df.columns if c not in [args.label, "image_path"]]
    cat_cols = [c for c in features if df[c].dtype == object]
    num_cols = [c for c in features if c not in cat_cols]

    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    model, preprocess = load_image_embedder()
    train_embs = embed_images(train_df["image_path"].tolist(), model, preprocess)
    val_embs = embed_images(val_df["image_path"].tolist(), model, preprocess)

    X_train_tab = ct.fit_transform(train_df[features])
    X_val_tab = ct.transform(val_df[features])
    X_train = np.hstack([X_train_tab.toarray() if hasattr(X_train_tab, "toarray") else X_train_tab, train_embs])
    X_val = np.hstack([X_val_tab.toarray() if hasattr(X_val_tab, "toarray") else X_val_tab, val_embs])
    y_train = train_df[args.label].values
    y_val = val_df[args.label].values

    rf = RandomForestRegressor(n_estimators=300, random_state=args.seed)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    mae = float(np.mean(np.abs(preds - y_val)))
    rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
    r2 = float(1 - np.sum((preds - y_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2}, f, ensure_ascii=False, indent=2)
    joblib.dump({"transformer": ct, "model": rf, "features": features, "label": args.label}, os.path.join(args.output_dir, "model.joblib"))

if __name__ == "__main__":
    main()

