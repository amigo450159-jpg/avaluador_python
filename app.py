import io
import os
import tempfile
import pandas as pd
import streamlit as st
import joblib
import torch
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import re

st.set_page_config(page_title="Avaluador Multimodal", layout="centered")
st.title("Avaluador Multimodal")

model_dir = st.text_input("Directorio del modelo", value=os.path.join("artifacts", "sklearn_multimodal"))
TOPE_MAXIMO = 0
MINIMO_PERMITIDO = 0
BOUNDS_PATH = os.path.join(os.getcwd(), "PRECIOS PCS.csv")
df_bounds = None
BRAND_SYNONYMS = {
    "MICROSFT": "MICROSOFT",
    "HUAWI": "HUAWEI",
    "XIAMI": "XIAOMI",
    "GYGABYTE": "GIGABYTE",
    "VICTUS-HP": "HP",
}

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()

def parse_money(x):
    if pd.isna(x):
        return 0
    s = str(x)
    s = s.replace("$", "").replace(",", "").strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return 0
    return int("".join(nums))

def load_bounds():
    global df_bounds
    if os.path.isfile(BOUNDS_PATH):
        df = pd.read_csv(BOUNDS_PATH)
        for c in df.columns:
            df[c] = df[c].apply(clean_text)
        df["MINIMO_VAL"] = df["MINIMO"].apply(parse_money)
        df["MAXIMO_VAL"] = df["MAXIMO"].apply(parse_money)
        df_bounds = df

def normalize_disk(disk_type, capacity_gb):
    t = disk_type.upper()
    if t == "NVME":
        t = "SSD"
    if capacity_gb >= 1000:
        cap = f"{int(round(capacity_gb/1000))}TB"
    else:
        cap = str(int(capacity_gb))
    return f"{t} {cap}".upper()

def cpu_tokens(proc):
    p = proc.upper()
    tokens = []
    if "I3" in p or "CORE I3" in p:
        tokens.append("CORE I3")
    if "I5" in p or "CORE I5" in p:
        tokens.append("CORE I5")
    if "I7" in p or "CORE I7" in p:
        tokens.append("CORE I7")
    if "PENTIUM" in p:
        tokens.append("PENTIUM")
    if "CELERON" in p:
        tokens.append("CELERON")
    if "RYZEN 3" in p:
        tokens.append("RYZEN 3")
    if "RYZEN 5" in p:
        tokens.append("RYZEN 5")
    if "RYZEN 7" in p:
        tokens.append("RYZEN 7")

    nums = re.findall(r"\d+", p)
    if nums:
        n = nums[0]
        if any(x in p for x in ["I3", "CORE I3", "I5", "CORE I5", "I7", "CORE I7"]):
            try:
                gen = int(n)
                if gen >= 10:
                    tokens.append(f"{gen}TH")
            except Exception:
                pass
        else:
            tokens.append(n)

    if not tokens:
        tokens = [p]
    return tokens

def get_bounds(brand, disk_type, disk_gb, ram_gb, processor, gpu_gamer):
    if df_bounds is None:
        return 0, 0
    df = df_bounds
    brand_u = clean_text(brand)
    brand_u = BRAND_SYNONYMS.get(brand_u, brand_u)
    disk_u = normalize_disk(disk_type, int(disk_gb))
    ram_u = f"RAM {int(ram_gb)}".upper()
    gpu_u = "(EN BLANCO)" if clean_text(gpu_gamer) == "NO" else "NVIDIA"
    toks = cpu_tokens(processor)
    cand = df.copy()
    if brand_u:
        bfiltered = df[df["MARCA"] == brand_u]
        cand = bfiltered if len(bfiltered) > 0 else df.copy()
    def proc_ok(s):
        su = clean_text(s)
        return all(t in su for t in toks)
    cand = cand[cand["PROCESADOR"].apply(proc_ok)]
    gpuu = clean_text(gpu_gamer)
    if gpuu == "NO":
        cand = cand[cand["GRAFICA GAMER"].isin(["(EN BLANCO)", "", " "])]
    elif "GTX 1650" in gpuu:
        cand = cand[cand["GRAFICA GAMER"].str.contains("GTX 1650", na=False)]
    else:
        cand = cand[cand["GRAFICA GAMER"].str.contains("NVIDIA", na=False)]
    strict = cand[(cand["DISCO"] == disk_u) & (cand["RAM"] == ram_u)]
    if len(strict) == 0:
        strict = cand[(cand["DISCO"] == disk_u)]
    if len(strict) == 0:
        strict = cand[(cand["RAM"] == ram_u)]
    if len(strict) == 0:
        strict = cand
    if len(strict) == 0:
        pool = df[df["PROCESADOR"].apply(proc_ok)]
        if len(pool) == 0:
            pool = df
        if len(pool) == 0:
            return 0, 0
        mn = int(np.nanmedian(pd.to_numeric(pool["MINIMO_VAL"], errors="coerce"))) if "MINIMO_VAL" in pool.columns else 0
        mx = int(np.nanmedian(pd.to_numeric(pool["MAXIMO_VAL"], errors="coerce"))) if "MAXIMO_VAL" in pool.columns else 0
        return mn, mx
    row = strict.iloc[0]
    mn = int(row["MINIMO_VAL"]) if not pd.isna(row["MINIMO_VAL"]) else 0
    mx = int(row["MAXIMO_VAL"]) if not pd.isna(row["MAXIMO_VAL"]) else 0
    return mn, mx

load_bounds()

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
st.subheader("Sube las fotos y especificaciones del equipo para recibir una valoración estimada de compra.")
steps_cols = st.columns(3)
with steps_cols[0]:
    st.success("1. Subir fotos")
with steps_cols[1]:
    st.success("2. Ingresar datos")
with steps_cols[2]:
    st.success("3. Recibir valoración")
uploaded_images = st.file_uploader(
    "Sube las fotos del equipo (mínimo 3)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Pantalla encendida\nTeclado visible\nCarcasa y bisagras\nFotos claras y sin flash"
)
if uploaded_images:
    cols = st.columns(min(3, len(uploaded_images)))
    for i, f in enumerate(uploaded_images[:3]):
        with cols[i % len(cols)]:
            st.image(f)

st.subheader("Especificaciones Técnicas")

brands = ["HP", "VICTUS-HP", "LENOVO", "DELL", "ASUS", "ACER", "MSI", "RAZER", "APPLE", "MICROSFT", "SAMSUNG", "TOSHIBA", "HUAWI", "XIAMI", "LG", "GYGABYTE"]
brand = st.selectbox("Marca", options=brands, index=0)

ram_options = ["4gb", "8gb", "12gb", "16gb", "32gb", "128 gb"]
ram_label = st.selectbox("Memoria RAM (GB)", options=ram_options, index=1)
ram_gb = int(re.findall(r"\d+", ram_label)[0])

disk_types = ["HDD", "SSD", "NVMe"]
disk_type = st.selectbox("Tipo de Disco", options=disk_types, index=1)

disk_capacities = [
    "120 GB", "160 GB", "240 GB", "250 GB", "256 GB", "320 GB", "480 GB", "500 GB",
    "1 TB", "2 TB", "3 TB", "4 TB", "6 TB", "8 TB", "10 TB"
]
disk_label = st.selectbox("Capacidad Disco", options=disk_capacities, index=5)
match = re.findall(r"(\d+)\s*(GB|TB)", disk_label, flags=re.IGNORECASE)
if match:
    num, unit = match[0]
    disk_gb = int(num) * (1000 if unit.upper() == "TB" else 1)
else:
    disk_gb = 500

processors = [
    "Ryzen 5 1600", "Ryzen 5 2600", "Ryzen 5 3600", "Ryzen 5 4500U", "Ryzen 5 5600X", "Ryzen 5 7600X",
    "Ryzen 7 8700G",
    "i3 1ª generación", "i3 2ª generación", "i3 3ª generación", "i3 4ª generación", "i3 5ª generación",
    "i3 6ª generación", "i3 7ª generación", "i3 8ª generación", "i3 9ª generación", "i3 10ª generación",
    "i3 11ª generación", "i3 12ª generación", "i3 13ª generación", "i3 14ª generación", "i3 15ª generación",
    "i5 1ª generación", "i5 2ª generación", "i5 3ª generación", "i5 4ª generación", "i5 5ª generación",
    "i5 6ª generación", "i5 7ª generación", "i5 8ª generación", "i5 9ª generación", "i5 10ª generación",
    "i5 11ª generación", "i5 12ª generación", "i5 13ª generación", "i5 14ª generación", "i5 15ª generación",
    "PENTIUM", "CELERON", "INTEL INSIDE"
]
processor = st.selectbox("Procesador", options=processors, index=0)

gpu_gamer = st.selectbox("¿Tiene gráfica gamer (RTX/GTX/RX)?", options=["NO", "NVIDIA GTX 1650", "NVIDIA (OTRA)"], index=0)

notas = st.text_area("Notas adicionales", placeholder="Problemas del equipo, reparaciones recientes, caja/cargador/accesorios")

# Estado del equipo con penalización configurable
estado_options = ["EXCELENTE", "BUENO", "CON DESGASTES", "DEFECTO MODERADO", "DEFECTO SEVERO"]
estado = st.selectbox("Estado del equipo", options=estado_options, index=1)
estado_penalty_map = {
    "EXCELENTE": 0.00,
    "BUENO": 0.05,
    "CON DESGASTES": 0.10,
    "DEFECTO MODERADO": 0.25,
    "DEFECTO SEVERO": 0.40,
}

blocked = any(x in processor.lower() for x in ["pentium", "celeron", "atom"]) 
if blocked:
    st.warning("Las características de su equipo no cumplen con los requisitos mínimos que manejamos. Agradecemos que nos haya tenido en cuenta; si lo desea, podemos evaluar otro artículo")

if st.button("Solicitar valoración"):
    if not uploaded_images or len(uploaded_images) < 3:
        st.error("Sube al menos 3 fotos claras.")
    elif blocked:
        st.error("Procesador no permitido por política.")
    else:
        try:
            bundle = joblib.load(os.path.join(model_dir, "model.joblib"))
            ct = bundle["transformer"]
            rf = bundle["model"]
            features = bundle["features"]
            cat_cols = bundle["transformer"].transformers_[0][2]
            num_cols = bundle["transformer"].transformers_[1][2]
            label = bundle.get("label", "price")
            model_img, preprocess = load_image_embedder()
        except Exception:
            bundle = None
            ct = None
            rf = None
            features = []
            cat_cols = []
            num_cols = []
            label = "price"
            model_img = None
            preprocess = None
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i, f in enumerate(uploaded_images):
                p = os.path.join(tmpdir, f"img_{i}.png")
                with open(p, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(p)
            rows = []
            for p in paths:
                rows.append({
                    "brand": brand,
                    "ram_gb": ram_gb,
                    "disk_type": disk_type,
                    "disk_capacity_gb": disk_gb,
                    "processor": processor,
                    "gamer_gpu": gpu_gamer,
                    "image_path": p,
                    "notes": notas
                })
            df_one = pd.DataFrame(rows)
            bmin, bmax = get_bounds(brand, disk_type, disk_gb, ram_gb, processor, gpu_gamer)
            if bundle is not None:
                for c in cat_cols:
                    if c not in df_one.columns:
                        df_one[c] = ""
                for c in num_cols:
                    if c not in df_one.columns:
                        df_one[c] = 0
                embs = embed_images(df_one["image_path"].tolist(), model_img, preprocess)
                X_tab = ct.transform(df_one[features])
                X = np.hstack([X_tab.toarray() if hasattr(X_tab, "toarray") else X_tab, embs])
                preds = rf.predict(X)
                series = pd.Series([float(p) for p in preds])
                price_mean = float(series.mean())
                price_min = float(series.min())
                price_max = float(series.max())
            else:
                if bmin == 0 and bmax == 0:
                    st.warning("Las características de su equipo no cumplen con los requisitos mínimos que manejamos. Agradecemos que nos haya tenido en cuenta; si lo desea, podemos evaluar otro artículo")
                    st.stop()
                price_mean = (float(bmin) + float(bmax)) / 2.0 if (bmin and bmax) else float(max(bmin, bmax))
                price_min = float(bmin) if bmin else price_mean
                price_max = float(bmax) if bmax else price_mean

            # Penalización por estado y calidad de imagen
            def quality_penalty(paths):
                import numpy as np
                from PIL import Image
                def lap_var(img_arr):
                    k = np.array([[0,1,0],[1,-4,1],[0,1,0]])
                    return float(np.var(np.abs(
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[:-2,:-2]*k[0,0] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[:-2,1:-1]*k[0,1] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[:-2,2:]*k[0,2] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[1:-1,:-2]*k[1,0] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[1:-1,1:-1]*k[1,1] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[1:-1,2:]*k[1,2] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[2:,:-2]*k[2,0] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[2:,1:-1]*k[2,1] +
                        np.pad(img_arr, ((1,1),(1,1)), mode='edge')[2:,2:]*k[2,2]
                    )))
                penalties = []
                for p in paths:
                    try:
                        g = Image.open(p).convert('L')
                        arr = np.array(g)/255.0
                        blur = lap_var(arr)
                        bright = float(arr.mean())
                        pen = 0.0
                        if blur < 0.0005:
                            pen += 0.05
                        if bright < 0.2 or bright > 0.9:
                            pen += 0.03
                        penalties.append(min(pen, 0.1))
                    except Exception:
                        penalties.append(0.05)
                return float(np.mean(penalties)) if penalties else 0.0

            penalty_total = estado_penalty_map.get(estado, 0.0) + quality_penalty(paths)
            penalty_total = min(penalty_total, 0.5)

            price_adj = price_mean * (1 - penalty_total)
            price_capped = price_adj
            if bmax and bmax > 0:
                price_capped = min(price_capped, float(bmax))
            if bmin and bmin > 0:
                price_capped = max(price_capped, float(bmin))

            loan_value = price_capped * 0.30
            st.success(f"El monto del préstamo es de ${loan_value:,.0f}. ¿Deseas continuar con tu contrato express? Sí o No.")
            print({
                "avaluo": int(price_capped),
                "prestamo_30": int(loan_value),
                "min": int(price_min),
                "max": int(price_max),
                "n_imagenes": len(paths),
                "notas": notas
            })

st.caption("Compraventas Standard Whatsapp 322 2626710 Respuesta en menos de 2 horas")


