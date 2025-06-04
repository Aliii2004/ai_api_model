from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse # FileResponse import qiling
from fastapi.staticfiles import StaticFiles # StaticFiles import qiling
from starlette.responses import HTMLResponse # HTMLResponse import qiling (index.html uchun)



from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np

# === ALINet Model Arxitekturasi Ta'riflari (Sizning notebookingizdan olindi) ===
# Bu klass ta'riflari sizning notebookingizdagi bilan bir xil bo'lishi kerak.
# Yaxshiroq tashkillashtirish uchun bularni alohida faylga joylashtirib, keyin import qilishingiz mumkin.

class ALIBlock(nn.Module):
    """
    ALINet asosiy bloki - Residual connection va multi-scale convolution
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ALIBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Multi-scale convolution
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.conv5x5 = nn.Conv2d(out_channels, out_channels, kernel_size=5,
                                padding=2, bias=False)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Channel attention
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        # Main path
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Multi-scale features
        out_1x1 = self.conv1x1(out)
        out_5x5 = self.conv5x5(out)
        out = out + out_1x1 + out_5x5

        # Channel attention
        out = self.channel_attention(out)

        # Shortcut connection
        out += self.shortcut(residual)
        out = F.relu(out)

        return out

class ChannelAttention(nn.Module):
    """
    Channel Attention Module - Muhim kanallarni ta'kidlash
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))

        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Combine and apply sigmoid
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1)

        return x * attention.expand_as(x)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module - Muhim hududlarni ta'kidlash
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(out))

        return x * attention

class FeatureFusion(nn.Module):
    """
    Feature Fusion Module - Turli miqyosdagi xususiyatlarni birlashtirish
    """
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Multi-scale convolutions
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)

        # Concatenate and fuse
        fused = torch.cat([out1, out3, out5], dim=1)
        output = F.relu(self.bn(self.fusion_conv(fused)))

        return output + x  # Residual connection

class ALINet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ALINet, self).__init__()

        # Kirish qatlami - 3 kanal (RGB)
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ALINet Block 1 - Low-level features
        self.block1 = ALIBlock(32, 32, stride=1)
        self.block2 = ALIBlock(32, 64, stride=2)

        # ALINet Block 2 - Mid-level features
        self.block3 = ALIBlock(64, 64, stride=1)
        self.block4 = ALIBlock(64, 128, stride=2)

        # ALINet Block 3 - High-level features
        self.block5 = ALIBlock(128, 128, stride=1)
        self.block6 = ALIBlock(128, 128, stride=1)
        self.block7 = ALIBlock(128, 128, stride=1)
        self.block8 = ALIBlock(128, 128, stride=1)

        # Attention Module
        self.attention = SpatialAttention(128)

        # Feature Fusion Module
        self.feature_fusion = FeatureFusion(128)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Initialize weights (API uchun keragi yo'q, yuklangan modelda bor)
        # self._initialize_weights()

    def forward(self, x):
        # Input layer
        x = self.input_conv(x)

        # Feature extraction blocks
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        x8 = self.block8(x7)

        # Attention mechanism
        x_att = self.attention(x8)

        # Feature fusion
        x_fused = self.feature_fusion(x_att)

        # Global pooling
        x_pooled = self.global_avg_pool(x_fused)
        x_flat = torch.flatten(x_pooled, 1)

        # Classification
        output = self.classifier(x_flat)

        return output

# === API Qismi ===

app = FastAPI(title="ALINet Malaria Detection API", version="1.0")

# === Statik fayllarni qo'shish ===
# "/" yo'lida index.html ni ko'rsatish
# Bu asosiy sahifa bo'ladi
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Agar loyiha ichida boshqa statik fayllar (rasmlar, boshqa css/js fayllar) bo'lsa
# Ularni ham shu orqali taqdim etishingiz mumkin.
# Hozirgi holatda faqat index.html uchun get("/") ishlatish kifoya
# app.mount("/static", StaticFiles(directory="."), name="static_files") # Agar kerak bo'lsa


# ... (Model yuklash qismi - avvalgidek qoladi) ...
MODEL_PATH = 'alinet_best_model.pth' # Model shu papkada joylashgan
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"API ishlatilayotgan qurilma: {DEVICE}")

model = ALINet(num_classes=2)

try:
    # weights_only=False ni o'zingizning xavfsizlik baholashingiz bo'yicha qo'shing
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False) # Yoki yuqoridagi 2-variant
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval() # Inference uchun eval mode
    print(f"Model muvaffaqiyatli yuklandi: {MODEL_PATH}")
except FileNotFoundError:
    raise FileNotFoundError(f"Xato: Model fayli topilmadi: {MODEL_PATH}. Iltimos, uni main.py bilan bir papkaga joylashtiring.")
except Exception as e:
    # collections.defaultdict xatosini boshqarish yoki weights_only=False ishlatish
    raise RuntimeError(f"Modelni yuklashda xato yuz berdi: {e}")


# ... (Transformatsiyalar va CLASS_NAMES - avvalgidek qoladi) ...
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = (128, 128)

inference_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

CLASS_NAMES = ['Parasitized', 'Uninfected']


# === API Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Yuklangan rasm bo'yicha bezgak tashxisini bashorat qilish.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Faqat rasm fayllarini yuklang")

    try:
        # Rasmni o'qish
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB") # RGB ga o'tkazish

        # Transformatsiyalarni qo'llash
        img_tensor = inference_transform(img)
        img_tensor = img_tensor.unsqueeze(0) # Batch o'lchamini qo'shish (1, C, H, W)
        img_tensor = img_tensor.to(DEVICE)

        # Inference qilish
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0] # Batch dan chiqarish

        # Natijalarni qayta ishlash
        predicted_prob, predicted_class_index = torch.max(probabilities, 0)

        result = {
            "filename": file.filename,
            "prediction": CLASS_NAMES[predicted_class_index.item()],
            "confidence": predicted_prob.item(),
            "probabilities": {name: prob.item() for name, prob in zip(CLASS_NAMES, probabilities)}
        }

        return JSONResponse(content=result)

    except Exception as e:
        # Xatolikni konsolga chiqarish (debugging uchun foydali)
        print(f"Inference xatosi: {e}")
        raise HTTPException(status_code=500, detail=f"Bashorat qilishda xato: {e}")


# === Ishga tushirish (lokal test qilish uchun) ===
if __name__ == "__main__":
    import uvicorn
    # Statik fayllarni taqdim etish uchun root_path="" kerak bo'lmasligi kerak
    # Lekin agar localhost dan boshqa joyda deploy qilsangiz kerak bo'lishi mumkin
    uvicorn.run(app, host="0.0.0.0", port=8000)
