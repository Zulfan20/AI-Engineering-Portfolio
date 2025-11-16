import torch
from torch import autocast
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

# 1. Inisialisasi Aplikasi FastAPI
app = FastAPI()

# 2. Konfigurasi CORS (SANGAT PENTING)
# Ini mengizinkan aplikasi React Anda (misal: localhost:5173)
# untuk berbicara dengan API backend Anda (localhost:8000)
origins = [
    "http://localhost:5173", # Ganti ini jika port React Anda berbeda
    "http://localhost:3000", # Port React default lainnya
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Request Body (agar FastAPI tahu data apa yang dikirim React)
class ImageRequest(BaseModel):
    prompt: str

# 4. Muat Model AI ke GPU (HANYA SEKALI SAAT STARTUP)
print("Memuat model Stable Diffusion 1.5 ke VRAM...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# == OPTIMASI UNTUK RTX 2050 (4GB VRAM) ==
# Pindahkan model ke GPU
pipe = pipe.to("cuda")
# Ini akan menghemat VRAM dengan mengorbankan sedikit kecepatan
pipe.enable_attention_slicing() 
print("Model berhasil dimuat ke GPU (cuda)!")


# 5. Definisikan Endpoint API Anda
@app.post("/generate-logo")
async def generate_logo(req: ImageRequest):
    print(f"Menerima prompt: {req.prompt}")

    # Jalankan model di GPU
    # 'with autocast' membantu mempercepat proses di kartu seri RTX
    with autocast("cuda"):
        # Anda bisa menambah parameter lain di sini, misal: num_inference_steps=25
        image = pipe(req.prompt, height=512, width=512).images[0]
    
    print("Gambar berhasil dibuat.")

    # 6. Simpan gambar ke memori (bukan file)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0) # Kembali ke awal file di memori

    # 7. Kirim gambar kembali ke React
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")