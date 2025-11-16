import React, { useState } from 'react'; // <-- INI DIA PERBAIKANNYA

// API URL sekarang menunjuk ke backend FastAPI LOKAL Anda
const API_URL = "http://127.0.0.1:8000/generate-logo";

function LogoGenerator() {
  const [prompt, setPrompt] = useState("");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async () => {
    if (!prompt) {
      setError("Silakan masukkan prompt.");
      return;
    }

    setLoading(true);
    setError(null);
    
    // Membersihkan URL gambar lama dari memori browser
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      setImageUrl(null);
    }

    try {
      // Mengirim request ke server LOKAL Anda
      const response = await fetch(
        API_URL,
        {
          method: "POST",
          headers: {
            // Kita tidak butuh "Authorization" (API Token) lagi
            "Content-Type": "application/json",
          },
          // Body sekarang HARUS cocok dengan model Pydantic di Python
          body: JSON.stringify({ prompt: prompt }),
        }
      );

      if (!response.ok) {
        // Menangkap error dari server FastAPI Anda
        const errorText = await response.text();
        throw new Error(errorText || "Gagal membuat gambar di server lokal.");
      }

      // Kode di bawah ini sama persis, karena server lokal
      // juga mengembalikan gambar (blob)
      const imageBlob = await response.blob();
      
      const objectURL = URL.createObjectURL(imageBlob);
      setImageUrl(objectURL);

    } catch (err: any) { 
      console.error(err);
      // Pesan error baru yang lebih relevan
      setError(`Error: ${err.message}. Pastikan server backend Python (main.py) Anda sudah berjalan.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="logo-generator">
      <h2>Generator Logo (Stable Diffusion 1.5)</h2>
      
      <input
        type="text"
        className="logo-input" 
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Contoh: A minimalist logo of a robot head, vector art"
        disabled={loading} // Nonaktifkan input saat loading
      />
      <button 
        className="generate-button" 
        onClick={handleGenerate} 
        disabled={loading}
      >
        {loading ? "Generating..." : "Generate Logo"}
      </button>

      {/* Tampilkan pesan error jika ada */}
      {error && <p className="error-message">{error}</p>}
      
      {/* Tampilkan pesan loading BARU */}
      {loading && (
        <div className="loading-message">
          <p><strong>GPU Anda sedang bekerja...</strong></p>
          <p>Membuat gambar (512x512) dengan Stable Diffusion 1.5. Ini mungkin butuh 10-20 detik.</p>
        </div>
      )}

      {/* Tampilkan gambar jika sudah ada */}
      {imageUrl && (
        <div className="result-container">
          <h3>Your Logo Results:</h3>
          <img 
            src={imageUrl} 
            alt="Generated AI Logo" 
            className="result-image"
          />
        </div>
      )}
    </div>
  );
}

export default LogoGenerator;