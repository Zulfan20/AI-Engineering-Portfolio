import React from 'react';
// Memuat CSS yang telah kita buat
import './App.css'; 

// Mengimpor komponen LogoGenerator dari folder components
// Pastikan ekstensi .tsx disertakan jika ada masalah resolusi
import LogoGenerator from './components/LogoGenerator.tsx';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Logo Generator</h1>
        <p>
          Runs locally on GPU (NVIDIA RTX 2050)
          <br/>
          Backend: Python (FastAPI) | Frontend: React (TypeScript)
        </p>
        
        <hr />

        {/* Memanggil komponen generator Anda */}
        <LogoGenerator />
        
        <hr />

      </header>
    </div>
  );
}

export default App;