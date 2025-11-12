import React, { useState } from 'react';
import { LogoGeneratorForm } from './components/LogoGeneratorForm';
import { LogoDisplay } from './components/LogoDisplay';
import { generateLogo } from './services/geminiService';
import { Header } from './components/Header';
import { Footer } from './components/Footer';

function App() {
  const [prompt, setPrompt] = useState<string>('A minimalist logo for "mylowker.id", featuring a shield with a checkmark inside. The color scheme should be professional and modern, using shades of blue and white.');
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateLogo = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!prompt || isLoading) return;

    setIsLoading(true);
    setError(null);
    setGeneratedImage(null);

    try {
      const imageDataUrl = await generateLogo(prompt);
      setGeneratedImage(imageDataUrl);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-gray-100 flex flex-col font-sans">
      <Header />
      <main className="flex-grow container mx-auto px-4 py-8 md:py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12">
          <div className="flex flex-col space-y-6">
            <h2 className="text-2xl md:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
              1. Describe Your Logo
            </h2>
            <LogoGeneratorForm
              prompt={prompt}
              setPrompt={setPrompt}
              onSubmit={handleGenerateLogo}
              isLoading={isLoading}
            />
          </div>
          <div className="flex flex-col space-y-6">
             <h2 className="text-2xl md:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
              2. Your AI-Generated Logo
            </h2>
            <LogoDisplay
              generatedImage={generatedImage}
              isLoading={isLoading}
              error={error}
            />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

export default App;
