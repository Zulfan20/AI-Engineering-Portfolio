import { GoogleGenAI } from "@google/genai";

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export async function generateLogo(prompt: string): Promise<string> {
    try {
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: `A high-resolution, vector-style logo on a clean white background. The logo should be described as: "${prompt}". It should be modern, clean, and suitable for a tech startup.`,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/png',
                aspectRatio: '1:1',
            },
        });
        
        if (response.generatedImages && response.generatedImages.length > 0) {
            const base64ImageBytes: string = response.generatedImages[0].image.imageBytes;
            return `data:image/png;base64,${base64ImageBytes}`;
        } else {
            throw new Error("No image was generated. The prompt may have been blocked.");
        }
    } catch (error) {
        console.error("Error generating logo with Gemini API:", error);
        throw new Error("Failed to generate logo. Please check your prompt or API key.");
    }
}
