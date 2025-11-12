import React from 'react';

interface LogoDisplayProps {
  generatedImage: string | null;
  isLoading: boolean;
  error: string | null;
}

const Placeholder: React.FC = () => (
    <div className="flex flex-col items-center justify-center h-full text-center text-slate-500">
        <svg xmlns="http://www.w3.org/2000/svg" className="w-16 h-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <h3 className="text-lg font-semibold text-slate-400">Your logo will appear here</h3>
        <p className="text-sm">Describe your vision and click "Generate Logo" to start.</p>
    </div>
);

const LoadingIndicator: React.FC = () => (
    <div className="flex flex-col items-center justify-center h-full text-center text-slate-400">
        <svg className="animate-spin h-12 w-12 text-blue-400 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <h3 className="text-lg font-semibold">Generating Your Masterpiece...</h3>
        <p className="text-sm">The AI is warming up. This might take a moment.</p>
    </div>
);

const ErrorDisplay: React.FC<{ message: string }> = ({ message }) => (
    <div className="flex flex-col items-center justify-center h-full text-center text-red-400 bg-red-900/20 p-4 rounded-md">
        <svg xmlns="http://www.w3.org/2000/svg" className="w-12 h-12 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-lg font-semibold">Oops! Something went wrong.</h3>
        <p className="text-sm">{message}</p>
    </div>
);

export const LogoDisplay: React.FC<LogoDisplayProps> = ({ generatedImage, isLoading, error }) => {
  return (
    <div className="w-full aspect-square bg-slate-800 rounded-lg border border-slate-700 flex items-center justify-center p-4 md:p-8 shadow-lg">
      {isLoading && <LoadingIndicator />}
      {!isLoading && error && <ErrorDisplay message={error} />}
      {!isLoading && !error && generatedImage && (
        <img src={generatedImage} alt="AI generated logo" className="max-w-full max-h-full object-contain rounded-md shadow-2xl"/>
      )}
      {!isLoading && !error && !generatedImage && <Placeholder />}
    </div>
  );
};
