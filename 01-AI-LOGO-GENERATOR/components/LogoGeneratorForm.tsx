import React from 'react';

interface LogoGeneratorFormProps {
  prompt: string;
  setPrompt: (prompt: string) => void;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
}

const LoadingSpinner: React.FC = () => (
    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);

export const LogoGeneratorForm: React.FC<LogoGeneratorFormProps> = ({ prompt, setPrompt, onSubmit, isLoading }) => {
  return (
    <form onSubmit={onSubmit} className="space-y-6 bg-slate-800 p-6 rounded-lg border border-slate-700 shadow-lg">
      <div className="flex flex-col space-y-2">
        <label htmlFor="prompt" className="text-sm font-medium text-slate-300">
            Logo Description
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., A minimalist fox logo, geometric style, orange and white"
          rows={6}
          className="w-full p-3 bg-slate-700 border border-slate-600 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow duration-200 text-gray-100 placeholder-slate-400"
          required
        />
        <p className="text-xs text-slate-400 pt-1">Be descriptive! Mention style, objects, colors, and the company name.</p>
      </div>
      
      <button
        type="submit"
        disabled={isLoading}
        className="w-full flex items-center justify-center py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white font-bold rounded-md transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-blue-500 shadow-md hover:shadow-lg"
      >
        {isLoading ? (
          <>
            <LoadingSpinner />
            Generating...
          </>
        ) : (
          'Generate Logo'
        )}
      </button>
    </form>
  );
};
