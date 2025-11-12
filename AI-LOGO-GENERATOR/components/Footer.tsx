import React from 'react';

export const Footer: React.FC = () => {
    return (
        <footer className="bg-slate-900 border-t border-slate-800 mt-auto">
            <div className="container mx-auto px-4 py-6 text-center text-slate-400">
                <p>&copy; {new Date().getFullYear()} AI Logo Generator. Powered by Google Gemini.</p>
            </div>
        </footer>
    );
};
