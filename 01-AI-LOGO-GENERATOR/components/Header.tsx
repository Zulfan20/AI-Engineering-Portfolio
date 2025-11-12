import React from 'react';

const ShieldCheckIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 20.944a12.02 12.02 0 009 2.056c4.509 0 8.305-2.585 9.873-6.253A11.95 11.95 0 0021.565 6.452a11.955 11.955 0 01-4.947-2.468z" />
  </svg>
);


export const Header: React.FC = () => {
    return (
        <header className="bg-slate-900/70 backdrop-blur-sm sticky top-0 z-10 border-b border-slate-700">
            <div className="container mx-auto px-4 py-4">
                <div className="flex items-center space-x-3">
                    <ShieldCheckIcon className="w-8 h-8 text-blue-400" />
                    <h1 className="text-2xl font-bold tracking-tight text-white">
                        AI Logo Generator
                    </h1>
                </div>
            </div>
        </header>
    );
};
