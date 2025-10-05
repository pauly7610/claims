'use client';

import React from 'react';

interface SmartInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  hint?: string;
  aiSuggestion?: string;
  icon?: React.ReactNode;
  onAIAssist?: () => void;
}

const SparklesIcon = () => (
  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3l14 14M5 21l14-14" />
  </svg>
);

export const SmartInput: React.FC<SmartInputProps> = ({
  label,
  error,
  hint,
  aiSuggestion,
  icon,
  onAIAssist,
  className = '',
  ...props
}) => {
  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-sm font-medium text-gray-700">{label}</label>
      )}
      <div className="relative">
        {icon && (
          <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
            {icon}
          </div>
        )}
        <input
          className={`
            w-full px-4 py-3 ${icon ? 'pl-10' : ''} 
            border-2 rounded-lg transition-all duration-200
            focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20
            ${error ? 'border-red-300 focus:border-red-500 focus:ring-red-500/20' : 'border-gray-200'}
            bg-white hover:border-gray-300
            ${className}
          `}
          {...props}
        />
        {aiSuggestion && onAIAssist && (
          <button
            type="button"
            onClick={onAIAssist}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-blue-500 hover:text-blue-600 transition-colors"
          >
            <SparklesIcon />
          </button>
        )}
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {hint && <p className="text-sm text-gray-500">{hint}</p>}
      {aiSuggestion && (
        <div className="flex items-center space-x-2 text-sm text-blue-600 bg-blue-50 px-3 py-2 rounded-md">
          <SparklesIcon />
          <span>AI Suggestion: {aiSuggestion}</span>
        </div>
      )}
    </div>
  );
};
