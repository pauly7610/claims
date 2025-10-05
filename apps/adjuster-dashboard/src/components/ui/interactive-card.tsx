'use client';

import React from 'react';

interface InteractiveCardProps {
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
  gradient?: boolean;
  glowing?: boolean;
  hover?: boolean;
}

export const InteractiveCard: React.FC<InteractiveCardProps> = ({
  children,
  onClick,
  className = '',
  gradient = false,
  glowing = false,
  hover = true
}) => {
  return (
    <div
      className={`
        bg-white rounded-2xl shadow-sm border border-gray-200
        ${hover ? 'hover:shadow-lg hover:border-gray-300 transform hover:-translate-y-1' : ''}
        transition-all duration-300 
        ${onClick ? 'cursor-pointer' : ''}
        ${gradient ? 'bg-gradient-to-br from-white to-gray-50' : ''}
        ${glowing ? 'ring-2 ring-blue-500/20 shadow-blue-500/10' : ''}
        ${className}
      `}
      onClick={onClick}
    >
      {children}
    </div>
  );
};

interface CardHeaderProps {
  children: React.ReactNode;
  className?: string;
}

export const CardHeader: React.FC<CardHeaderProps> = ({ children, className = '' }) => (
  <div className={`px-6 py-4 border-b border-gray-100 ${className}`}>
    {children}
  </div>
);

interface CardContentProps {
  children: React.ReactNode;
  className?: string;
}

export const CardContent: React.FC<CardContentProps> = ({ children, className = '' }) => (
  <div className={`px-6 py-4 ${className}`}>
    {children}
  </div>
);

interface CardTitleProps {
  children: React.ReactNode;
  className?: string;
}

export const CardTitle: React.FC<CardTitleProps> = ({ children, className = '' }) => (
  <h3 className={`text-lg font-semibold text-gray-900 ${className}`}>
    {children}
  </h3>
);
