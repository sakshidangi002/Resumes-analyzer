import React, { useState, useRef, useEffect } from 'react';

interface Option {
  value: string | number;
  label: string;
  disabled?: boolean;
}

interface CustomSelectProps {
  value: string | number;
  onChange: (value: any) => void;
  options: Option[];
  style?: React.CSSProperties;
  className?: string;
  placeholder?: string;
  disabled?: boolean;
  onToggle?: (isOpen: boolean) => void;
}

const CustomSelect: React.FC<CustomSelectProps> = ({ value, onChange, options, style, className, placeholder, disabled, onToggle }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const selectedOption = options.find(opt => opt.value === value) || (placeholder ? null : options[0]);

  const toggle = (val: boolean) => {
    setIsOpen(val);
    if (onToggle) onToggle(val);
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        if (isOpen) toggle(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  return (
    <div 
      ref={containerRef}
      className={`custom-select-container ${className || ''}`}
      style={{ 
        position: 'relative', 
        width: '100%',
        height: '42px',
        userSelect: 'none',
        zIndex: isOpen ? 9999 : 1,
        ...style 
      }}
    >
      <div 
        className="custom-select-trigger"
        onClick={() => {
          if (disabled) return;
          toggle(!isOpen);
        }}
        style={{
          height: '42px',
          padding: '0 1rem',
          background: 'rgba(255, 255, 255, 0.06)',
          border: '1px solid rgba(255, 255, 255, 0.14)',
          borderRadius: '10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: disabled ? 'not-allowed' : 'pointer',
          color: '#fff',
          opacity: disabled ? 0.6 : 1,
          fontSize: '0.875rem'
        }}
      >
        <span>{selectedOption ? selectedOption.label : placeholder}</span>
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          width="12" height="12" 
          viewBox="0 0 24 24" fill="none" 
          stroke="white" strokeWidth="2.5" 
          strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}
        >
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </div>

      {isOpen && (
        <div 
          className="custom-select-options"
          style={{
            position: 'absolute',
            top: '46px',
            left: 0,
            right: 0,
            background: '#121212',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '10px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
            zIndex: 9999,
            maxHeight: '250px',
            overflowY: 'auto'
          }}
        >
          {options.map((opt) => (
            <div
              key={opt.value}
              className="custom-select-option"
              onClick={() => {
                if (opt.disabled) return;
                onChange(opt.value);
                toggle(false);
              }}
              style={{
                padding: '0.75rem 1rem',
                cursor: opt.disabled ? 'not-allowed' : 'pointer',
                fontSize: '0.875rem',
                color: '#fff',
                background: 'transparent',
                transition: 'background 0.15s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                opacity: opt.disabled ? 0.5 : 1
              }}
              onMouseEnter={(e) => {
                if (!opt.disabled) e.currentTarget.style.background = '#153273';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
              }}
            >
              <span>{opt.label}</span>
              {opt.value === value && (
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CustomSelect;
