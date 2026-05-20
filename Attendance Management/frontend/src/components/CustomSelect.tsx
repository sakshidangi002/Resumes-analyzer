import React, { useState, useRef, useEffect, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';

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
  const triggerRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuRect, setMenuRect] = useState<{ top: number; left: number; width: number; openUp: boolean }>({
    top: 0,
    left: 0,
    width: 0,
    openUp: false,
  });

  const selectedOption = options.find(opt => opt.value === value) || (placeholder ? null : options[0]);

  const toggle = (val: boolean) => {
    setIsOpen(val);
    if (onToggle) onToggle(val);
  };

  const computeMenuPosition = () => {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const rect = trigger.getBoundingClientRect();
    const estimatedMenuHeight = Math.min(250, Math.max(120, options.length * 42));
    const spaceBelow = window.innerHeight - rect.bottom;
    const openUp = spaceBelow < estimatedMenuHeight + 16 && rect.top > spaceBelow;
    setMenuRect({
      top: openUp ? rect.top - 4 : rect.bottom + 4,
      left: rect.left,
      width: rect.width,
      openUp,
    });
  };

  useLayoutEffect(() => {
    if (!isOpen) return;
    computeMenuPosition();
    const handler = () => computeMenuPosition();
    window.addEventListener('scroll', handler, true);
    window.addEventListener('resize', handler);
    return () => {
      window.removeEventListener('scroll', handler, true);
      window.removeEventListener('resize', handler);
    };
  }, [isOpen]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      const inTrigger = containerRef.current?.contains(target);
      const inMenu = menuRef.current?.contains(target);
      if (!inTrigger && !inMenu && isOpen) toggle(false);
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
        ...style
      }}
    >
      <div
        ref={triggerRef}
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
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="12" height="12"
          viewBox="0 0 24 24" fill="none"
          stroke="white" strokeWidth="2.5"
          strokeLinecap="round" strokeLinejoin="round"
          style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', flexShrink: 0, marginLeft: '0.5rem' }}
        >
          <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
      </div>

      {isOpen && createPortal(
        <div
          ref={menuRef}
          className="custom-select-options"
          style={{
            position: 'fixed',
            top: menuRect.openUp ? undefined : menuRect.top,
            bottom: menuRect.openUp ? window.innerHeight - menuRect.top : undefined,
            left: menuRect.left,
            width: menuRect.width,
            background: '#0b0f19',
            border: '1px solid rgba(255, 255, 255, 0.14)',
            borderRadius: '10px',
            boxShadow: '0 16px 40px rgba(0, 0, 0, 0.55)',
            zIndex: 10000,
            maxHeight: '250px',
            overflowY: 'auto',
            padding: '0.35rem',
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
                padding: '0.6rem 0.85rem',
                borderRadius: '6px',
                cursor: opt.disabled ? 'not-allowed' : 'pointer',
                fontSize: '0.875rem',
                color: '#e2e8f0',
                background: opt.value === value ? '#153273' : 'transparent',
                transition: 'background 0.12s',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                opacity: opt.disabled ? 0.5 : 1
              }}
              onMouseEnter={(e) => {
                if (!opt.disabled && opt.value !== value) e.currentTarget.style.background = 'rgba(21, 50, 115, 0.55)';
              }}
              onMouseLeave={(e) => {
                if (opt.value !== value) e.currentTarget.style.background = 'transparent';
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
        </div>,
        document.body
      )}
    </div>
  );
};

export default CustomSelect;
