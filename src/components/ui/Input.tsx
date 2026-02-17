import { InputHTMLAttributes, forwardRef } from 'react';
import { clsx } from 'clsx';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, helperText, id, ...props }, ref) => {
    const inputId = id || `input-${Math.random().toString(36).substring(7)}`;

    return (
      <div className="space-y-2">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-sm font-medium text-pie-text"
          >
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          className={clsx(
            'w-full px-4 py-2.5 bg-pie-surface border rounded-lg',
            'text-pie-text placeholder-pie-text-muted',
            'focus:outline-none focus:ring-2 focus:ring-pie-accent focus:border-transparent',
            'transition-all duration-200',
            error
              ? 'border-red-500 focus:ring-red-500'
              : 'border-pie-border hover:border-pie-text-muted',
            className
          )}
          {...props}
        />
        {(error || helperText) && (
          <p
            className={clsx(
              'text-sm',
              error ? 'text-red-400' : 'text-pie-text-muted'
            )}
          >
            {error || helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
