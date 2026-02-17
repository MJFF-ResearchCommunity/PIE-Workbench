import { ButtonHTMLAttributes, forwardRef } from 'react';
import { clsx } from 'clsx';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, children, disabled, ...props }, ref) => {
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={clsx(
          'inline-flex items-center justify-center font-medium rounded-lg transition-all duration-200',
          'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-pie-bg',
          {
            // Variants
            'bg-gradient-to-r from-pie-accent to-pie-accent/80 text-white hover:from-pie-accent/90 hover:to-pie-accent/70 focus:ring-pie-accent':
              variant === 'primary',
            'bg-pie-card border border-pie-border text-pie-text hover:bg-pie-surface focus:ring-pie-border':
              variant === 'secondary',
            'text-pie-text-muted hover:text-pie-text hover:bg-pie-card focus:ring-pie-border':
              variant === 'ghost',
            'bg-red-500/20 border border-red-500/50 text-red-400 hover:bg-red-500/30 focus:ring-red-500':
              variant === 'danger',
            // Sizes
            'text-sm px-3 py-1.5 gap-1.5': size === 'sm',
            'text-sm px-4 py-2 gap-2': size === 'md',
            'text-base px-6 py-3 gap-2': size === 'lg',
            // States
            'opacity-50 cursor-not-allowed': disabled || loading,
          },
          className
        )}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;
