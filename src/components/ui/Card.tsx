import { HTMLAttributes, forwardRef } from 'react';
import { clsx } from 'clsx';
import { motion } from 'framer-motion';

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass' | 'elevated';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  animate?: boolean;
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', padding = 'md', animate = true, children, ...props }, ref) => {
    const content = (
      <div
        ref={ref}
        className={clsx(
          'rounded-xl transition-all duration-200',
          {
            // Variants
            'bg-pie-card border border-pie-border': variant === 'default',
            'glass': variant === 'glass',
            'bg-pie-card border border-pie-border shadow-lg shadow-black/20': variant === 'elevated',
            // Padding
            'p-0': padding === 'none',
            'p-4': padding === 'sm',
            'p-6': padding === 'md',
            'p-8': padding === 'lg',
          },
          className
        )}
        {...props}
      >
        {children}
      </div>
    );

    if (animate) {
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          {content}
        </motion.div>
      );
    }

    return content;
  }
);

Card.displayName = 'Card';

export function CardHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx('mb-4', className)} {...props} />;
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={clsx('text-lg font-semibold text-pie-text', className)}
      {...props}
    />
  );
}

export function CardDescription({ className, ...props }: HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p
      className={clsx('text-sm text-pie-text-muted mt-1', className)}
      {...props}
    />
  );
}

export function CardContent({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx('', className)} {...props} />;
}

export function CardFooter({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={clsx('mt-4 pt-4 border-t border-pie-border flex items-center gap-3', className)}
      {...props}
    />
  );
}

export default Card;
