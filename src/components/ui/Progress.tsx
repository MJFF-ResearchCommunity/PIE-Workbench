import { clsx } from 'clsx';
import { motion } from 'framer-motion';

interface ProgressProps {
  value: number;
  max?: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
  variant?: 'default' | 'gradient';
}

export default function Progress({
  value,
  max = 100,
  size = 'md',
  showLabel = false,
  label,
  variant = 'default',
}: ProgressProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  return (
    <div className="space-y-2">
      {(showLabel || label) && (
        <div className="flex justify-between items-center">
          <span className="text-sm text-pie-text-muted">{label}</span>
          {showLabel && (
            <span className="text-sm font-medium text-pie-text">
              {Math.round(percentage)}%
            </span>
          )}
        </div>
      )}
      <div
        className={clsx(
          'w-full bg-pie-surface rounded-full overflow-hidden',
          {
            'h-1': size === 'sm',
            'h-2': size === 'md',
            'h-3': size === 'lg',
          }
        )}
      >
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={clsx(
            'h-full rounded-full',
            {
              'bg-pie-accent': variant === 'default',
              'bg-gradient-to-r from-pie-accent to-pie-accent-secondary': variant === 'gradient',
            }
          )}
        />
      </div>
    </div>
  );
}
