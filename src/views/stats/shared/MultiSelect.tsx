import { useState, useEffect, useRef, useMemo } from 'react';
import { Search, X, Check } from 'lucide-react';
import { clsx } from 'clsx';

export interface MultiSelectOption {
  value: string;
  label: string;
  hint?: string;
}

interface MultiSelectProps {
  label?: string;
  options: MultiSelectOption[];
  value: string[];
  onChange: (value: string[]) => void;
  placeholder?: string;
  maxDropdown?: number;
}

export default function MultiSelect({
  label,
  options,
  value,
  onChange,
  placeholder = 'Select...',
  maxDropdown = 100,
}: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    if (!q) return options.slice(0, maxDropdown);
    return options
      .filter((o) => o.label.toLowerCase().includes(q) || o.value.toLowerCase().includes(q))
      .slice(0, maxDropdown);
  }, [options, search, maxDropdown]);

  const toggle = (v: string) => {
    if (value.includes(v)) onChange(value.filter((x) => x !== v));
    else onChange([...value, v]);
  };

  return (
    <div className="space-y-1.5" ref={ref}>
      {label && <label className="block text-sm font-medium text-pie-text">{label}</label>}

      {/* Chips */}
      <div
        className="min-h-[42px] w-full px-2 py-1.5 rounded-lg bg-pie-surface border border-pie-border flex flex-wrap gap-1.5 cursor-text"
        onClick={() => setOpen(true)}
      >
        {value.length === 0 && (
          <span className="text-sm text-pie-text-muted self-center pl-1">{placeholder}</span>
        )}
        {value.map((v) => {
          const opt = options.find((o) => o.value === v);
          return (
            <span
              key={v}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-pie-accent/20 text-pie-accent text-xs"
            >
              {opt?.label ?? v}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onChange(value.filter((x) => x !== v));
                }}
                className="hover:text-pie-text"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          );
        })}
      </div>

      {open && (
        <div className="relative">
          <div className="absolute z-40 top-0 left-0 w-full max-h-72 overflow-hidden rounded-lg border border-pie-border bg-pie-card shadow-xl">
            <div className="p-2 border-b border-pie-border relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-pie-text-muted pointer-events-none" />
              <input
                autoFocus
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search..."
                className="w-full pl-8 pr-3 py-1.5 rounded bg-pie-surface border border-pie-border text-pie-text text-sm focus:outline-none focus:ring-2 focus:ring-pie-accent/50"
              />
            </div>
            <div className="max-h-56 overflow-y-auto py-1">
              {filtered.length === 0 ? (
                <div className="px-3 py-2 text-sm text-pie-text-muted text-center">No matches</div>
              ) : (
                filtered.map((o) => {
                  const selected = value.includes(o.value);
                  return (
                    <button
                      key={o.value}
                      onClick={() => toggle(o.value)}
                      className={clsx(
                        'w-full text-left px-3 py-1.5 text-sm flex items-center gap-2 hover:bg-pie-surface',
                        selected && 'bg-pie-accent/10 text-pie-accent'
                      )}
                    >
                      <div
                        className={clsx(
                          'w-4 h-4 rounded border flex items-center justify-center flex-shrink-0',
                          selected ? 'bg-pie-accent border-pie-accent' : 'border-pie-border'
                        )}
                      >
                        {selected && <Check className="w-3 h-3 text-white" />}
                      </div>
                      <span className="flex-1 truncate">{o.label}</span>
                      {o.hint && <span className="text-xs text-pie-text-muted">{o.hint}</span>}
                    </button>
                  );
                })
              )}
            </div>
            {filtered.length === maxDropdown && (
              <div className="px-3 py-1.5 text-xs text-pie-text-muted text-center border-t border-pie-border">
                Showing first {maxDropdown} — refine your search
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
