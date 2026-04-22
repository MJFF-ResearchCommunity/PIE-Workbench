export interface ColumnInfo {
  name: string;
  dtype: string;
  is_numeric: boolean;
  is_categorical: boolean;
  unique_count?: number;
}

export interface TabProps {
  cacheKey: string;
  columns: ColumnInfo[];
}

/** Format a p-value with star annotation. */
export function formatP(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return 'n/a';
  if (p < 0.001) return '<0.001 ***';
  if (p < 0.01) return p.toFixed(3) + ' **';
  if (p < 0.05) return p.toFixed(3) + ' *';
  return p.toFixed(3);
}

/** Round a number to k significant figures for table display. */
export function sig(n: number | null | undefined, k: number = 3): string {
  if (n == null || Number.isNaN(n)) return 'n/a';
  if (n === 0) return '0';
  const magnitude = Math.floor(Math.log10(Math.abs(n)));
  const factor = Math.pow(10, k - 1 - magnitude);
  return (Math.round(n * factor) / factor).toString();
}

/** Cohen's d interpretation. */
export function interpretCohensD(d: number): { label: string; color: string } {
  const abs = Math.abs(d);
  if (abs < 0.2) return { label: 'negligible', color: 'text-pie-text-muted' };
  if (abs < 0.5) return { label: 'small', color: 'text-blue-400' };
  if (abs < 0.8) return { label: 'medium', color: 'text-pie-warning' };
  return { label: 'large', color: 'text-pie-success' };
}

/** η² interpretation. */
export function interpretEtaSquared(e: number): { label: string; color: string } {
  if (e < 0.01) return { label: 'negligible', color: 'text-pie-text-muted' };
  if (e < 0.06) return { label: 'small', color: 'text-blue-400' };
  if (e < 0.14) return { label: 'medium', color: 'text-pie-warning' };
  return { label: 'large', color: 'text-pie-success' };
}
