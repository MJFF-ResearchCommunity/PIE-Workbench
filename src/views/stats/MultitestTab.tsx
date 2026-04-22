import { useEffect, useState } from 'react';
import { Loader2, Info } from 'lucide-react';
import { clsx } from 'clsx';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import { TabProps, sig, formatP } from './shared/types';

export default function MultitestTab(_props: TabProps) {
  const { addToast, lastPValues } = useStore();
  const [text, setText] = useState('');
  const [method, setMethod] = useState('fdr_bh');
  const [alpha, setAlpha] = useState(0.05);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Auto-populate with the last batch of p-values the user ran elsewhere
  useEffect(() => {
    if (lastPValues.length > 0 && !text) {
      setText(lastPValues.map((p) => p.toFixed(6)).join('\n'));
    }
  }, [lastPValues, text]);

  const run = async () => {
    const parsed: number[] = [];
    for (const token of text.split(/[\s,]+/)) {
      const t = token.trim();
      if (!t) continue;
      const n = parseFloat(t);
      if (isNaN(n) || n < 0 || n > 1) {
        addToast(`Skipping invalid p-value: ${t}`, 'warning');
        continue;
      }
      parsed.push(n);
    }
    if (parsed.length === 0) return addToast('Provide at least one p-value', 'error');
    setLoading(true);
    try {
      const r = await statsApi.multitestAdjust({ p_values: parsed, method, alpha });
      setResult(r.data);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Adjustment failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const rejectedCount = result ? result.rejected.filter(Boolean).length : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Multiple-Testing Correction</CardTitle>
        <CardDescription>
          Adjust p-values for multiplicity. Automatically pre-populates from your most recent Compare / Correlate / Regress / LMM / Cox result.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-pie-text mb-2">
            P-values (one per line, or comma/whitespace separated)
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={6}
            className="w-full px-4 py-3 rounded-lg bg-pie-surface border border-pie-border text-pie-text font-mono text-sm focus:outline-none focus:ring-2 focus:ring-pie-accent/50"
            placeholder="0.01&#10;0.04&#10;0.0032&#10;0.12"
          />
          {lastPValues.length > 0 && (
            <div className="mt-2 text-xs text-pie-text-muted flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5" /> Auto-loaded {lastPValues.length} p-value(s) from your last analysis. Edit freely.
            </div>
          )}
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Select
            label="Method"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            options={[
              { value: 'bonferroni', label: 'Bonferroni' },
              { value: 'holm', label: 'Holm-Bonferroni' },
              { value: 'sidak', label: 'Šidák' },
              { value: 'fdr_bh', label: 'Benjamini-Hochberg (FDR)' },
              { value: 'fdr_by', label: 'Benjamini-Yekutieli (FDR)' },
              { value: 'fdr_tsbh', label: 'Two-stage BH (FDR)' },
            ]}
          />
          <Select
            label="α"
            value={alpha.toString()}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            options={[
              { value: '0.01', label: '0.01' },
              { value: '0.05', label: '0.05' },
              { value: '0.10', label: '0.10' },
            ]}
          />
          <div className="flex items-end">
            <Button onClick={run} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Adjust
            </Button>
          </div>
        </div>

        {result && (
          <div className="space-y-3">
            <div className="p-3 rounded bg-pie-surface text-sm">
              <span className="text-pie-text-muted">After {result.method} correction at α={result.alpha}: </span>
              <span className="font-semibold text-pie-text">{rejectedCount}</span> of {result.original.length} rejected
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-pie-border text-pie-text-muted">
                    <th className="text-left py-1.5 px-2">#</th>
                    <th className="text-right px-2">original p</th>
                    <th className="text-right px-2">adjusted p</th>
                    <th className="text-right px-2">reject?</th>
                  </tr>
                </thead>
                <tbody>
                  {result.original.map((p: number, i: number) => (
                    <tr
                      key={i}
                      className={clsx(
                        'border-b border-pie-border/50',
                        result.rejected[i] && 'bg-pie-accent/5'
                      )}
                    >
                      <td className="py-1.5 px-2 text-pie-text-muted">{i + 1}</td>
                      <td className="text-right px-2 font-mono">{formatP(p)}</td>
                      <td className="text-right px-2 font-mono">{sig(result.adjusted[i], 4)}</td>
                      <td className={clsx('text-right px-2', result.rejected[i] && 'text-pie-accent font-medium')}>
                        {result.rejected[i] ? '✓' : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
