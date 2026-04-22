import { useState, useMemo } from 'react';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import MultiSelect from './shared/MultiSelect';
import { TabProps, sig, formatP } from './shared/types';

type Mode = 'pair' | 'partial' | 'matrix';
type Method = 'pearson' | 'spearman' | 'kendall';

export default function CorrelateTab({ cacheKey, columns }: TabProps) {
  const { addToast, setLastPValues } = useStore();
  const [mode, setMode] = useState<Mode>('pair');
  const [method, setMethod] = useState<Method>('pearson');
  const [loading, setLoading] = useState(false);

  const numericOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );
  const multiNumericOpts = useMemo(
    () => columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name })),
    [columns]
  );

  // Pair
  const [pairX, setPairX] = useState('');
  const [pairY, setPairY] = useState('');
  const [pairResult, setPairResult] = useState<any>(null);

  // Partial
  const [partX, setPartX] = useState('');
  const [partY, setPartY] = useState('');
  const [partCovars, setPartCovars] = useState<string[]>([]);
  const [partResult, setPartResult] = useState<any>(null);
  const [partDirect, setPartDirect] = useState<any>(null);

  // Matrix
  const [matVars, setMatVars] = useState<string[]>([]);
  const [matResult, setMatResult] = useState<any>(null);

  const runPair = async () => {
    if (!pairX || !pairY) return addToast('Pick both variables', 'error');
    if (pairX === pairY) return addToast('Pick two different variables', 'error');
    setLoading(true);
    try {
      const r = await statsApi.scatter({ cache_key: cacheKey, x_variable: pairX, y_variable: pairY, method });
      setPairResult(r.data);
      setLastPValues([r.data.p_value]);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Correlation failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runPartial = async () => {
    if (!partX || !partY || partCovars.length === 0) {
      return addToast('Pick x, y, and at least one covariate', 'error');
    }
    setLoading(true);
    try {
      const [part, direct] = await Promise.all([
        statsApi.correlatePartial({ cache_key: cacheKey, x: partX, y: partY, covariates: partCovars, method }),
        statsApi.scatter({ cache_key: cacheKey, x_variable: partX, y_variable: partY, method }),
      ]);
      setPartResult(part.data);
      setPartDirect(direct.data);
      setLastPValues([part.data.p_value]);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Partial correlation failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runMatrix = async () => {
    if (matVars.length < 2) return addToast('Pick at least 2 variables', 'error');
    setLoading(true);
    try {
      const r = await statsApi.correlateMatrix({ cache_key: cacheKey, variables: matVars, method });
      setMatResult(r.data);
      // Collect off-diagonal p-values for hand-off
      const pvals: number[] = [];
      for (let i = 0; i < matVars.length; i++) {
        for (let j = i + 1; j < matVars.length; j++) {
          pvals.push(r.data.p_values[matVars[i]][matVars[j]]);
        }
      }
      setLastPValues(pvals);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Matrix correlation failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex items-end gap-4 flex-wrap">
        <div className="flex rounded-lg bg-pie-surface p-1 max-w-md">
          {(['pair', 'partial', 'matrix'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={clsx(
                'py-2 px-3 rounded-md text-sm font-medium transition-all',
                mode === m ? 'bg-pie-accent text-white' : 'text-pie-text-muted hover:text-pie-text'
              )}
            >
              {m === 'pair' ? 'Pairwise' : m === 'partial' ? 'Partial' : 'Matrix'}
            </button>
          ))}
        </div>
        <div className="w-48">
          <Select
            label="Method"
            value={method}
            onChange={(e) => setMethod(e.target.value as Method)}
            options={[
              { value: 'pearson', label: 'Pearson' },
              { value: 'spearman', label: 'Spearman' },
              { value: 'kendall', label: "Kendall's τ" },
            ]}
          />
        </div>
      </div>

      {mode === 'pair' && (
        <Card>
          <CardHeader>
            <CardTitle>Pairwise Correlation</CardTitle>
            <CardDescription>Scatter plot with regression fit and the chosen correlation coefficient.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Select label="X" value={pairX} onChange={(e) => setPairX(e.target.value)} options={numericOpts} />
              <Select label="Y" value={pairY} onChange={(e) => setPairY(e.target.value)} options={numericOpts} />
            </div>
            <Button onClick={runPair} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Compute
            </Button>
            {pairResult && (
              <div className="grid grid-cols-3 gap-4">
                <div className="p-4 rounded-lg bg-pie-surface space-y-2">
                  <div className="text-sm text-pie-text-muted">{pairResult.method}</div>
                  <div className="text-2xl font-bold text-pie-text">r = {sig(pairResult.correlation)}</div>
                  <div className="text-sm">p = <span className="font-mono">{formatP(pairResult.p_value)}</span></div>
                  <div className="text-xs text-pie-text-muted">n = {pairResult.n.toLocaleString()}</div>
                </div>
                <div className="col-span-2 h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis type="number" dataKey="x" name={pairX} stroke="#9ca3af" fontSize={11} />
                      <YAxis type="number" dataKey="y" name={pairY} stroke="#9ca3af" fontSize={11} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />
                      <Scatter data={pairResult.points} fill="#f97316" fillOpacity={0.5} />
                      {pairResult.regression?.endpoints && (
                        <Scatter data={pairResult.regression.endpoints} line={{ stroke: '#14b8a6', strokeWidth: 2 }} shape={() => <g />} />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {mode === 'partial' && (
        <Card>
          <CardHeader>
            <CardTitle>Partial Correlation</CardTitle>
            <CardDescription>Correlation of X and Y after regressing out the chosen covariates. Essential for disentangling confounders (e.g. is age → UPDRS the real story, or is it disease duration?).</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Select label="X" value={partX} onChange={(e) => setPartX(e.target.value)} options={numericOpts} />
              <Select label="Y" value={partY} onChange={(e) => setPartY(e.target.value)} options={numericOpts} />
            </div>
            <MultiSelect
              label="Covariates to adjust for"
              options={multiNumericOpts}
              value={partCovars}
              onChange={setPartCovars}
              placeholder="Pick one or more numeric covariates…"
            />
            <Button onClick={runPartial} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Compute partial correlation
            </Button>
            {partResult && partDirect && (
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-pie-surface">
                  <div className="text-sm text-pie-text-muted mb-1">Without adjustment</div>
                  <div className="text-xl font-bold">r = {sig(partDirect.correlation)}</div>
                  <div className="text-sm">p = <span className="font-mono">{formatP(partDirect.p_value)}</span></div>
                </div>
                <div className="p-4 rounded-lg bg-pie-accent/10 border border-pie-accent/40">
                  <div className="text-sm text-pie-accent mb-1">After adjusting for {partCovars.join(', ')}</div>
                  <div className="text-xl font-bold text-pie-text">r = {sig(partResult.r)}</div>
                  <div className="text-sm">p = <span className="font-mono">{formatP(partResult.p_value)}</span></div>
                  <div className="text-xs text-pie-text-muted mt-1">n = {partResult.n.toLocaleString()}</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {mode === 'matrix' && (
        <Card>
          <CardHeader>
            <CardTitle>Correlation Matrix</CardTitle>
            <CardDescription>Heatmap of all pairwise correlations. Cells with bold borders have a Benjamini-Hochberg FDR-adjusted p &lt; 0.05.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <MultiSelect
              label="Numeric variables"
              options={multiNumericOpts}
              value={matVars}
              onChange={setMatVars}
              placeholder="Pick 2 or more numeric columns…"
            />
            <Button onClick={runMatrix} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Compute matrix
            </Button>
            {matResult && <CorrelationHeatmap variables={matVars} result={matResult} />}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function CorrelationHeatmap({ variables, result }: { variables: string[]; result: any }) {
  // HSL color per r value: -1 = deep blue, +1 = deep orange
  const colorFor = (r: number) => {
    const clamped = Math.max(-1, Math.min(1, r));
    const hue = clamped < 0 ? 210 : 18;
    const L = 65 - Math.abs(clamped) * 35;
    return `hsl(${hue}, 70%, ${L}%)`;
  };
  return (
    <div>
      <div className="text-xs text-pie-text-muted mb-2">n = {result.n.toLocaleString()} complete rows · FDR method: {result.fdr_method}</div>
      <div className="overflow-x-auto">
        <table className="border-separate" style={{ borderSpacing: 2 }}>
          <thead>
            <tr>
              <th className="sticky left-0 bg-pie-card"></th>
              {variables.map((v) => (
                <th key={v} className="px-2 py-1 text-xs font-mono text-pie-text-muted min-w-[80px]" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
                  {v}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {variables.map((v) => (
              <tr key={v}>
                <th className="sticky left-0 bg-pie-card px-2 py-1 text-xs font-mono text-right text-pie-text-muted">{v}</th>
                {variables.map((w) => {
                  const r = result.matrix[v][w];
                  const pAdj = result.p_values_adjusted[v][w];
                  const sig05 = v !== w && pAdj < 0.05;
                  return (
                    <td
                      key={w}
                      className={clsx(
                        'text-center text-xs font-mono text-white min-w-[54px] h-[28px] rounded',
                        sig05 && 'ring-2 ring-pie-accent'
                      )}
                      style={{ background: colorFor(r) }}
                      title={`${v} ~ ${w}: r=${r.toFixed(3)}, p(adj)=${pAdj.toExponential(1)}`}
                    >
                      {r.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
