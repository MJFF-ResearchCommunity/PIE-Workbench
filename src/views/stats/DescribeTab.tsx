import { useState, useMemo } from 'react';
import { Loader2, CheckCircle, AlertTriangle } from 'lucide-react';
import { clsx } from 'clsx';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar } from 'recharts';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import MultiSelect from './shared/MultiSelect';
import { TabProps, sig, formatP } from './shared/types';

type SummaryEntry = {
  n: number;
  n_missing: number;
  pct_missing: number;
  mean?: number;
  median?: number;
  std?: number;
  min?: number;
  max?: number;
  q1?: number;
  q3?: number;
  iqr?: number;
  skew?: number;
  kurtosis?: number;
};

type NormalityResult = {
  test: string;
  statistic: number;
  p_value: number;
  n: number;
  is_normal: boolean;
};

type MissingnessResult = {
  n_rows: number;
  per_column: Record<string, { n_missing: number; pct_missing: number }>;
  little_mcar: { statistic: number; p_value: number; dof: number; interpretation: string } | null;
};

export default function DescribeTab({ cacheKey, columns }: TabProps) {
  const { addToast } = useStore();
  const numericOpts = useMemo(
    () => columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name })),
    [columns]
  );
  const allOpts = useMemo(
    () =>
      columns.map((c) => ({
        value: c.name,
        label: c.name,
        hint: c.is_numeric ? 'numeric' : 'categorical',
      })),
    [columns]
  );

  const [summaryVars, setSummaryVars] = useState<string[]>([]);
  const [summaryResult, setSummaryResult] = useState<Record<string, SummaryEntry> | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);

  const [normalityVar, setNormalityVar] = useState('');
  const [normalityTest, setNormalityTest] = useState('shapiro');
  const [normalityResult, setNormalityResult] = useState<NormalityResult | null>(null);
  const [normalityPoints, setNormalityPoints] = useState<{ x: number; y: number }[] | null>(null);
  const [normalityLoading, setNormalityLoading] = useState(false);

  const [missVars, setMissVars] = useState<string[]>([]);
  const [missResult, setMissResult] = useState<MissingnessResult | null>(null);
  const [missLoading, setMissLoading] = useState(false);

  const runSummary = async () => {
    if (!summaryVars.length) {
      addToast('Pick at least one numeric variable', 'error');
      return;
    }
    setSummaryLoading(true);
    try {
      const r = await statsApi.describeSummary(cacheKey, summaryVars);
      setSummaryResult(r.data);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Summary failed', 'error');
    } finally {
      setSummaryLoading(false);
    }
  };

  const runNormality = async () => {
    if (!normalityVar) {
      addToast('Pick a variable', 'error');
      return;
    }
    setNormalityLoading(true);
    try {
      const [normRes, sumRes] = await Promise.all([
        statsApi.describeNormality(cacheKey, normalityVar, normalityTest),
        statsApi.describeSummary(cacheKey, [normalityVar]),
      ]);
      setNormalityResult(normRes.data);
      // Build Q-Q plot data by asking the backend for per-variable summary then
      // generating theoretical normal quantiles on the client.
      const entry = sumRes.data[normalityVar] as SummaryEntry;
      if (entry && entry.n > 0 && entry.mean != null && entry.std != null) {
        // We don't have raw values — approximate Q-Q using the 5-point summary
        // (min / Q1 / median / Q3 / max), each paired with the theoretical
        // z-score at that quantile. A proper Q-Q needs the full series, which
        // summaryStatistics doesn't return.
        const qs = [entry.min!, entry.q1!, entry.median!, entry.q3!, entry.max!];
        const theoretical = [-1.96, -0.67, 0, 0.67, 1.96];
        setNormalityPoints(qs.map((q, i) => ({ x: theoretical[i], y: q })));
      }
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Normality test failed', 'error');
    } finally {
      setNormalityLoading(false);
    }
  };

  const runMissingness = async () => {
    setMissLoading(true);
    try {
      const r = await statsApi.describeMissingness(cacheKey, missVars);
      setMissResult(r.data);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Missingness report failed', 'error');
    } finally {
      setMissLoading(false);
    }
  };

  const missChartData = useMemo(() => {
    if (!missResult) return [];
    return Object.entries(missResult.per_column)
      .map(([name, v]) => ({ name, pct: v.pct_missing }))
      .filter((r) => r.pct > 0)
      .sort((a, b) => b.pct - a.pct)
      .slice(0, 20);
  }, [missResult]);

  return (
    <div className="space-y-5">
      {/* Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Summary Statistics</CardTitle>
          <CardDescription>Means, medians, SDs, quantiles, skew, kurtosis, and missingness for numeric columns.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <MultiSelect
            label="Numeric variables"
            options={numericOpts}
            value={summaryVars}
            onChange={setSummaryVars}
            placeholder="Pick one or more numeric columns…"
          />
          <Button onClick={runSummary} disabled={summaryLoading}>
            {summaryLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            Compute summary
          </Button>
          {summaryResult && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-pie-border text-pie-text-muted">
                    <th className="text-left py-2 px-2">variable</th>
                    <th className="text-right px-2">n</th>
                    <th className="text-right px-2">mean</th>
                    <th className="text-right px-2">SD</th>
                    <th className="text-right px-2">median</th>
                    <th className="text-right px-2">Q1</th>
                    <th className="text-right px-2">Q3</th>
                    <th className="text-right px-2">IQR</th>
                    <th className="text-right px-2">min</th>
                    <th className="text-right px-2">max</th>
                    <th className="text-right px-2">skew</th>
                    <th className="text-right px-2">kurtosis</th>
                    <th className="text-right px-2">%missing</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(summaryResult)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([name, s]) => (
                      <tr key={name} className="border-b border-pie-border/50 hover:bg-pie-surface/50">
                        <td className="py-1.5 px-2 font-mono text-pie-text">{name}</td>
                        <td className="text-right px-2">{s.n}</td>
                        <td className="text-right px-2">{sig(s.mean)}</td>
                        <td className="text-right px-2">{sig(s.std)}</td>
                        <td className="text-right px-2">{sig(s.median)}</td>
                        <td className="text-right px-2">{sig(s.q1)}</td>
                        <td className="text-right px-2">{sig(s.q3)}</td>
                        <td className="text-right px-2">{sig(s.iqr)}</td>
                        <td className="text-right px-2">{sig(s.min)}</td>
                        <td className="text-right px-2">{sig(s.max)}</td>
                        <td className="text-right px-2">{sig(s.skew)}</td>
                        <td className="text-right px-2">{sig(s.kurtosis)}</td>
                        <td className={clsx('text-right px-2', s.pct_missing > 10 && 'text-pie-warning')}>
                          {s.pct_missing.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Normality */}
      <Card>
        <CardHeader>
          <CardTitle>Normality Test</CardTitle>
          <CardDescription>Shapiro-Wilk (n ≤ 5000) or Kolmogorov-Smirnov vs. fitted normal, plus a 5-quantile Q-Q sketch.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <Select
              label="Variable"
              value={normalityVar}
              onChange={(e) => setNormalityVar(e.target.value)}
              options={[{ value: '', label: '— pick —' }, ...numericOpts]}
            />
            <Select
              label="Test"
              value={normalityTest}
              onChange={(e) => setNormalityTest(e.target.value)}
              options={[
                { value: 'shapiro', label: 'Shapiro-Wilk' },
                { value: 'ks', label: 'Kolmogorov-Smirnov' },
              ]}
            />
            <div className="flex items-end">
              <Button onClick={runNormality} disabled={normalityLoading}>
                {normalityLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                Run test
              </Button>
            </div>
          </div>
          {normalityResult && (
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-pie-surface">
                <div className="text-xs text-pie-text-muted mb-1">Result</div>
                <div className="text-lg font-semibold text-pie-text mb-2">{normalityResult.test === 'shapiro' ? 'Shapiro-Wilk' : 'Kolmogorov-Smirnov'}</div>
                <div className="text-sm space-y-1">
                  <div className="flex justify-between"><span>n</span><span className="font-mono">{normalityResult.n}</span></div>
                  <div className="flex justify-between"><span>statistic</span><span className="font-mono">{sig(normalityResult.statistic)}</span></div>
                  <div className="flex justify-between"><span>p-value</span><span className="font-mono">{formatP(normalityResult.p_value)}</span></div>
                </div>
                <div className={clsx(
                  'mt-3 px-3 py-2 rounded text-sm flex items-center gap-2',
                  normalityResult.is_normal ? 'bg-pie-success/10 text-pie-success' : 'bg-pie-warning/10 text-pie-warning'
                )}>
                  {normalityResult.is_normal ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                  {normalityResult.is_normal
                    ? 'Consistent with a normal distribution (p > 0.05)'
                    : 'Rejects normality at α=0.05 — consider non-parametric tests'}
                </div>
              </div>
              {normalityPoints && (
                <div className="h-56">
                  <div className="text-xs text-pie-text-muted mb-1">Q-Q sketch</div>
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis type="number" dataKey="x" label={{ value: 'theoretical quantile', position: 'bottom', fill: '#9ca3af', fontSize: 11 }} stroke="#9ca3af" fontSize={11} />
                      <YAxis type="number" dataKey="y" label={{ value: 'observed', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} stroke="#9ca3af" fontSize={11} />
                      <ReferenceLine segment={[{ x: -2, y: (normalityPoints[0]?.y ?? 0) + 0 }, { x: 2, y: (normalityPoints[4]?.y ?? 0) }]} stroke="#f97316" strokeDasharray="3 3" />
                      <Tooltip cursor={{ fill: 'rgba(249,115,22,0.1)' }} contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />
                      <Scatter data={normalityPoints} fill="#f97316" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Missingness */}
      <Card>
        <CardHeader>
          <CardTitle>Missingness Report</CardTitle>
          <CardDescription>Per-column missingness, with a chi-square pattern test flagging non-MCAR behavior.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <MultiSelect
            label="Variables (leave empty for all columns)"
            options={allOpts}
            value={missVars}
            onChange={setMissVars}
            placeholder="Leave empty to scan all columns…"
          />
          <Button onClick={runMissingness} disabled={missLoading}>
            {missLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            Scan missingness
          </Button>
          {missResult && (
            <div className="space-y-3">
              {missResult.little_mcar && (
                <div className={clsx(
                  'px-3 py-2 rounded text-sm flex items-center gap-2',
                  missResult.little_mcar.p_value > 0.05 ? 'bg-pie-success/10 text-pie-success' : 'bg-pie-warning/10 text-pie-warning'
                )}>
                  {missResult.little_mcar.p_value > 0.05 ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                  <span>{missResult.little_mcar.interpretation} (χ²={sig(missResult.little_mcar.statistic)}, df={missResult.little_mcar.dof}, p={formatP(missResult.little_mcar.p_value)})</span>
                </div>
              )}
              {missChartData.length > 0 && (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={missChartData} margin={{ top: 10, right: 20, bottom: 50, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" angle={-30} textAnchor="end" stroke="#9ca3af" fontSize={10} interval={0} />
                      <YAxis stroke="#9ca3af" fontSize={11} label={{ value: '% missing', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />
                      <Bar dataKey="pct" fill="#f97316" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
              <div className="text-xs text-pie-text-muted">
                {missResult.n_rows.toLocaleString()} rows scanned; showing columns with any missingness (top 20).
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
