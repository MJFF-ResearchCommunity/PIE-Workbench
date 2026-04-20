import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  BarChart3,
  Plus,
  Activity,
  TrendingUp,
  Clock,
  Loader2,
} from 'lucide-react';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import Button from '../components/ui/Button';
import Select from '../components/ui/Select';
import { useStore } from '../store/useStore';
import { statsApi, dataApi } from '../services/api';
import { clsx } from 'clsx';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

interface ColumnInfo {
  name: string;
  dtype: string;
  is_numeric: boolean;
  is_categorical: boolean;
}

interface StatResult {
  test_name: string;
  description: string;
  p_value: number;
  significant: boolean;
  interpretation: string;
  statistic?: number;
  correlation?: number;
  // For t-test / ANOVA: { mean: { groupA: 1.2, ... }, std: { ... }, count: { ... } }
  group_statistics?: {
    mean?: Record<string, number>;
    std?: Record<string, number>;
    count?: Record<string, number>;
  };
  // For chi-square
  chi2_statistic?: number;
  degrees_of_freedom?: number;
  contingency_table?: Record<string, Record<string, number>>;
}

interface ScatterResult {
  test_name: string;
  method: 'pearson' | 'spearman' | 'kendall';
  x_variable: string;
  y_variable: string;
  n: number;
  n_plotted: number;
  correlation: number;
  p_value: number;
  significant: boolean;
  regression: {
    slope: number;
    intercept: number;
    r_squared: number;
    endpoints: { x: number; y: number }[];
  };
  points: { x: number; y: number }[];
  interpretation: string;
}

interface SurvivalCurve {
  group: string;
  timeline: number[];
  survival: number[];
  median_survival: number | null;
  n_subjects?: number;
  n_events?: number;
  n_censored?: number;
  follow_up_max?: number | null;
}

// ---------------------------------------------------------------------------
// Result-panel sub-components
// ---------------------------------------------------------------------------

function GroupStatsBlock({
  stats,
  yLabel,
}: {
  stats: { mean?: Record<string, number>; std?: Record<string, number>; count?: Record<string, number> };
  yLabel: string;
}) {
  const groups = Object.keys(stats.mean || {});
  if (groups.length === 0) return null;

  const palette = ['#ff6b4a', '#4ecdc4', '#a78bfa', '#facc15', '#34d399', '#f472b6'];
  const chartData = groups.map((g) => ({
    group: g,
    mean: stats.mean?.[g] ?? 0,
    std: stats.std?.[g] ?? 0,
  }));

  return (
    <div className="p-3 rounded-lg bg-pie-surface border border-pie-border space-y-3">
      <h5 className="font-medium text-pie-text text-sm">Group Means — {yLabel}</h5>

      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
            <XAxis dataKey="group" stroke="#8b9dc3" fontSize={11} />
            <YAxis stroke="#8b9dc3" fontSize={11} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a2540', border: '1px solid #2a3a5c' }}
              labelStyle={{ color: '#e8eff8' }}
              formatter={(v: number) => v.toFixed(3)}
            />
            <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={palette[i % palette.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Tabular per-group stats */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-pie-text-muted border-b border-pie-border">
              <th className="text-left py-1 pr-2 font-medium">Group</th>
              <th className="text-right py-1 px-2 font-medium">n</th>
              <th className="text-right py-1 px-2 font-medium">Mean</th>
              <th className="text-right py-1 pl-2 font-medium">SD</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g, i) => (
              <tr key={g} className="border-b border-pie-border/40">
                <td className="py-1 pr-2 text-pie-text">
                  <span
                    className="inline-block w-2 h-2 rounded-full mr-2 align-middle"
                    style={{ background: palette[i % palette.length] }}
                  />
                  {g}
                </td>
                <td className="text-right py-1 px-2 font-mono text-pie-text">
                  {stats.count?.[g] != null ? Math.round(stats.count[g]) : '—'}
                </td>
                <td className="text-right py-1 px-2 font-mono text-pie-text">
                  {stats.mean?.[g]?.toFixed(3) ?? '—'}
                </td>
                <td className="text-right py-1 pl-2 font-mono text-pie-text-muted">
                  {stats.std?.[g] != null && Number.isFinite(stats.std[g])
                    ? stats.std[g].toFixed(3)
                    : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ScatterResultsBlock({ result }: { result: ScatterResult }) {
  const r = result.correlation;
  const absR = Math.abs(r);
  const strength =
    absR >= 0.8 ? 'very strong' :
    absR >= 0.6 ? 'strong' :
    absR >= 0.4 ? 'moderate' :
    absR >= 0.2 ? 'weak' : 'very weak';
  const direction = r >= 0 ? 'positive' : 'negative';
  const accent = r >= 0 ? '#4ecdc4' : '#ff6b4a';

  // Combine the regression line endpoints with the scatter cloud so they
  // render in the same coordinate space.
  const lineData = result.regression.endpoints;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-3"
    >
      {/* Test info */}
      <div className="p-3 rounded-lg bg-pie-surface">
        <h4 className="font-semibold text-pie-text">{result.test_name}</h4>
        <p className="text-xs text-pie-text-muted mt-0.5">
          {result.x_variable} vs {result.y_variable}
          <span className="ml-2 text-pie-text-muted/70">· n = {result.n.toLocaleString()}</span>
          {result.n_plotted < result.n && (
            <span className="ml-1 text-pie-text-muted/70">
              (plotting {result.n_plotted.toLocaleString()})
            </span>
          )}
        </p>
      </div>

      {/* Headline metrics: r, p, R² */}
      <div className="grid grid-cols-3 gap-2">
        <div className="p-2.5 rounded-lg bg-pie-surface border border-pie-border">
          <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">
            {result.method === 'kendall' ? 'Kendall τ' : result.method === 'spearman' ? 'Spearman ρ' : 'Pearson r'}
          </div>
          <div className="text-xl font-mono font-bold mt-0.5" style={{ color: accent }}>
            {r.toFixed(4)}
          </div>
          <div className="text-[11px] text-pie-text-muted mt-0.5 capitalize">
            {strength} {direction}
          </div>
        </div>
        <div
          className={clsx(
            'p-2.5 rounded-lg border',
            result.significant
              ? 'bg-pie-success/10 border-pie-success/50'
              : 'bg-pie-surface border-pie-border'
          )}
        >
          <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">P-value</div>
          <div
            className={clsx(
              'text-xl font-mono font-bold mt-0.5',
              result.significant ? 'text-pie-success' : 'text-pie-text'
            )}
          >
            {result.p_value < 0.0001 ? '< 0.0001' : result.p_value.toFixed(4)}
          </div>
          <div className="text-[11px] mt-0.5">
            {result.significant
              ? <span className="text-pie-success">✓ Significant</span>
              : <span className="text-pie-text-muted">Not significant</span>}
          </div>
        </div>
        <div className="p-2.5 rounded-lg bg-pie-surface border border-pie-border">
          <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">R²</div>
          <div className="text-xl font-mono font-bold text-pie-text mt-0.5">
            {result.regression.r_squared.toFixed(4)}
          </div>
          <div className="text-[11px] text-pie-text-muted mt-0.5">
            {(result.regression.r_squared * 100).toFixed(1)}% variance
          </div>
        </div>
      </div>

      {/* Scatter + regression line */}
      <div className="p-3 rounded-lg bg-pie-surface border border-pie-border">
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
              <XAxis
                type="number"
                dataKey="x"
                name={result.x_variable}
                stroke="#8b9dc3"
                fontSize={11}
                tickFormatter={(v: number) => formatTick(v)}
                label={{ value: result.x_variable, position: 'insideBottom', offset: -8, fill: '#8b9dc3', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={result.y_variable}
                stroke="#8b9dc3"
                fontSize={11}
                tickFormatter={(v: number) => formatTick(v)}
                label={{ value: result.y_variable, angle: -90, position: 'insideLeft', fill: '#8b9dc3', fontSize: 11 }}
              />
              <ZAxis range={[12, 12]} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ backgroundColor: '#1a2540', border: '1px solid #2a3a5c' }}
                labelStyle={{ color: '#e8eff8' }}
                formatter={(value: number, name: string) => [Number(value).toFixed(3), name]}
              />
              <Scatter
                name="observations"
                data={result.points}
                fill={accent}
                fillOpacity={0.45}
                shape="circle"
              />
              <Scatter
                name="regression"
                data={lineData}
                line={{ stroke: '#facc15', strokeWidth: 2 }}
                lineType="fitting"
                shape={() => null as any}
                legendType="none"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="text-[11px] text-pie-text-muted mt-1 font-mono">
          Best fit:&nbsp;y = {result.regression.slope.toFixed(4)} · x
          {result.regression.intercept >= 0 ? ' + ' : ' − '}
          {Math.abs(result.regression.intercept).toFixed(4)}
        </div>
      </div>

      {/* Interpretation */}
      <div className="p-3 rounded-lg bg-pie-card border border-pie-border">
        <h5 className="font-medium text-pie-text text-sm mb-1">Interpretation</h5>
        <p className="text-xs text-pie-text-muted leading-relaxed">{result.interpretation}</p>
      </div>
    </motion.div>
  );
}

// Compact axis tick formatter — keeps wide-range medical data readable.
function formatTick(v: number): string {
  if (!Number.isFinite(v)) return '';
  const abs = Math.abs(v);
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  if (abs > 0 && abs < 0.01) return v.toExponential(1);
  return v.toFixed(abs < 1 ? 3 : abs < 10 ? 2 : 1);
}

function ContingencyTable({ table }: { table: Record<string, Record<string, number>> }) {
  // The pandas `.to_dict()` orientation is { y_value: { x_value: count } }.
  const yValues = Object.keys(table);
  const xValues = Array.from(
    new Set(yValues.flatMap((y) => Object.keys(table[y] || {}))),
  );
  if (yValues.length === 0 || xValues.length === 0) return null;

  // Compute max for cell-shading
  const max = Math.max(
    1,
    ...yValues.flatMap((y) => xValues.map((x) => Number(table[y]?.[x] ?? 0))),
  );

  return (
    <div className="p-3 rounded-lg bg-pie-surface border border-pie-border">
      <h5 className="font-medium text-pie-text text-sm mb-2">Contingency Table</h5>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr>
              <th className="text-left py-1 pr-2 text-pie-text-muted font-medium"></th>
              {xValues.map((x) => (
                <th key={x} className="text-right py-1 px-2 text-pie-text-muted font-medium">
                  {x}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {yValues.map((y) => (
              <tr key={y} className="border-t border-pie-border/40">
                <td className="py-1 pr-2 text-pie-text font-medium">{y}</td>
                {xValues.map((x) => {
                  const v = Number(table[y]?.[x] ?? 0);
                  const intensity = v / max;
                  return (
                    <td
                      key={x}
                      className="text-right py-1 px-2 font-mono text-pie-text"
                      style={{ background: `rgba(255, 107, 74, ${0.05 + intensity * 0.3})` }}
                    >
                      {v}
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

export default function StatsLab() {
  const navigate = useNavigate();
  const { project, data, addToast } = useStore();
  
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [activeTab, setActiveTab] = useState<'comparison' | 'correlation' | 'survival'>('comparison');
  
  // Comparison state
  const [xVariable, setXVariable] = useState('');
  const [yVariable, setYVariable] = useState('');
  const [statResult, setStatResult] = useState<StatResult | null>(null);

  // Correlation state
  const [corrMethod, setCorrMethod] = useState<'pearson' | 'spearman' | 'kendall'>('pearson');
  const [scatterResult, setScatterResult] = useState<ScatterResult | null>(null);
  
  // Survival state
  const [timeVariable, setTimeVariable] = useState('');
  const [eventVariable, setEventVariable] = useState('');
  const [groupVariable, setGroupVariable] = useState('');
  const [survivalCurves, setSurvivalCurves] = useState<SurvivalCurve[]>([]);
  const [survivalStats, setSurvivalStats] = useState<{
    logrank?: { test_statistic: number; p_value: number; significant: boolean };
  } | null>(null);
  
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!project || !data.loaded) {
      navigate('/data');
      return;
    }
    loadColumns();
  }, [project, data.loaded, navigate]);

  const loadColumns = async () => {
    if (!data.cacheKey) return;
    
    try {
      const response = await dataApi.getColumns(data.cacheKey);
      setColumns(response.data.columns);
    } catch (error) {
      console.error('Failed to load columns:', error);
    }
  };

  const runStatisticalTest = async () => {
    if (!data.cacheKey || !xVariable || !yVariable) {
      addToast('Please select both variables', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await statsApi.autoTest({
        cache_key: data.cacheKey,
        x_variable: xVariable,
        y_variable: yVariable,
      });
      setStatResult(response.data);
    } catch (error) {
      addToast('Failed to run statistical test', 'error');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const runCorrelation = async () => {
    if (!data.cacheKey || !xVariable || !yVariable) {
      addToast('Please select both variables', 'error');
      return;
    }
    if (xVariable === yVariable) {
      addToast('Pick two different variables', 'error');
      return;
    }

    setLoading(true);
    setScatterResult(null);
    try {
      const response = await statsApi.scatter({
        cache_key: data.cacheKey,
        x_variable: xVariable,
        y_variable: yVariable,
        method: corrMethod,
      });
      setScatterResult(response.data);
    } catch (error: any) {
      const detail = error?.response?.data?.detail || 'Failed to compute correlation';
      addToast(detail, 'error');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const runSurvivalAnalysis = async () => {
    if (!data.cacheKey || !timeVariable || !eventVariable) {
      addToast('Please select time and event variables', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await statsApi.survival({
        cache_key: data.cacheKey,
        time_variable: timeVariable,
        event_variable: eventVariable,
        grouping_variable: groupVariable || undefined,
      });
      setSurvivalCurves(response.data.curves);
      setSurvivalStats(response.data.statistics);
    } catch (error) {
      addToast('Failed to run survival analysis', 'error');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const numericColumns = columns.filter((c) => c.is_numeric);
  const categoricalColumns = columns.filter((c) => c.is_categorical);
  const allColumnOptions = columns.map((c) => ({
    value: c.name,
    label: `${c.name} (${c.is_numeric ? 'numeric' : 'categorical'})`,
  }));

  const tabs = [
    { id: 'comparison' as const, label: 'Group Comparison', icon: BarChart3 },
    { id: 'correlation' as const, label: 'Correlation', icon: TrendingUp },
    { id: 'survival' as const, label: 'Survival Analysis', icon: Clock },
  ];

  // Prepare survival chart data
  const survivalChartData = survivalCurves.length > 0 
    ? survivalCurves[0].timeline.map((time, i) => {
        const point: Record<string, number> = { time };
        survivalCurves.forEach((curve) => {
          point[curve.group] = curve.survival[i];
        });
        return point;
      })
    : [];

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold text-pie-text mb-2">
          Statistical Workbench
        </h1>
        <p className="text-pie-text-muted">
          Perform statistical tests and visualize relationships in your data
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg transition-all',
                activeTab === tab.id
                  ? 'bg-pie-accent text-white'
                  : 'bg-pie-surface text-pie-text-muted hover:text-pie-text'
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="grid grid-cols-2 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>
              {activeTab === 'comparison' && 'Compare Groups'}
              {activeTab === 'correlation' && 'Correlation Analysis'}
              {activeTab === 'survival' && 'Survival Analysis'}
            </CardTitle>
            <CardDescription>
              {activeTab === 'comparison' && 'Drag variables to compare across groups'}
              {activeTab === 'correlation' && 'Analyze relationships between numeric variables'}
              {activeTab === 'survival' && 'Configure Kaplan-Meier analysis'}
            </CardDescription>
          </CardHeader>

          <CardContent>
            {activeTab === 'comparison' && (
              <div className="space-y-6">
                {/* Drop Zones */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-pie-text">X-Axis (Groups)</label>
                    <Select
                      value={xVariable}
                      onChange={(e) => setXVariable(e.target.value)}
                      options={allColumnOptions}
                      placeholder="Select grouping variable..."
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-pie-text">Y-Axis (Values)</label>
                    <Select
                      value={yVariable}
                      onChange={(e) => setYVariable(e.target.value)}
                      options={allColumnOptions}
                      placeholder="Select outcome variable..."
                    />
                  </div>
                </div>

                <Button
                  variant="primary"
                  className="w-full"
                  onClick={runStatisticalTest}
                  disabled={!xVariable || !yVariable || loading}
                  loading={loading}
                >
                  <Activity className="w-4 h-4" />
                  Run Auto Test
                </Button>
              </div>
            )}

            {activeTab === 'correlation' && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Select
                    label="X Variable"
                    value={xVariable}
                    onChange={(e) => setXVariable(e.target.value)}
                    options={numericColumns.map((c) => ({ value: c.name, label: c.name }))}
                    placeholder="Select variable..."
                  />
                  <Select
                    label="Y Variable"
                    value={yVariable}
                    onChange={(e) => setYVariable(e.target.value)}
                    options={numericColumns.map((c) => ({ value: c.name, label: c.name }))}
                    placeholder="Select variable..."
                  />
                </div>

                <Select
                  label="Correlation method"
                  value={corrMethod}
                  onChange={(e) => setCorrMethod(e.target.value as typeof corrMethod)}
                  options={[
                    { value: 'pearson', label: 'Pearson (linear)' },
                    { value: 'spearman', label: 'Spearman (rank)' },
                    { value: 'kendall', label: 'Kendall (rank, robust)' },
                  ]}
                />

                <Button
                  variant="primary"
                  className="w-full"
                  onClick={runCorrelation}
                  disabled={!xVariable || !yVariable || loading}
                  loading={loading}
                >
                  <TrendingUp className="w-4 h-4" />
                  Calculate Correlation
                </Button>
              </div>
            )}

            {activeTab === 'survival' && (
              <div className="space-y-4">
                <Select
                  label="Time Variable"
                  value={timeVariable}
                  onChange={(e) => setTimeVariable(e.target.value)}
                  options={numericColumns.map((c) => ({ value: c.name, label: c.name }))}
                  placeholder="Select time variable..."
                />
                
                <Select
                  label="Event Variable (1=event, 0=censored)"
                  value={eventVariable}
                  onChange={(e) => setEventVariable(e.target.value)}
                  options={allColumnOptions}
                  placeholder="Select event variable..."
                />
                
                <Select
                  label="Grouping Variable (optional)"
                  value={groupVariable}
                  onChange={(e) => setGroupVariable(e.target.value)}
                  options={[{ value: '', label: 'None' }, ...categoricalColumns.map((c) => ({ value: c.name, label: c.name }))]}
                />

                <Button
                  variant="primary"
                  className="w-full"
                  onClick={runSurvivalAnalysis}
                  disabled={!timeVariable || !eventVariable || loading}
                  loading={loading}
                >
                  <Clock className="w-4 h-4" />
                  Run Kaplan-Meier Analysis
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
          </CardHeader>

          <CardContent>
            {loading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-pie-accent" />
              </div>
            )}

            {!loading && activeTab === 'correlation' && scatterResult && (
              <ScatterResultsBlock result={scatterResult} />
            )}

            {!loading && activeTab === 'comparison' && statResult && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-3"
              >
                {/* Test Info */}
                <div className="p-3 rounded-lg bg-pie-surface">
                  <h4 className="font-semibold text-pie-text">{statResult.test_name}</h4>
                  <p className="text-xs text-pie-text-muted mt-0.5">{statResult.description}</p>
                </div>

                {/* Headline metrics: statistic + p-value (and r when present) */}
                <div className="grid grid-cols-2 gap-3">
                  {(() => {
                    const stat =
                      statResult.statistic ?? statResult.chi2_statistic ?? statResult.correlation;
                    const statLabel = statResult.correlation !== undefined
                      ? 'Pearson r'
                      : statResult.chi2_statistic !== undefined
                      ? `χ² (df=${statResult.degrees_of_freedom ?? '?'})`
                      : statResult.test_name.includes('ANOVA')
                      ? 'F-statistic'
                      : 't-statistic';
                    return stat !== undefined ? (
                      <div className="p-3 rounded-lg bg-pie-surface border border-pie-border">
                        <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">
                          {statLabel}
                        </div>
                        <div className="text-2xl font-mono font-bold text-pie-text mt-1">
                          {stat.toFixed(4)}
                        </div>
                      </div>
                    ) : <div />;
                  })()}
                  <div
                    className={clsx(
                      'p-3 rounded-lg border',
                      statResult.significant
                        ? 'bg-pie-success/10 border-pie-success/50'
                        : 'bg-pie-surface border-pie-border'
                    )}
                  >
                    <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">P-value</div>
                    <div
                      className={clsx(
                        'text-2xl font-mono font-bold mt-1',
                        statResult.significant ? 'text-pie-success' : 'text-pie-text'
                      )}
                    >
                      {statResult.p_value < 0.0001 ? '< 0.0001' : statResult.p_value.toFixed(4)}
                    </div>
                    <div className="text-[11px] mt-1">
                      {statResult.significant
                        ? <span className="text-pie-success">✓ Significant (p &lt; 0.05)</span>
                        : <span className="text-pie-text-muted">Not significant</span>}
                    </div>
                  </div>
                </div>

                {/* Per-group statistics — t-test / ANOVA */}
                {statResult.group_statistics?.mean && (
                  <GroupStatsBlock stats={statResult.group_statistics} yLabel={yVariable} />
                )}

                {/* Contingency table — chi-square */}
                {statResult.contingency_table && (
                  <ContingencyTable table={statResult.contingency_table} />
                )}

                {/* Interpretation */}
                <div className="p-3 rounded-lg bg-pie-card border border-pie-border">
                  <h5 className="font-medium text-pie-text text-sm mb-1">Interpretation</h5>
                  <p className="text-xs text-pie-text-muted leading-relaxed">{statResult.interpretation}</p>
                </div>
              </motion.div>
            )}

            {!loading && activeTab === 'survival' && survivalCurves.length > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-4"
              >
                {/* Kaplan-Meier Curve */}
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={survivalChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#8b9dc3"
                        fontSize={12}
                        label={{ value: 'Time', position: 'bottom', fill: '#8b9dc3' }}
                      />
                      <YAxis 
                        stroke="#8b9dc3"
                        fontSize={12}
                        domain={[0, 1]}
                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                        label={{ value: 'Survival', angle: -90, position: 'left', fill: '#8b9dc3' }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1a2540', border: '1px solid #2a3a5c' }}
                        labelStyle={{ color: '#e8eff8' }}
                      />
                      <Legend />
                      {survivalCurves.map((curve, i) => (
                        <Line
                          key={curve.group}
                          type="stepAfter"
                          dataKey={curve.group}
                          stroke={i === 0 ? '#ff6b4a' : '#4ecdc4'}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Per-group survival summary */}
                <div
                  className="grid gap-3"
                  style={{
                    gridTemplateColumns: `repeat(${Math.min(survivalCurves.length, 3)}, minmax(0, 1fr))`,
                  }}
                >
                  {survivalCurves.map((curve, i) => {
                    const color = i === 0 ? '#ff6b4a' : i === 1 ? '#4ecdc4' : '#a78bfa';
                    const eventPct =
                      curve.n_subjects && curve.n_events !== undefined
                        ? (curve.n_events / curve.n_subjects) * 100
                        : null;
                    return (
                      <div key={curve.group} className="p-3 rounded-lg bg-pie-surface border border-pie-border">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="inline-block w-2 h-2 rounded-full" style={{ background: color }} />
                          <h5 className="font-medium text-pie-text text-sm truncate">{curve.group}</h5>
                        </div>
                        <dl className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <dt className="text-pie-text-muted">Subjects</dt>
                            <dd className="font-mono text-pie-text">{curve.n_subjects ?? '—'}</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt className="text-pie-text-muted">Events</dt>
                            <dd className="font-mono text-pie-text">
                              {curve.n_events ?? '—'}
                              {eventPct !== null && (
                                <span className="text-pie-text-muted ml-1">({eventPct.toFixed(1)}%)</span>
                              )}
                            </dd>
                          </div>
                          <div className="flex justify-between">
                            <dt className="text-pie-text-muted">Censored</dt>
                            <dd className="font-mono text-pie-text">{curve.n_censored ?? '—'}</dd>
                          </div>
                          <div className="flex justify-between">
                            <dt className="text-pie-text-muted">Median surv.</dt>
                            <dd className="font-mono text-pie-text">
                              {curve.median_survival !== null
                                ? curve.median_survival.toFixed(1)
                                : 'Not reached'}
                            </dd>
                          </div>
                          {curve.follow_up_max != null && (
                            <div className="flex justify-between">
                              <dt className="text-pie-text-muted">Max follow-up</dt>
                              <dd className="font-mono text-pie-text">{curve.follow_up_max.toFixed(1)}</dd>
                            </div>
                          )}
                        </dl>
                      </div>
                    );
                  })}
                </div>

                {/* Log-rank test — chi2 + p-value */}
                {survivalStats?.logrank && (
                  <div
                    className={clsx(
                      'p-3 rounded-lg border',
                      survivalStats.logrank.significant
                        ? 'bg-pie-success/10 border-pie-success/50'
                        : 'bg-pie-surface border-pie-border'
                    )}
                  >
                    <h5 className="font-medium text-pie-text text-sm mb-2">Log-rank Test</h5>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">χ² statistic</div>
                        <div className="text-xl font-mono font-bold text-pie-text mt-0.5">
                          {survivalStats.logrank.test_statistic.toFixed(4)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">P-value</div>
                        <div
                          className={clsx(
                            'text-xl font-mono font-bold mt-0.5',
                            survivalStats.logrank.significant ? 'text-pie-success' : 'text-pie-text'
                          )}
                        >
                          {survivalStats.logrank.p_value < 0.0001
                            ? '< 0.0001'
                            : survivalStats.logrank.p_value.toFixed(4)}
                        </div>
                      </div>
                    </div>
                    {survivalStats.logrank.significant && (
                      <p className="text-xs text-pie-success mt-2">
                        ✓ Survival differs significantly between groups (p &lt; 0.05)
                      </p>
                    )}
                  </div>
                )}
              </motion.div>
            )}

            {!loading && !statResult && !scatterResult && survivalCurves.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Plus className="w-12 h-12 text-pie-text-muted mb-4" />
                <p className="text-pie-text-muted">
                  Select variables and run a test to see results
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
