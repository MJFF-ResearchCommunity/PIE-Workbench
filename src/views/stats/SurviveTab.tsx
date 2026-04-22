import { useState, useMemo } from 'react';
import { Loader2, AlertTriangle, CheckCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import MultiSelect from './shared/MultiSelect';
import { TabProps, sig, formatP } from './shared/types';

type Mode = 'km' | 'cox';

export default function SurviveTab({ cacheKey, columns }: TabProps) {
  const { addToast, setLastPValues } = useStore();
  const [mode, setMode] = useState<Mode>('km');
  const [loading, setLoading] = useState(false);

  const numericOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );
  const eventOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_numeric || (c.unique_count != null && c.unique_count <= 5)).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );
  const groupOpts = useMemo(
    () => [{ value: '', label: '— none (single curve) —' }, ...columns.filter((c) => c.is_categorical || (c.unique_count != null && c.unique_count <= 20)).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );
  const numericMultiOpts = useMemo(
    () => columns.filter((c) => c.is_numeric || (c.unique_count != null && c.unique_count <= 10)).map((c) => ({ value: c.name, label: c.name })),
    [columns]
  );

  // Shared inputs
  const [time, setTime] = useState('');
  const [event, setEvent] = useState('');

  // KM
  const [group, setGroup] = useState('');
  const [kmResult, setKmResult] = useState<any>(null);
  const [logrank, setLogrank] = useState<any>(null);

  // Cox
  const [covariates, setCovariates] = useState<string[]>([]);
  const [coxResult, setCoxResult] = useState<any>(null);

  const runKM = async () => {
    if (!time || !event) return addToast('Pick time and event variables', 'error');
    setLoading(true);
    try {
      const [km, lr] = await Promise.all([
        statsApi.surviveKM({ cache_key: cacheKey, time, event, group: group || null }),
        group ? statsApi.surviveLogrank({ cache_key: cacheKey, time, event, group }) : Promise.resolve({ data: null } as any),
      ]);
      setKmResult(km.data);
      setLogrank(lr?.data ?? null);
      if (lr?.data?.p_value != null) setLastPValues([lr.data.p_value]);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Survival analysis failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runCox = async () => {
    if (!time || !event || covariates.length === 0) return addToast('Pick time, event, and ≥1 covariate', 'error');
    setLoading(true);
    try {
      const r = await statsApi.surviveCox({ cache_key: cacheKey, time, event, covariates });
      setCoxResult(r.data);
      setLastPValues(r.data.coefficients.map((c: any) => c.p_value));
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Cox regression failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  // KM chart data
  const kmChartData = useMemo(() => {
    if (!kmResult) return [];
    const { timeline, survival } = kmResult;
    return (timeline as number[]).map((t, i) => {
      const row: Record<string, number> = { time: t };
      for (const key of Object.keys(survival)) {
        row[key] = survival[key][i];
      }
      return row;
    });
  }, [kmResult]);

  const kmGroups = kmResult ? Object.keys(kmResult.survival) : [];

  return (
    <div className="space-y-5">
      <div className="flex rounded-lg bg-pie-surface p-1 max-w-md">
        {(['km', 'cox'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={clsx(
              'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all',
              mode === m ? 'bg-pie-accent text-white' : 'text-pie-text-muted hover:text-pie-text'
            )}
          >
            {m === 'km' ? 'Kaplan-Meier + Log-rank' : 'Cox Proportional Hazards'}
          </button>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Shared inputs</CardTitle>
          <CardDescription>Time to event and event indicator (1 = event occurred, 0 = censored).</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <Select label="Time variable" value={time} onChange={(e) => setTime(e.target.value)} options={numericOpts} />
            <Select label="Event variable" value={event} onChange={(e) => setEvent(e.target.value)} options={eventOpts} />
          </div>
        </CardContent>
      </Card>

      {mode === 'km' && (
        <Card>
          <CardHeader>
            <CardTitle>Kaplan-Meier Estimator</CardTitle>
            <CardDescription>Non-parametric survival curves. With a grouping variable, the log-rank test compares survival across groups.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Select label="Grouping (optional)" value={group} onChange={(e) => setGroup(e.target.value)} options={groupOpts} />
            <Button onClick={runKM} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Fit KM
            </Button>
            {kmResult && (
              <div className="space-y-3">
                {logrank && (
                  <div className={clsx(
                    'px-3 py-2 rounded text-sm flex items-center gap-2',
                    logrank.p_value < 0.05 ? 'bg-pie-warning/10 text-pie-warning' : 'bg-pie-surface text-pie-text'
                  )}>
                    {logrank.p_value < 0.05 ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle className="w-4 h-4" />}
                    Log-rank: χ²={sig(logrank.statistic)}, p={formatP(logrank.p_value)} across {logrank.n_groups} group(s)
                  </div>
                )}
                <div className="h-72 p-4 rounded-lg bg-pie-surface">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={kmChartData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} label={{ value: 'time', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis domain={[0, 1]} stroke="#9ca3af" fontSize={11} label={{ value: 'S(t)', angle: -90, position: 'insideLeft', fill: '#9ca3af', fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />
                      <Legend />
                      {kmGroups.map((g, i) => (
                        <Line key={g} type="stepAfter" dataKey={g} stroke={`hsl(${15 + i * 50}, 75%, 60%)`} dot={false} strokeWidth={2} isAnimationActive={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {mode === 'cox' && (
        <Card>
          <CardHeader>
            <CardTitle>Cox Proportional Hazards</CardTitle>
            <CardDescription>Semi-parametric hazard regression. Hazard ratios quantify the relative risk of each covariate; the Schoenfeld residual test flags PH-assumption violations.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <MultiSelect label="Covariates" options={numericMultiOpts} value={covariates} onChange={setCovariates} placeholder="Pick covariates…" />
            <Button onClick={runCox} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Fit Cox model
            </Button>
            {coxResult && <CoxResult result={coxResult} />}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function CoxResult({ result }: { result: any }) {
  const concord = result.concordance;
  const concordLabel = concord >= 0.7 ? 'good' : concord >= 0.6 ? 'fair' : 'poor';
  const concordColor = concord >= 0.7 ? 'text-pie-success' : concord >= 0.6 ? 'text-pie-warning' : 'text-red-400';

  // Forest plot data — one row per covariate
  const forest = result.coefficients.map((c: any) => ({
    name: c.predictor, hr: c.hazard_ratio, lo: c.hr_ci_lower, hi: c.hr_ci_upper,
  }));

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-3">
        <Stat k="n" v={result.n.toLocaleString()} />
        <Stat k="events" v={result.n_events.toLocaleString()} />
        <Stat
          k="Concordance"
          v={<span><span className="font-mono">{sig(concord)}</span> <span className={clsx('ml-2 text-xs', concordColor)}>({concordLabel})</span></span>}
        />
        <Stat k="log-lik" v={sig(result.log_likelihood)} />
      </div>

      <div className="p-4 rounded-lg bg-pie-surface overflow-x-auto">
        <div className="text-sm text-pie-text-muted mb-2">Hazard ratios</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-pie-border text-pie-text-muted">
              <th className="text-left py-1.5 px-2">predictor</th>
              <th className="text-right px-2">coef</th>
              <th className="text-right px-2">HR</th>
              <th className="text-right px-2">95% CI (HR)</th>
              <th className="text-right px-2">z</th>
              <th className="text-right px-2">p</th>
            </tr>
          </thead>
          <tbody>
            {result.coefficients.map((c: any) => (
              <tr key={c.predictor} className="border-b border-pie-border/50">
                <td className="py-1.5 px-2 font-mono">{c.predictor}</td>
                <td className="text-right px-2">{sig(c.coef)}</td>
                <td className="text-right px-2 font-mono">{sig(c.hazard_ratio)}</td>
                <td className="text-right px-2 font-mono">[{sig(c.hr_ci_lower)}, {sig(c.hr_ci_upper)}]</td>
                <td className="text-right px-2">{sig(c.z_statistic)}</td>
                <td className="text-right px-2 font-mono">{formatP(c.p_value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Forest plot — simple horizontal-bar visualization */}
      <div className="p-4 rounded-lg bg-pie-surface">
        <div className="text-sm text-pie-text-muted mb-3">Hazard-ratio forest plot (log scale)</div>
        <ForestPlot rows={forest} />
      </div>

      {result.ph_test?.length > 0 && (
        <div className="p-4 rounded-lg bg-pie-surface">
          <div className="text-sm text-pie-text-muted mb-2">Proportional-hazards assumption test (Schoenfeld)</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-pie-border text-pie-text-muted">
                <th className="text-left py-1.5 px-2">predictor</th>
                <th className="text-right px-2">statistic</th>
                <th className="text-right px-2">p</th>
                <th className="text-right px-2">PH violation?</th>
              </tr>
            </thead>
            <tbody>
              {result.ph_test.map((p: any) => (
                <tr key={p.predictor} className="border-b border-pie-border/50">
                  <td className="py-1.5 px-2 font-mono">{p.predictor}</td>
                  <td className="text-right px-2">{sig(p.test_statistic)}</td>
                  <td className="text-right px-2 font-mono">{formatP(p.p_value)}</td>
                  <td className={clsx('text-right px-2', p.violates_ph && 'text-pie-warning')}>{p.violates_ph ? '⚠ yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ForestPlot({ rows }: { rows: { name: string; hr: number; lo: number; hi: number }[] }) {
  // Log-scale bounds
  const lo = Math.min(...rows.map((r) => r.lo), 0.25);
  const hi = Math.max(...rows.map((r) => r.hi), 4);
  const logLo = Math.log10(lo);
  const logHi = Math.log10(hi);
  const pos = (v: number) => ((Math.log10(v) - logLo) / (logHi - logLo)) * 100;

  return (
    <div className="space-y-2">
      {rows.map((r) => (
        <div key={r.name} className="grid grid-cols-[140px_1fr_100px] items-center gap-3">
          <div className="text-sm font-mono text-pie-text truncate" title={r.name}>{r.name}</div>
          <div className="relative h-6">
            <div className="absolute inset-y-1/2 left-0 right-0 h-px bg-pie-border" />
            <div className="absolute h-full w-px bg-pie-border" style={{ left: `${pos(1)}%` }} title="HR=1 (no effect)" />
            <div
              className="absolute top-1/2 -translate-y-1/2 h-0.5 bg-pie-accent"
              style={{ left: `${pos(r.lo)}%`, width: `${pos(r.hi) - pos(r.lo)}%` }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 bg-pie-accent rounded-full"
              style={{ left: `calc(${pos(r.hr)}% - 5px)` }}
            />
          </div>
          <div className="text-xs font-mono text-right text-pie-text">
            {sig(r.hr)} [{sig(r.lo)}, {sig(r.hi)}]
          </div>
        </div>
      ))}
      <div className="flex justify-between text-[10px] text-pie-text-muted mt-1 ml-[140px] mr-[100px]">
        <span>{sig(lo)}</span>
        <span>1.0</span>
        <span>{sig(hi)}</span>
      </div>
    </div>
  );
}

function Stat({ k, v }: { k: string; v: any }) {
  return (
    <div className="p-3 rounded-lg bg-pie-surface">
      <div className="text-xs text-pie-text-muted">{k}</div>
      <div className="text-lg font-bold text-pie-text">{v}</div>
    </div>
  );
}
