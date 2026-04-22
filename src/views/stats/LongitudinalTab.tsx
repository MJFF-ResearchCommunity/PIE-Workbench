import { useState, useMemo } from 'react';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import MultiSelect from './shared/MultiSelect';
import { TabProps, sig, formatP } from './shared/types';

type Mode = 'lmm' | 'change';

export default function LongitudinalTab({ cacheKey, columns }: TabProps) {
  const { addToast, setLastPValues } = useStore();
  const [mode, setMode] = useState<Mode>('lmm');
  const [loading, setLoading] = useState(false);

  const numericOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );
  const numericMultiOpts = useMemo(
    () => columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name })),
    [columns]
  );
  const allOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );

  // LMM
  const [lmmOutcome, setLmmOutcome] = useState('');
  const [lmmFixed, setLmmFixed] = useState<string[]>([]);
  const [lmmGroup, setLmmGroup] = useState('');
  const [lmmRandomSlopes, setLmmRandomSlopes] = useState<string[]>([]);
  const [lmmResult, setLmmResult] = useState<any>(null);

  // Change from baseline
  const [cfbSubject, setCfbSubject] = useState('');
  const [cfbTime, setCfbTime] = useState('');
  const [cfbOutcome, setCfbOutcome] = useState('');
  const [cfbBaseline, setCfbBaseline] = useState('0');
  const [cfbResult, setCfbResult] = useState<any>(null);

  const runLMM = async () => {
    if (!lmmOutcome || !lmmGroup || lmmFixed.length === 0) {
      return addToast('Pick outcome, group (subject ID), and ≥1 fixed effect', 'error');
    }
    setLoading(true);
    try {
      const r = await statsApi.longitudinalLMM({
        cache_key: cacheKey,
        outcome: lmmOutcome,
        fixed_effects: lmmFixed,
        group: lmmGroup,
        random_slopes: lmmRandomSlopes.length ? lmmRandomSlopes : null,
      });
      setLmmResult(r.data);
      setLastPValues(r.data.fixed_effects.map((c: any) => c.p_value));
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'LMM failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runCFB = async () => {
    if (!cfbSubject || !cfbTime || !cfbOutcome) {
      return addToast('Pick subject, time, and outcome', 'error');
    }
    setLoading(true);
    try {
      const baselineVal = isNaN(parseFloat(cfbBaseline)) ? cfbBaseline : parseFloat(cfbBaseline);
      const r = await statsApi.longitudinalChange({
        cache_key: cacheKey,
        subject: cfbSubject,
        time: cfbTime,
        outcome: cfbOutcome,
        baseline_time: baselineVal,
      });
      setCfbResult(r.data);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Change-from-baseline failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Spaghetti data — one series per subject
  const spaghettiData = useMemo(() => {
    if (!cfbResult) return [];
    const bySubject = new Map<string, { time: number; outcome: number }[]>();
    for (const row of cfbResult.per_subject) {
      const sid = String(row[cfbSubject]);
      if (!bySubject.has(sid)) bySubject.set(sid, []);
      bySubject.get(sid)!.push({ time: Number(row[cfbTime]), outcome: Number(row[cfbOutcome]) });
    }
    // Convert to recharts shape: wide-format per time point
    const allTimes = Array.from(new Set(cfbResult.per_subject.map((r: any) => Number(r[cfbTime])))).sort((a: any, b: any) => a - b);
    return (allTimes as number[]).map((t) => {
      const row: Record<string, number | string> = { time: t };
      for (const [sid, series] of bySubject) {
        const point = series.find((p) => p.time === t);
        if (point) row[sid] = point.outcome;
      }
      return row;
    });
  }, [cfbResult, cfbSubject, cfbTime, cfbOutcome]);

  const spaghettiSubjects = useMemo(() => {
    if (!cfbResult) return [] as string[];
    return Array.from(new Set(cfbResult.per_subject.map((r: any) => String(r[cfbSubject])))) as string[];
  }, [cfbResult, cfbSubject]);

  return (
    <div className="space-y-5">
      <div className="flex rounded-lg bg-pie-surface p-1 max-w-md">
        {(['lmm', 'change'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={clsx(
              'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all',
              mode === m ? 'bg-pie-accent text-white' : 'text-pie-text-muted hover:text-pie-text'
            )}
          >
            {m === 'lmm' ? 'Mixed-Effects Model' : 'Change from Baseline'}
          </button>
        ))}
      </div>

      {mode === 'lmm' && (
        <Card>
          <CardHeader>
            <CardTitle>Linear Mixed-Effects Model</CardTitle>
            <CardDescription>
              Model an outcome that's repeatedly measured per subject. A random intercept is fit per group (typically PATNO). Fixed effects are population-level predictors (e.g. visit, cohort, age).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Select label="Outcome" value={lmmOutcome} onChange={(e) => setLmmOutcome(e.target.value)} options={numericOpts} />
              <Select label="Group (subject ID)" value={lmmGroup} onChange={(e) => setLmmGroup(e.target.value)} options={allOpts} />
              <div />
            </div>
            <MultiSelect label="Fixed effects" options={numericMultiOpts} value={lmmFixed} onChange={setLmmFixed} placeholder="Pick fixed-effect predictors…" />
            <MultiSelect label="Random slopes (optional)" options={numericMultiOpts.filter((o) => lmmFixed.includes(o.value))} value={lmmRandomSlopes} onChange={setLmmRandomSlopes} placeholder="Pick predictors to vary per subject…" />
            <Button onClick={runLMM} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Fit
            </Button>
            {lmmResult && (
              <div className="space-y-3">
                <div className="grid grid-cols-4 gap-3">
                  <Stat k="n obs" v={lmmResult.n_obs.toLocaleString()} />
                  <Stat k="n subjects" v={lmmResult.n_groups.toLocaleString()} />
                  <Stat k="AIC" v={sig(lmmResult.aic)} />
                  <Stat k="BIC" v={sig(lmmResult.bic)} />
                </div>
                <div className="p-4 rounded-lg bg-pie-surface overflow-x-auto">
                  <div className="text-sm text-pie-text-muted mb-2">Fixed effects</div>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-pie-border text-pie-text-muted">
                        <th className="text-left py-1.5 px-2">predictor</th>
                        <th className="text-right px-2">β</th>
                        <th className="text-right px-2">SE</th>
                        <th className="text-right px-2">z</th>
                        <th className="text-right px-2">p</th>
                        <th className="text-right px-2">95% CI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {lmmResult.fixed_effects.map((c: any) => (
                        <tr key={c.predictor} className="border-b border-pie-border/50">
                          <td className="py-1.5 px-2 font-mono">{c.predictor}</td>
                          <td className="text-right px-2">{sig(c.estimate)}</td>
                          <td className="text-right px-2">{sig(c.std_error)}</td>
                          <td className="text-right px-2">{sig(c.z_statistic)}</td>
                          <td className="text-right px-2 font-mono">{formatP(c.p_value)}</td>
                          <td className="text-right px-2 font-mono">[{sig(c.ci_lower)}, {sig(c.ci_upper)}]</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="text-xs text-pie-text-muted mt-2">
                    Random-effect variance: <span className="font-mono">{sig(lmmResult.random_effect_variance)}</span> · Residual variance: <span className="font-mono">{sig(lmmResult.residual_variance)}</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {mode === 'change' && (
        <Card>
          <CardHeader>
            <CardTitle>Change from Baseline</CardTitle>
            <CardDescription>
              Per-subject change from baseline visit. Spaghetti plot shows individual trajectories; summary table aggregates by time point.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-4 gap-4">
              <Select label="Subject ID" value={cfbSubject} onChange={(e) => setCfbSubject(e.target.value)} options={allOpts} />
              <Select label="Time" value={cfbTime} onChange={(e) => setCfbTime(e.target.value)} options={allOpts} />
              <Select label="Outcome" value={cfbOutcome} onChange={(e) => setCfbOutcome(e.target.value)} options={numericOpts} />
              <div>
                <label className="block text-sm font-medium text-pie-text mb-2">Baseline value (time)</label>
                <input
                  type="text"
                  value={cfbBaseline}
                  onChange={(e) => setCfbBaseline(e.target.value)}
                  className="w-full px-4 py-2.5 rounded-lg bg-pie-surface border border-pie-border text-pie-text"
                />
              </div>
            </div>
            <Button onClick={runCFB} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Compute
            </Button>
            {cfbResult && (
              <div className="space-y-3">
                <Stat k="n subjects" v={cfbResult.n_subjects.toLocaleString()} />
                <div className="p-4 rounded-lg bg-pie-surface h-72">
                  <div className="text-sm text-pie-text-muted mb-2">Per-subject trajectories</div>
                  <ResponsiveContainer width="100%" height="90%">
                    <LineChart data={spaghettiData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} />
                      <YAxis stroke="#9ca3af" fontSize={11} />
                      <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />
                      {spaghettiSubjects.slice(0, 60).map((sid, i) => (
                        <Line
                          key={sid}
                          type="monotone"
                          dataKey={sid}
                          stroke={`hsl(${(i * 137) % 360}, 60%, 60%)`}
                          strokeWidth={1}
                          strokeOpacity={0.5}
                          dot={false}
                          connectNulls
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                  {spaghettiSubjects.length > 60 && (
                    <div className="text-xs text-pie-text-muted text-center">Showing 60 of {spaghettiSubjects.length} subjects (color-coded for visibility only)</div>
                  )}
                </div>
                <div className="p-4 rounded-lg bg-pie-surface">
                  <div className="text-sm text-pie-text-muted mb-2">Summary by time</div>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-pie-border text-pie-text-muted">
                        <th className="text-left py-1.5 px-2">time</th>
                        <th className="text-right px-2">n</th>
                        <th className="text-right px-2">mean change</th>
                        <th className="text-right px-2">SD</th>
                        <th className="text-right px-2">mean % change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(cfbResult.summary_by_time).map(([t, s]: [string, any]) => (
                        <tr key={t} className="border-b border-pie-border/50">
                          <td className="py-1.5 px-2 font-mono">{t}</td>
                          <td className="text-right px-2">{s.n}</td>
                          <td className="text-right px-2">{sig(s.mean_change)}</td>
                          <td className="text-right px-2">{sig(s.sd_change)}</td>
                          <td className="text-right px-2">{sig(s.mean_pct_change)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
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
