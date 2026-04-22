import { useState, useMemo } from 'react';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import MultiSelect from './shared/MultiSelect';
import { TabProps, sig, formatP } from './shared/types';

type Mode = 'linear' | 'logistic' | 'ancova';

export default function RegressTab({ cacheKey, columns }: TabProps) {
  const { addToast, setLastPValues } = useStore();
  const [mode, setMode] = useState<Mode>('linear');
  const [loading, setLoading] = useState(false);

  const numericOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );
  const numericMultiOpts = useMemo(
    () => columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name })),
    [columns]
  );
  const categoricalOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_categorical || (c.unique_count != null && c.unique_count <= 20)).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );

  // Linear
  const [linOutcome, setLinOutcome] = useState('');
  const [linPredictors, setLinPredictors] = useState<string[]>([]);
  const [linStandardize, setLinStandardize] = useState(false);
  const [linResult, setLinResult] = useState<any>(null);

  // Logistic
  const [logOutcome, setLogOutcome] = useState('');
  const [logPredictors, setLogPredictors] = useState<string[]>([]);
  const [logResult, setLogResult] = useState<any>(null);

  // ANCOVA
  const [anOutcome, setAnOutcome] = useState('');
  const [anGroup, setAnGroup] = useState('');
  const [anCovariates, setAnCovariates] = useState<string[]>([]);
  const [anResult, setAnResult] = useState<any>(null);

  const runLinear = async () => {
    if (!linOutcome || linPredictors.length === 0) return addToast('Pick an outcome and ≥1 predictor', 'error');
    setLoading(true);
    try {
      const r = await statsApi.regressLinear({
        cache_key: cacheKey, outcome: linOutcome, predictors: linPredictors, standardize: linStandardize,
      });
      setLinResult(r.data);
      setLastPValues(r.data.coefficients.map((c: any) => c.p_value));
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Regression failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runLogistic = async () => {
    if (!logOutcome || logPredictors.length === 0) return addToast('Pick an outcome and ≥1 predictor', 'error');
    setLoading(true);
    try {
      const r = await statsApi.regressLogistic({
        cache_key: cacheKey, outcome: logOutcome, predictors: logPredictors,
      });
      setLogResult(r.data);
      setLastPValues(r.data.coefficients.map((c: any) => c.p_value));
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Regression failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runAncova = async () => {
    if (!anOutcome || !anGroup || anCovariates.length === 0) {
      return addToast('Pick outcome, group, and ≥1 covariate', 'error');
    }
    setLoading(true);
    try {
      const r = await statsApi.regressAncova({
        cache_key: cacheKey, outcome: anOutcome, group: anGroup, covariates: anCovariates,
      });
      setAnResult(r.data);
      setLastPValues(r.data.effects.map((e: any) => e.p_value).filter((x: any) => x != null));
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'ANCOVA failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex rounded-lg bg-pie-surface p-1 max-w-lg">
        {(['linear', 'logistic', 'ancova'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={clsx(
              'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all',
              mode === m ? 'bg-pie-accent text-white' : 'text-pie-text-muted hover:text-pie-text'
            )}
          >
            {m === 'linear' ? 'Linear' : m === 'logistic' ? 'Logistic' : 'ANCOVA'}
          </button>
        ))}
      </div>

      {mode === 'linear' && (
        <Card>
          <CardHeader>
            <CardTitle>Linear Regression</CardTitle>
            <CardDescription>OLS with full diagnostics: VIF, Durbin-Watson, residual plot, Q-Q sketch.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Select label="Outcome" value={linOutcome} onChange={(e) => setLinOutcome(e.target.value)} options={numericOpts} />
              <MultiSelect label="Predictors" options={numericMultiOpts} value={linPredictors} onChange={setLinPredictors} placeholder="Pick predictors…" />
            </div>
            <label className="inline-flex items-center gap-2 text-sm text-pie-text">
              <input type="checkbox" checked={linStandardize} onChange={(e) => setLinStandardize(e.target.checked)} className="rounded" />
              Standardize predictors (z-score) to compare effect magnitudes
            </label>
            <Button onClick={runLinear} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Fit
            </Button>
            {linResult && <LinearResult result={linResult} />}
          </CardContent>
        </Card>
      )}

      {mode === 'logistic' && (
        <Card>
          <CardHeader>
            <CardTitle>Logistic Regression</CardTitle>
            <CardDescription>Binary outcome. Reports odds ratios with 95% CIs, pseudo-R², and ROC/AUC on the training fit.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <Select label="Outcome (binary)" value={logOutcome} onChange={(e) => setLogOutcome(e.target.value)} options={categoricalOpts} />
              <MultiSelect label="Predictors" options={numericMultiOpts} value={logPredictors} onChange={setLogPredictors} placeholder="Pick predictors…" />
            </div>
            <Button onClick={runLogistic} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Fit
            </Button>
            {logResult && <LogisticResult result={logResult} />}
          </CardContent>
        </Card>
      )}

      {mode === 'ancova' && (
        <Card>
          <CardHeader>
            <CardTitle>ANCOVA</CardTitle>
            <CardDescription>One-way ANOVA with continuous covariates. Test group differences after adjusting for confounders like age and baseline disease severity.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Select label="Outcome" value={anOutcome} onChange={(e) => setAnOutcome(e.target.value)} options={numericOpts} />
              <Select label="Group" value={anGroup} onChange={(e) => setAnGroup(e.target.value)} options={categoricalOpts} />
              <div />
            </div>
            <MultiSelect label="Covariates" options={numericMultiOpts} value={anCovariates} onChange={setAnCovariates} placeholder="Pick covariates to adjust for…" />
            <Button onClick={runAncova} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Fit
            </Button>
            {anResult && <AncovaResult result={anResult} />}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function LinearResult({ result }: { result: any }) {
  const d = result.diagnostics;
  // Build residual-vs-fitted data
  const rvfData = (d.fitted as number[]).map((f, i) => ({ x: f, y: d.residuals[i] }));
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-3">
        <Stat k="n" v={result.n.toLocaleString()} />
        <Stat k="R²" v={sig(result.r_squared)} />
        <Stat k="adj R²" v={sig(result.adj_r_squared)} />
        <Stat k="F p-value" v={formatP(result.f_p_value)} />
      </div>
      <div className="p-4 rounded-lg bg-pie-surface overflow-x-auto">
        <div className="text-sm text-pie-text-muted mb-2">Coefficients</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-pie-border text-pie-text-muted">
              <th className="text-left py-1.5 px-2">predictor</th>
              <th className="text-right px-2">β</th>
              <th className="text-right px-2">SE</th>
              <th className="text-right px-2">t</th>
              <th className="text-right px-2">p</th>
              <th className="text-right px-2">95% CI</th>
              <th className="text-right px-2">VIF</th>
            </tr>
          </thead>
          <tbody>
            {result.coefficients.map((c: any) => (
              <tr key={c.predictor} className="border-b border-pie-border/50">
                <td className="py-1.5 px-2 font-mono">{c.predictor}</td>
                <td className="text-right px-2">{sig(c.estimate)}</td>
                <td className="text-right px-2">{sig(c.std_error)}</td>
                <td className="text-right px-2">{sig(c.t_statistic)}</td>
                <td className="text-right px-2 font-mono">{formatP(c.p_value)}</td>
                <td className="text-right px-2 font-mono">[{sig(c.ci_lower)}, {sig(c.ci_upper)}]</td>
                <td className={clsx('text-right px-2', d.vif[c.predictor] > 5 && 'text-pie-warning')}>{sig(d.vif[c.predictor])}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="text-xs text-pie-text-muted mt-2">
          Durbin-Watson: <span className="font-mono">{sig(d.durbin_watson)}</span> — values near 2 indicate no autocorrelation. VIF &gt; 5 suggests multicollinearity.
        </div>
      </div>
      <div className="p-4 rounded-lg bg-pie-surface h-64">
        <div className="text-sm text-pie-text-muted mb-2">Residuals vs. fitted</div>
        <ResponsiveContainer width="100%" height="90%">
          <ScatterChart margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis type="number" dataKey="x" name="fitted" stroke="#9ca3af" fontSize={11} />
            <YAxis type="number" dataKey="y" name="residual" stroke="#9ca3af" fontSize={11} />
            <ReferenceLine y={0} stroke="#14b8a6" strokeDasharray="3 3" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />
            <Scatter data={rvfData} fill="#f97316" fillOpacity={0.5} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function LogisticResult({ result }: { result: any }) {
  const roc = result.roc_curve.fpr.map((f: number, i: number) => ({ fpr: f, tpr: result.roc_curve.tpr[i] }));
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-4 gap-3">
        <Stat k="n" v={result.n.toLocaleString()} />
        <Stat k="pseudo-R²" v={sig(result.pseudo_r2)} />
        <Stat k="log-lik" v={sig(result.log_likelihood)} />
        <Stat k="AUC" v={sig(result.auc)} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="p-4 rounded-lg bg-pie-surface overflow-x-auto">
          <div className="text-sm text-pie-text-muted mb-2">Coefficients</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-pie-border text-pie-text-muted">
                <th className="text-left py-1.5 px-2">predictor</th>
                <th className="text-right px-2">β</th>
                <th className="text-right px-2">OR</th>
                <th className="text-right px-2">95% CI (OR)</th>
                <th className="text-right px-2">p</th>
              </tr>
            </thead>
            <tbody>
              {result.coefficients.map((c: any) => (
                <tr key={c.predictor} className="border-b border-pie-border/50">
                  <td className="py-1.5 px-2 font-mono">{c.predictor}</td>
                  <td className="text-right px-2">{sig(c.estimate)}</td>
                  <td className="text-right px-2 font-mono">{sig(c.odds_ratio)}</td>
                  <td className="text-right px-2 font-mono">[{sig(c.or_ci_lower)}, {sig(c.or_ci_upper)}]</td>
                  <td className="text-right px-2 font-mono">{formatP(c.p_value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="p-4 rounded-lg bg-pie-surface h-56">
          <div className="text-sm text-pie-text-muted mb-2">ROC curve (training fit)</div>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={roc} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" dataKey="fpr" name="FPR" domain={[0, 1]} stroke="#9ca3af" fontSize={11} />
              <YAxis type="number" dataKey="tpr" name="TPR" domain={[0, 1]} stroke="#9ca3af" fontSize={11} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#6b7280" strokeDasharray="3 3" />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151' }} />
              <Line type="monotone" dataKey="tpr" stroke="#f97316" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function AncovaResult({ result }: { result: any }) {
  return (
    <div className="p-4 rounded-lg bg-pie-surface">
      <div className="grid grid-cols-3 gap-3 mb-3">
        <Stat k="n" v={result.n.toLocaleString()} />
        <Stat k="R²" v={sig(result.r_squared)} />
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-pie-border text-pie-text-muted">
            <th className="text-left py-1.5 px-2">source</th>
            <th className="text-right px-2">SS</th>
            <th className="text-right px-2">df</th>
            <th className="text-right px-2">F</th>
            <th className="text-right px-2">p</th>
          </tr>
        </thead>
        <tbody>
          {result.effects.map((e: any) => (
            <tr key={e.source} className="border-b border-pie-border/50">
              <td className="py-1.5 px-2 font-mono">{e.source}</td>
              <td className="text-right px-2">{sig(e.sum_sq)}</td>
              <td className="text-right px-2">{e.df}</td>
              <td className="text-right px-2">{sig(e.f_statistic)}</td>
              <td className="text-right px-2 font-mono">{formatP(e.p_value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
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
