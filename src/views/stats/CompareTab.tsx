import { useState, useMemo } from 'react';
import { Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import { TabProps, sig, formatP, interpretCohensD, interpretEtaSquared } from './shared/types';

type Mode = 'two_group' | 'multi_group' | 'categorical';

export default function CompareTab({ cacheKey, columns }: TabProps) {
  const { addToast, setLastPValues } = useStore();
  const [mode, setMode] = useState<Mode>('two_group');

  // Shared variable choices
  const numericOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );
  const categoricalOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_categorical || (c.unique_count != null && c.unique_count <= 20)).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );

  // Two-group
  const [tgVar, setTgVar] = useState('');
  const [tgGroup, setTgGroup] = useState('');
  const [tgTest, setTgTest] = useState('auto');
  const [tgResult, setTgResult] = useState<any>(null);

  // Multi-group
  const [mgVar, setMgVar] = useState('');
  const [mgGroup, setMgGroup] = useState('');
  const [mgTest, setMgTest] = useState('auto');
  const [mgPost, setMgPost] = useState<'tukey' | 'dunn' | 'none'>('tukey');
  const [mgResult, setMgResult] = useState<any>(null);

  // Categorical
  const [cA, setCA] = useState('');
  const [cB, setCB] = useState('');
  const [cTest, setCTest] = useState('auto');
  const [cResult, setCResult] = useState<any>(null);

  const [loading, setLoading] = useState(false);

  const runTwoGroup = async () => {
    if (!tgVar || !tgGroup) return addToast('Pick a variable and a grouping variable', 'error');
    setLoading(true);
    try {
      const r = await statsApi.compareTwoGroup({
        cache_key: cacheKey, variable: tgVar, grouping_variable: tgGroup, test: tgTest,
      });
      setTgResult(r.data);
      setLastPValues([r.data.p_value]);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Test failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runMultiGroup = async () => {
    if (!mgVar || !mgGroup) return addToast('Pick a variable and a grouping variable', 'error');
    setLoading(true);
    try {
      const r = await statsApi.compareMultiGroup({
        cache_key: cacheKey,
        variable: mgVar,
        grouping_variable: mgGroup,
        test: mgTest,
        posthoc: mgPost === 'none' ? null : mgPost,
      });
      setMgResult(r.data);
      const pvals: number[] = [r.data.main.p_value];
      if (r.data.posthoc?.pairwise) for (const p of r.data.posthoc.pairwise) pvals.push(p.p_adj);
      setLastPValues(pvals);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Test failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  const runCategorical = async () => {
    if (!cA || !cB) return addToast('Pick two categorical variables', 'error');
    setLoading(true);
    try {
      const r = await statsApi.compareCategorical({
        cache_key: cacheKey, variable_a: cA, variable_b: cB, test: cTest,
      });
      setCResult(r.data);
      setLastPValues([r.data.p_value]);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'Test failed', 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex rounded-lg bg-pie-surface p-1 max-w-xl">
        {(['two_group', 'multi_group', 'categorical'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={clsx(
              'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all',
              mode === m ? 'bg-pie-accent text-white' : 'text-pie-text-muted hover:text-pie-text'
            )}
          >
            {m === 'two_group' ? 'Two Groups' : m === 'multi_group' ? 'Multiple Groups' : 'Categorical × Categorical'}
          </button>
        ))}
      </div>

      {mode === 'two_group' && (
        <Card>
          <CardHeader>
            <CardTitle>Two-Group Comparison</CardTitle>
            <CardDescription>
              Compare a numeric outcome between two groups. Auto picks Student's t; override for Welch, Mann-Whitney, paired t, or Wilcoxon signed-rank.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Select label="Outcome (numeric)" value={tgVar} onChange={(e) => setTgVar(e.target.value)} options={numericOpts} />
              <Select label="Grouping (2 levels)" value={tgGroup} onChange={(e) => setTgGroup(e.target.value)} options={categoricalOpts} />
              <Select
                label="Test"
                value={tgTest}
                onChange={(e) => setTgTest(e.target.value)}
                options={[
                  { value: 'auto', label: 'Auto (Student\'s t)' },
                  { value: 'independent_t', label: 'Independent t' },
                  { value: 'welch_t', label: 'Welch\'s t' },
                  { value: 'paired_t', label: 'Paired t' },
                  { value: 'mann_whitney', label: 'Mann-Whitney U' },
                  { value: 'wilcoxon', label: 'Wilcoxon signed-rank' },
                ]}
              />
            </div>
            <Button onClick={runTwoGroup} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Run test
            </Button>
            {tgResult && <TwoGroupResult result={tgResult} />}
          </CardContent>
        </Card>
      )}

      {mode === 'multi_group' && (
        <Card>
          <CardHeader>
            <CardTitle>Multi-Group Comparison</CardTitle>
            <CardDescription>One-way ANOVA or Kruskal-Wallis across ≥ 3 groups, with optional Tukey/Dunn post-hoc.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-4 gap-4">
              <Select label="Outcome (numeric)" value={mgVar} onChange={(e) => setMgVar(e.target.value)} options={numericOpts} />
              <Select label="Grouping" value={mgGroup} onChange={(e) => setMgGroup(e.target.value)} options={categoricalOpts} />
              <Select
                label="Test"
                value={mgTest}
                onChange={(e) => setMgTest(e.target.value)}
                options={[
                  { value: 'auto', label: 'Auto (ANOVA)' },
                  { value: 'anova', label: 'One-way ANOVA' },
                  { value: 'kruskal', label: 'Kruskal-Wallis' },
                ]}
              />
              <Select
                label="Post-hoc"
                value={mgPost}
                onChange={(e) => setMgPost(e.target.value as any)}
                options={[
                  { value: 'tukey', label: 'Tukey HSD' },
                  { value: 'dunn', label: 'Dunn (Bonferroni)' },
                  { value: 'none', label: 'None' },
                ]}
              />
            </div>
            <Button onClick={runMultiGroup} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Run test
            </Button>
            {mgResult && <MultiGroupResult result={mgResult} />}
          </CardContent>
        </Card>
      )}

      {mode === 'categorical' && (
        <Card>
          <CardHeader>
            <CardTitle>Categorical × Categorical</CardTitle>
            <CardDescription>Chi-square test of independence; auto switches to Fisher's exact when any expected cell count drops below 5 in a 2×2 table.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Select label="Variable A" value={cA} onChange={(e) => setCA(e.target.value)} options={categoricalOpts} />
              <Select label="Variable B" value={cB} onChange={(e) => setCB(e.target.value)} options={categoricalOpts} />
              <Select
                label="Test"
                value={cTest}
                onChange={(e) => setCTest(e.target.value)}
                options={[
                  { value: 'auto', label: 'Auto' },
                  { value: 'chi_square', label: 'Chi-square' },
                  { value: 'fisher', label: "Fisher's exact (2×2)" },
                ]}
              />
            </div>
            <Button onClick={runCategorical} disabled={loading}>
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Run test
            </Button>
            {cResult && <CategoricalResult result={cResult} />}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function TwoGroupResult({ result }: { result: any }) {
  const effect = result.cohens_d != null ? interpretCohensD(result.cohens_d) : null;
  return (
    <div className="p-4 rounded-lg bg-pie-surface space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-pie-text-muted">Test</span>
        <span className="font-medium text-pie-text">{result.test}</span>
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <Row k="statistic" v={sig(result.statistic)} />
        <Row k="df" v={result.df} />
        <Row k="p-value" v={formatP(result.p_value)} />
        <Row k="n" v={`${result.n1 ?? result.n_pairs ?? '?'} / ${result.n2 ?? '—'}`} />
        {result.mean1 != null && <Row k="mean (group 1)" v={sig(result.mean1)} />}
        {result.mean2 != null && <Row k="mean (group 2)" v={sig(result.mean2)} />}
        {result.median1 != null && <Row k="median (group 1)" v={sig(result.median1)} />}
        {result.median2 != null && <Row k="median (group 2)" v={sig(result.median2)} />}
        {result.mean_diff != null && <Row k="mean diff" v={sig(result.mean_diff)} />}
        {result.cohens_d != null && (
          <Row k="Cohen's d" v={<span><span className="font-mono">{sig(result.cohens_d)}</span> {effect && <span className={clsx('ml-2 text-xs', effect.color)}>({effect.label})</span>}</span>} />
        )}
        {result.hedges_g != null && <Row k="Hedges' g" v={sig(result.hedges_g)} />}
      </div>
    </div>
  );
}

function MultiGroupResult({ result }: { result: any }) {
  const m = result.main;
  const effect = m.eta_squared != null ? interpretEtaSquared(m.eta_squared) : null;
  return (
    <div className="space-y-3">
      <div className="p-4 rounded-lg bg-pie-surface space-y-2">
        <div className="text-sm text-pie-text-muted">Main test</div>
        <div className="font-medium text-pie-text mb-2">{m.test}</div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <Row k="statistic" v={sig(m.statistic)} />
          <Row k="df" v={m.df != null ? m.df : `${m.df_between}, ${m.df_within}`} />
          <Row k="p-value" v={formatP(m.p_value)} />
          {m.eta_squared != null && (
            <Row k="η²" v={<span><span className="font-mono">{sig(m.eta_squared)}</span> {effect && <span className={clsx('ml-2 text-xs', effect.color)}>({effect.label})</span>}</span>} />
          )}
        </div>
      </div>
      {result.posthoc && (
        <div className="p-4 rounded-lg bg-pie-surface">
          <div className="text-sm text-pie-text-muted mb-2">Post-hoc: {result.posthoc.method}</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-pie-text-muted border-b border-pie-border">
                <th className="text-left py-1.5 px-2">group 1</th>
                <th className="text-left px-2">group 2</th>
                {result.posthoc.pairwise[0]?.mean_diff != null && <th className="text-right px-2">mean diff</th>}
                <th className="text-right px-2">p (adj)</th>
                <th className="text-right px-2">reject?</th>
              </tr>
            </thead>
            <tbody>
              {result.posthoc.pairwise.map((p: any, i: number) => (
                <tr key={i} className="border-b border-pie-border/50">
                  <td className="py-1 px-2 font-mono">{p.group1}</td>
                  <td className="px-2 font-mono">{p.group2}</td>
                  {p.mean_diff != null && <td className="text-right px-2">{sig(p.mean_diff)}</td>}
                  <td className="text-right px-2 font-mono">{formatP(p.p_adj)}</td>
                  <td className={clsx('text-right px-2', p.reject && 'text-pie-accent font-medium')}>
                    {p.reject ? '✓' : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function CategoricalResult({ result }: { result: any }) {
  return (
    <div className="space-y-3">
      <div className="p-4 rounded-lg bg-pie-surface space-y-2">
        <div className="text-sm text-pie-text-muted">Test</div>
        <div className="font-medium text-pie-text mb-2">{result.test}</div>
        <div className="grid grid-cols-2 gap-3 text-sm">
          {result.statistic != null && <Row k="statistic" v={sig(result.statistic)} />}
          {result.dof != null && <Row k="df" v={result.dof} />}
          {result.odds_ratio != null && <Row k="odds ratio" v={sig(result.odds_ratio)} />}
          <Row k="p-value" v={formatP(result.p_value)} />
        </div>
      </div>
      <div className="p-4 rounded-lg bg-pie-surface">
        <div className="text-sm text-pie-text-muted mb-2">Contingency table</div>
        <div className="overflow-x-auto">
          <table className="text-sm">
            <thead>
              <tr>
                <th></th>
                {result.col_labels.map((c: string) => (
                  <th key={c} className="px-3 py-1.5 font-mono text-pie-text-muted">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.contingency.map((row: number[], i: number) => (
                <tr key={i}>
                  <th className="px-3 py-1 font-mono text-pie-text-muted">{result.row_labels[i]}</th>
                  {row.map((v, j) => (
                    <td key={j} className="px-3 py-1 text-right font-mono">{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: any }) {
  return (
    <div className="flex justify-between">
      <span className="text-pie-text-muted">{k}</span>
      <span className="font-mono text-pie-text">{v}</span>
    </div>
  );
}
