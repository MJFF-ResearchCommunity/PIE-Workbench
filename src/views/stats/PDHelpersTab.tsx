import { useState, useEffect, useMemo } from 'react';
import { Loader2, Plus, X, Info } from 'lucide-react';
import { clsx } from 'clsx';
import Button from '../../components/ui/Button';
import Select from '../../components/ui/Select';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/Card';
import { statsApi } from '../../services/api';
import { useStore } from '../../store/useStore';
import MultiSelect from './shared/MultiSelect';
import { TabProps, sig } from './shared/types';

type DrugRow = { id: string; drug: string; dose: string };

export default function PDHelpersTab({ cacheKey, columns }: TabProps) {
  const { addToast } = useStore();

  // LEDD
  const [factors, setFactors] = useState<Record<string, number>>({});
  const [rows, setRows] = useState<DrugRow[]>([{ id: 'r1', drug: 'levodopa_ir', dose: '300' }]);
  const [leddResult, setLeddResult] = useState<any>(null);
  const [leddLoading, setLeddLoading] = useState(false);

  // UPDRS
  const [part1, setPart1] = useState<string[]>([]);
  const [part2, setPart2] = useState<string[]>([]);
  const [part3, setPart3] = useState<string[]>([]);
  const [part4, setPart4] = useState<string[]>([]);
  const [updrsResult, setUpdrsResult] = useState<any>(null);
  const [updrsLoading, setUpdrsLoading] = useState(false);

  // H&Y
  const [hyVar, setHyVar] = useState('');
  const [hyResult, setHyResult] = useState<any>(null);
  const [hyLoading, setHyLoading] = useState(false);

  useEffect(() => {
    statsApi.pdLeddFactors().then((r) => setFactors(r.data.factors)).catch(() => undefined);
  }, []);

  const numericMultiOpts = useMemo(
    () => columns.filter((c) => c.is_numeric).map((c) => ({ value: c.name, label: c.name })),
    [columns]
  );
  const numericOpts = useMemo(
    () => [{ value: '', label: '— pick —' }, ...columns.filter((c) => c.is_numeric || (c.unique_count != null && c.unique_count <= 10)).map((c) => ({ value: c.name, label: c.name }))],
    [columns]
  );

  const runLEDD = async () => {
    const doses: Record<string, number> = {};
    for (const r of rows) {
      const n = parseFloat(r.dose);
      if (!isNaN(n) && n > 0 && r.drug) {
        doses[r.drug] = (doses[r.drug] || 0) + n;
      }
    }
    if (Object.keys(doses).length === 0) return addToast('Add at least one drug dose', 'error');
    setLeddLoading(true);
    try {
      const r = await statsApi.pdLedd(doses);
      setLeddResult(r.data);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'LEDD computation failed', 'error');
    } finally {
      setLeddLoading(false);
    }
  };

  const runUPDRS = async () => {
    if (![part1, part2, part3, part4].some((p) => p.length > 0)) {
      return addToast('Assign at least one column to a UPDRS part', 'error');
    }
    setUpdrsLoading(true);
    try {
      const r = await statsApi.pdUpdrs({
        cache_key: cacheKey,
        part1_cols: part1.length ? part1 : undefined,
        part2_cols: part2.length ? part2 : undefined,
        part3_cols: part3.length ? part3 : undefined,
        part4_cols: part4.length ? part4 : undefined,
      });
      setUpdrsResult(r.data);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'UPDRS aggregation failed', 'error');
    } finally {
      setUpdrsLoading(false);
    }
  };

  const runHY = async () => {
    if (!hyVar) return addToast('Pick an H&Y variable', 'error');
    setHyLoading(true);
    try {
      const r = await statsApi.pdHoehnYahr({ cache_key: cacheKey, variable: hyVar });
      setHyResult(r.data);
    } catch (e: any) {
      addToast(e?.response?.data?.detail || 'H&Y summary failed', 'error');
    } finally {
      setHyLoading(false);
    }
  };

  const drugOptions = useMemo(
    () => Object.keys(factors).sort().map((k) => ({ value: k, label: `${k} (×${factors[k]})` })),
    [factors]
  );

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader>
          <CardTitle>LEDD Calculator</CardTitle>
          <CardDescription>
            Convert per-drug daily doses into levodopa-equivalent daily dose (LEDD). Uses the Tomlinson et al. 2010 conversion factors.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {rows.map((row, i) => (
            <div key={row.id} className="grid grid-cols-[1fr_140px_40px] gap-2 items-end">
              <Select
                label={i === 0 ? 'Drug' : undefined}
                value={row.drug}
                onChange={(e) =>
                  setRows((prev) => prev.map((r) => (r.id === row.id ? { ...r, drug: e.target.value } : r)))
                }
                options={drugOptions}
              />
              <div>
                {i === 0 && <label className="block text-sm font-medium text-pie-text mb-2">Dose (mg/day)</label>}
                <input
                  type="number"
                  value={row.dose}
                  onChange={(e) => setRows((prev) => prev.map((r) => (r.id === row.id ? { ...r, dose: e.target.value } : r)))}
                  className="w-full px-3 py-2.5 rounded-lg bg-pie-surface border border-pie-border text-pie-text font-mono"
                />
              </div>
              <button
                onClick={() => setRows((prev) => prev.filter((r) => r.id !== row.id))}
                className="h-[42px] flex items-center justify-center text-pie-text-muted hover:text-pie-warning"
                disabled={rows.length === 1}
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
          <div className="flex gap-2">
            <Button
              variant="secondary"
              onClick={() => setRows((prev) => [...prev, { id: `r${Date.now()}`, drug: 'pramipexole', dose: '' }])}
            >
              <Plus className="w-4 h-4" /> Add drug
            </Button>
            <Button onClick={runLEDD} disabled={leddLoading}>
              {leddLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              Compute LEDD
            </Button>
          </div>

          {leddResult && (
            <div className="p-4 rounded-lg bg-pie-surface">
              <div className="text-sm text-pie-text-muted">Total LEDD</div>
              <div className="text-3xl font-bold text-pie-text mb-3">{sig(leddResult.total_ledd_mg)} mg/day</div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-pie-border text-pie-text-muted">
                    <th className="text-left py-1.5 px-2">drug</th>
                    <th className="text-right px-2">dose (mg)</th>
                    <th className="text-right px-2">factor</th>
                    <th className="text-right px-2">LEDD (mg)</th>
                    <th className="text-left px-2">note</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(leddResult.per_drug).map(([drug, info]: [string, any]) => (
                    <tr key={drug} className="border-b border-pie-border/50">
                      <td className="py-1.5 px-2 font-mono">{drug}</td>
                      <td className="text-right px-2 font-mono">{info.dose_mg}</td>
                      <td className="text-right px-2 font-mono">{info.factor ?? '—'}</td>
                      <td className="text-right px-2 font-mono">{info.ledd_mg != null ? sig(info.ledd_mg) : '—'}</td>
                      <td className={clsx('px-2 text-xs', info.note && 'text-pie-warning')}>{info.note || ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-500/10 text-blue-400 text-xs">
            <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>
              Tolcapone / entacapone factors apply to the daily levodopa dose (COMT inhibitors boost levodopa bioavailability). Pass those alongside a levodopa entry to reflect the combined LEDD.
            </span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>MDS-UPDRS Aggregator</CardTitle>
          <CardDescription>Pick the columns that form each MDS-UPDRS part; we compute per-row part totals and an overall total.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <MultiSelect label="Part I columns (non-motor)" options={numericMultiOpts} value={part1} onChange={setPart1} />
            <MultiSelect label="Part II columns (motor aspects of daily life)" options={numericMultiOpts} value={part2} onChange={setPart2} />
            <MultiSelect label="Part III columns (motor exam)" options={numericMultiOpts} value={part3} onChange={setPart3} />
            <MultiSelect label="Part IV columns (complications)" options={numericMultiOpts} value={part4} onChange={setPart4} />
          </div>
          <Button onClick={runUPDRS} disabled={updrsLoading}>
            {updrsLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
            Compute
          </Button>
          {updrsResult && (
            <div className="space-y-3">
              <div className="p-3 rounded bg-pie-surface text-sm">
                {updrsResult.n_rows.toLocaleString()} rows · {updrsResult.columns.join(', ')}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-pie-border text-pie-text-muted">
                      {updrsResult.columns.map((c: string) => (
                        <th key={c} className="text-right px-2 py-1.5">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {updrsResult.preview.map((row: any, i: number) => (
                      <tr key={i} className="border-b border-pie-border/50">
                        {updrsResult.columns.map((c: string) => (
                          <td key={c} className="text-right px-2 py-1 font-mono">{row[c] ?? '—'}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="text-xs text-pie-text-muted mt-1">Showing first 20 of {updrsResult.n_rows}.</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Hoehn &amp; Yahr Staging Summary</CardTitle>
          <CardDescription>Distribution of H&amp;Y stages with counts, proportions, and median stage.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <Select label="H&Y variable" value={hyVar} onChange={(e) => setHyVar(e.target.value)} options={numericOpts} />
            <div className="flex items-end col-span-2">
              <Button onClick={runHY} disabled={hyLoading}>
                {hyLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                Summarize
              </Button>
            </div>
          </div>
          {hyResult && (
            <div className="p-4 rounded-lg bg-pie-surface">
              <div className="grid grid-cols-3 gap-3 mb-3">
                <Stat k="n" v={hyResult.n.toLocaleString()} />
                <Stat k="mean stage" v={sig(hyResult.mean_stage)} />
                <Stat k="median stage" v={sig(hyResult.median_stage)} />
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-pie-border text-pie-text-muted">
                    <th className="text-left py-1.5 px-2">stage</th>
                    <th className="text-right px-2">count</th>
                    <th className="text-right px-2">proportion</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(hyResult.counts).map(([stage, count]: [string, any]) => (
                    <tr key={stage} className="border-b border-pie-border/50">
                      <td className="py-1.5 px-2 font-mono">{stage}</td>
                      <td className="text-right px-2 font-mono">{count}</td>
                      <td className="text-right px-2 font-mono">{(hyResult.proportions[stage] * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
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
