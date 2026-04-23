import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Trophy,
  Download,
  BarChart2,
  Target,
  Layers,
  ExternalLink,
  Share2,
  RefreshCw,
  Loader2,
  GitBranch,
  Network,
  Code2,
  ChevronDown,
  ChevronRight,
  Info,
} from 'lucide-react';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import Button from '../components/ui/Button';
import { useStore } from '../store/useStore';
import { analysisApi } from '../services/api';
import { clsx } from 'clsx';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

interface FeatureImportance {
  features: string[];
  importances: number[];
}

interface ConfusionCell {
  actual: string;
  predicted: string;
  value: number;
}

interface ModelResults {
  best_model_name: string;
  metrics: Record<string, number | null>;
  confusion_matrix: ConfusionCell[];
  class_labels: string[];
  comparison: Record<string, Record<string, number | string>> | null;
}

interface ModelStructure {
  supported: boolean;
  structure: Record<string, unknown> | null;
  model_type: string;
  n_features: number;
  tree_viz_supported: boolean;
  n_trees: number;
  bn_viz_supported: boolean;
  reason?: string;
}

const METRIC_COLORS: Record<string, string> = {
  'Accuracy': '#ff6b4a',
  'AUC': '#4ecdc4',
  'Prec.': '#fbbf24',
  'Recall': '#4ade80',
  'F1': '#a78bfa',
  'MCC': '#f472b6',
  'Kappa': '#38bdf8',
};

const METRIC_LABELS: Record<string, string> = {
  'Prec.': 'Precision',
  'F1': 'F1 Score',
};

// Walks an endgame tree payload and returns basic shape stats.  Child node
// keys vary across tree types — Oblique uses left/right, C5.0 uses branches,
// sklearn-derived trees use children — so this does a structural walk that
// recurses into any nested dicts, using the node's own 'depth' field when
// present for accuracy.
function walkTree(root: unknown): { nodes: number; leaves: number; maxDepth: number } {
  let nodes = 0;
  let leaves = 0;
  let maxDepth = 0;
  const nodeKeyHints = new Set(['type', 'depth', 'n_samples', 'impurity', 'feature', 'threshold', 'class']);
  const isNodeLike = (n: unknown): n is Record<string, unknown> => {
    if (!n || typeof n !== 'object' || Array.isArray(n)) return false;
    const keys = Object.keys(n as object);
    return keys.some((k) => nodeKeyHints.has(k));
  };
  // Collect every nested node-like dict under `n`, descending through
  // non-node wrappers (e.g. C5.0's `branches:[{edge,child}]`) until a
  // node-like dict is found.
  const collectChildNodes = (n: unknown, out: Record<string, unknown>[]) => {
    if (n == null || typeof n !== 'object') return;
    if (Array.isArray(n)) {
      for (const v of n) collectChildNodes(v, out);
      return;
    }
    if (isNodeLike(n)) {
      out.push(n);
      return;
    }
    for (const v of Object.values(n)) collectChildNodes(v, out);
  };
  const visit = (n: Record<string, unknown>, depth: number) => {
    nodes += 1;
    const nodeDepth = typeof n.depth === 'number' ? (n.depth as number) : depth;
    if (nodeDepth > maxDepth) maxDepth = nodeDepth;
    const childNodes: Record<string, unknown>[] = [];
    for (const v of Object.values(n)) collectChildNodes(v, childNodes);
    if (childNodes.length === 0 || n.type === 'leaf' || n.is_leaf === true) {
      leaves += 1;
    }
    for (const c of childNodes) visit(c, nodeDepth + 1);
  };
  if (isNodeLike(root)) visit(root, 0);
  return { nodes, leaves, maxDepth };
}

// --- Structure renderers ---------------------------------------------------
// One per structure_type.  Each accepts the raw `structure` dict and the
// model's classes (for multi-class payloads).  Unknown/empty types fall
// through to GenericStructureView which shows a compact key/value grid and
// defers the detailed payload to the collapsed raw-JSON view.

type StructureDict = Record<string, unknown>;

function formatNumber(n: number, sig: number = 4): string {
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '0';
  const abs = Math.abs(n);
  if (abs >= 0.01 && abs < 1000) return n.toFixed(sig >= 4 ? 4 : sig);
  return n.toExponential(2);
}

function truncateLabel(name: string, max: number = 28): string {
  return name.length > max ? name.slice(0, max - 1) + '…' : name;
}

function AdditiveStructureView({ structure }: { structure: StructureDict }) {
  // EBM / MARS — `terms: [{name, type:'main'|'interaction', importance, ...}]`
  const rawTerms = (structure.terms as Array<Record<string, unknown>>) || [];
  const intercept = structure.intercept as number[] | number | undefined;
  const classes = (structure.classes as unknown[] | undefined)?.map(String);

  const terms = useMemo(() => {
    return rawTerms
      .map((t) => ({
        name: String(t.name ?? ''),
        type: String(t.type ?? 'main'),
        importance: Number(t.importance ?? 0),
      }))
      .filter((t) => Number.isFinite(t.importance))
      .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));
  }, [rawTerms]);

  const topTerms = terms.slice(0, 15);
  const chartData = topTerms.map((t) => ({
    name: truncateLabel(t.name),
    fullName: t.name,
    importance: t.importance,
    type: t.type,
  })).reverse();  // recharts horizontal layout flips order

  return (
    <div className="space-y-4">
      <div>
        <div className="text-xs uppercase tracking-wide text-pie-text-muted mb-2">
          Top terms by importance ({terms.length} total)
        </div>
        <div className="h-80 rounded-lg border border-pie-border bg-pie-surface/40 p-3">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 160, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
              <XAxis type="number" stroke="#8b9dc3" fontSize={11} />
              <YAxis type="category" dataKey="name" stroke="#8b9dc3" fontSize={11} width={150} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1a2540', border: '1px solid #2a3a5c', borderRadius: 8 }}
                labelStyle={{ color: '#e8eff8' }}
                formatter={(v: number, _n, p: { payload?: { fullName?: string; type?: string } }) => [
                  formatNumber(v), `${p.payload?.fullName || ''} (${p.payload?.type || ''})`,
                ]}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {chartData.map((t, i) => (
                  <Cell key={i} fill={t.type === 'interaction' ? '#a78bfa' : '#4ecdc4'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex items-center gap-4 text-[11px] text-pie-text-muted">
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-[#4ecdc4]" /> main effect</span>
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-[#a78bfa]" /> interaction</span>
        </div>
      </div>

      {intercept != null && (
        <div>
          <div className="text-xs uppercase tracking-wide text-pie-text-muted mb-2">Intercept</div>
          <div className="flex flex-wrap gap-2">
            {(Array.isArray(intercept) ? intercept : [intercept]).map((v, i) => (
              <div key={i} className="px-3 py-1.5 rounded border border-pie-border bg-pie-surface font-mono text-xs">
                {classes && classes[i] ? <span className="text-pie-text-muted mr-2">{classes[i]}</span> : null}
                <span className="text-pie-text">{formatNumber(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function LinearStructureView({ structure }: { structure: StructureDict }) {
  // LinearClassifier / LogisticRegression / LDA / ordinal — `coefficients`
  // is `list[dict[feature→weight]]` for multi-class or a single dict for
  // binary / regression.  Show top-|β| coefficients per class (switcher).
  const coefsRaw = structure.coefficients as unknown;
  const intercept = structure.intercept as number[] | number | undefined;
  const classes = (structure.classes as unknown[] | undefined)?.map(String);
  const featureNames = (structure.feature_names as string[] | undefined) || [];

  const coefPerClass = useMemo(() => {
    // Normalize into list-of-dicts so we only have one render path below.
    if (Array.isArray(coefsRaw)) return coefsRaw as Array<Record<string, number>>;
    if (coefsRaw && typeof coefsRaw === 'object') return [coefsRaw as Record<string, number>];
    return [];
  }, [coefsRaw]);

  const [classIdx, setClassIdx] = useState(0);
  const classIdxSafe = Math.min(classIdx, Math.max(0, coefPerClass.length - 1));

  const topCoefs = useMemo(() => {
    const entries = Object.entries(coefPerClass[classIdxSafe] || {});
    return entries
      .map(([name, value]) => {
        // LinearClassifier returns generic 'x0', 'x1' keys — map back to
        // actual feature names when the index is recoverable.
        const m = name.match(/^x(\d+)$/);
        const display = m && featureNames[Number(m[1])] ? featureNames[Number(m[1])] : name;
        return { name: display, value: Number(value) };
      })
      .filter((c) => Number.isFinite(c.value))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 15);
  }, [coefPerClass, classIdxSafe, featureNames]);

  const chartData = topCoefs.map((c) => ({
    name: truncateLabel(c.name),
    fullName: c.name,
    value: c.value,
  })).reverse();

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-pie-text-muted">
          Top coefficients by |β|
        </div>
        {coefPerClass.length > 1 && (
          <div className="flex items-center gap-1 text-xs">
            <span className="text-pie-text-muted">Class:</span>
            {coefPerClass.map((_, i) => {
              const label = classes?.[i] ?? `class ${i}`;
              return (
                <button
                  key={i}
                  onClick={() => setClassIdx(i)}
                  className={clsx(
                    'px-2 py-0.5 rounded font-mono',
                    i === classIdxSafe ? 'bg-pie-accent text-white' : 'bg-pie-surface text-pie-text-muted hover:text-pie-text',
                  )}
                >
                  {label}
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div className="h-80 rounded-lg border border-pie-border bg-pie-surface/40 p-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 160, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
            <XAxis type="number" stroke="#8b9dc3" fontSize={11} />
            <YAxis type="category" dataKey="name" stroke="#8b9dc3" fontSize={11} width={150} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a2540', border: '1px solid #2a3a5c', borderRadius: 8 }}
              labelStyle={{ color: '#e8eff8' }}
              formatter={(v: number, _n, p: { payload?: { fullName?: string } }) => [
                formatNumber(v), p.payload?.fullName || '',
              ]}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((c, i) => (
                <Cell key={i} fill={c.value >= 0 ? '#4ade80' : '#f87171'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {intercept != null && (
        <div>
          <div className="text-xs uppercase tracking-wide text-pie-text-muted mb-2">Intercept</div>
          <div className="flex flex-wrap gap-2">
            {(Array.isArray(intercept) ? intercept : [intercept]).map((v, i) => (
              <div key={i} className="px-3 py-1.5 rounded border border-pie-border bg-pie-surface font-mono text-xs">
                {classes && classes[i] ? <span className="text-pie-text-muted mr-2">{classes[i]}</span> : null}
                <span className="text-pie-text">{formatNumber(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RulesStructureView({ structure }: { structure: StructureDict }) {
  // RuleFit / FURIA / CORELS — `rules: [{condition|antecedent, consequent|class, support?, coef?, ...}]`
  const rules = (structure.rules as Array<Record<string, unknown>>) || [];
  if (rules.length === 0) return <GenericStructureView structure={structure} />;

  return (
    <div className="space-y-2">
      <div className="text-xs uppercase tracking-wide text-pie-text-muted">
        {rules.length} rule{rules.length === 1 ? '' : 's'}
      </div>
      <div className="space-y-1 max-h-96 overflow-auto rounded-lg border border-pie-border bg-pie-surface/40 p-2">
        {rules.slice(0, 200).map((r, i) => {
          const antecedent = String(r.condition ?? r.antecedent ?? r.rule ?? '');
          const consequent = String(r.consequent ?? r.class ?? r.prediction ?? '');
          const coef = typeof r.coef === 'number' ? r.coef : (typeof r.coefficient === 'number' ? r.coefficient : null);
          const support = typeof r.support === 'number' ? r.support : null;
          return (
            <div key={i} className="text-xs font-mono text-pie-text flex items-start gap-2 py-1 px-1.5 hover:bg-pie-surface rounded">
              <span className="text-pie-text-muted select-none w-6 flex-shrink-0">{i + 1}.</span>
              <div className="flex-1">
                <span className="text-pie-accent-secondary">IF</span> {antecedent}
                {consequent && (
                  <>
                    {' '}<span className="text-pie-accent-secondary">THEN</span> {consequent}
                  </>
                )}
              </div>
              {coef != null && (
                <span className="text-pie-text-muted flex-shrink-0">β={formatNumber(coef)}</span>
              )}
              {support != null && (
                <span className="text-pie-text-muted flex-shrink-0">n={support}</span>
              )}
            </div>
          );
        })}
        {rules.length > 200 && (
          <div className="text-[11px] text-pie-text-muted text-center py-1">
            … {rules.length - 200} more rules (see raw JSON)
          </div>
        )}
      </div>
    </div>
  );
}

function BayesianNetworkView({ structure }: { structure: StructureDict }) {
  const nodes = (structure.nodes as Array<Record<string, unknown>>) || [];
  const edges = (structure.edges as Array<Record<string, unknown>>) || [];
  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <div className="px-3 py-2 rounded border border-pie-border bg-pie-surface">
          <div className="text-[11px] uppercase text-pie-text-muted">Nodes</div>
          <div className="text-sm font-mono">{nodes.length}</div>
        </div>
        <div className="px-3 py-2 rounded border border-pie-border bg-pie-surface">
          <div className="text-[11px] uppercase text-pie-text-muted">Edges</div>
          <div className="text-sm font-mono">{edges.length}</div>
        </div>
      </div>
      {edges.length > 0 && (
        <div className="max-h-72 overflow-auto rounded-lg border border-pie-border bg-pie-surface/40 p-2">
          <div className="text-xs uppercase tracking-wide text-pie-text-muted mb-2 px-1">
            Dependency edges
          </div>
          {edges.slice(0, 200).map((e, i) => (
            <div key={i} className="text-xs font-mono text-pie-text py-0.5 px-1">
              <span className="text-pie-accent">{String(e.parent ?? e.from ?? '?')}</span>
              <span className="text-pie-text-muted mx-2">→</span>
              <span>{String(e.child ?? e.to ?? '?')}</span>
            </div>
          ))}
          {edges.length > 200 && (
            <div className="text-[11px] text-pie-text-muted text-center py-1">
              … {edges.length - 200} more edges
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function BoxesStructureView({ structure }: { structure: StructureDict }) {
  const boxes = (structure.boxes as Array<Record<string, unknown>>) || [];
  const perClass = structure.per_class as Record<string, unknown> | undefined;

  if (boxes.length === 0 && perClass) {
    return (
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-wide text-pie-text-muted">
          Per-class boxes
        </div>
        <pre className="text-[11px] text-pie-text bg-pie-surface/40 border border-pie-border rounded p-2 overflow-auto max-h-72">
          {JSON.stringify(perClass, null, 2)}
        </pre>
      </div>
    );
  }
  if (boxes.length === 0) return <GenericStructureView structure={structure} />;

  return (
    <div className="space-y-2 max-h-96 overflow-auto">
      {boxes.map((b, i) => (
        <div key={i} className="rounded-lg border border-pie-border bg-pie-surface/40 p-3">
          <div className="text-xs font-semibold text-pie-accent-secondary mb-1">Box {i + 1}</div>
          <pre className="text-[11px] text-pie-text font-mono">
            {JSON.stringify(b, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  );
}

function GenericStructureView({ structure }: { structure: StructureDict }) {
  // Fallback: two-column key/value grid for scalar entries; arrays and
  // nested objects are summarized as their shape to avoid wall-of-text.
  const interesting = Object.entries(structure).filter(
    ([k]) => !['model_type', 'structure_type', 'n_features', 'n_classes', 'feature_names', 'classes'].includes(k),
  );
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
      {interesting.map(([k, v]) => {
        let display: string;
        if (v == null) display = '—';
        else if (Array.isArray(v)) display = `list[${v.length}]`;
        else if (typeof v === 'object') display = `object(${Object.keys(v).length} keys)`;
        else if (typeof v === 'number') display = formatNumber(v);
        else display = String(v);
        return (
          <div key={k} className="px-3 py-2 rounded border border-pie-border bg-pie-surface">
            <div className="text-[11px] uppercase tracking-wide text-pie-text-muted truncate">{k}</div>
            <div className="text-sm font-mono text-pie-text mt-0.5 truncate" title={display}>{display}</div>
          </div>
        );
      })}
    </div>
  );
}

function StructureDetailView({ structure }: { structure: StructureDict }) {
  const st = String(structure.structure_type || 'generic');
  switch (st) {
    case 'additive':         return <AdditiveStructureView structure={structure} />;
    case 'linear':           return <LinearStructureView structure={structure} />;
    case 'rules':
    case 'fuzzy_rules':      return <RulesStructureView structure={structure} />;
    case 'bayesian_network': return <BayesianNetworkView structure={structure} />;
    case 'boxes':            return <BoxesStructureView structure={structure} />;
    // tree / tree_ensemble: the Open Tree Visualizer button handles the
    // interactive view; the summary grid above already shows nodes/depth.
    // Anything else: fall through to the compact key/value grid.
    default:                 return <GenericStructureView structure={structure} />;
  }
}

export default function Results() {
  const navigate = useNavigate();
  const { project, analysis, addToast } = useStore();

  const [featureImportance, setFeatureImportance] = useState<FeatureImportance | null>(null);
  const [modelResults, setModelResults] = useState<ModelResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [resultsLoading, setResultsLoading] = useState(true);
  const [reportLoading, setReportLoading] = useState(false);

  const [structure, setStructure] = useState<ModelStructure | null>(null);
  const [structureLoading, setStructureLoading] = useState(false);
  const [treeVizLoading, setTreeVizLoading] = useState(false);
  const [bnVizLoading, setBnVizLoading] = useState(false);
  const [treeIndex, setTreeIndex] = useState(0);
  const [showRawStructure, setShowRawStructure] = useState(false);

  useEffect(() => {
    if (!project || !analysis.modelId) {
      navigate('/ml');
      return;
    }
    loadFeatureImportance();
    loadModelResults();
    loadStructure();
  }, [project, analysis.modelId, navigate]);

  const loadFeatureImportance = async () => {
    if (!analysis.modelId) return;

    setLoading(true);
    try {
      const response = await analysisApi.getFeatureImportance(analysis.modelId, 15);
      if (!response.data.error) {
        setFeatureImportance(response.data);
      }
    } catch (error) {
      console.error('Failed to load feature importance:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadModelResults = async () => {
    if (!analysis.modelId) return;

    setResultsLoading(true);
    try {
      const response = await analysisApi.getModelResults(analysis.modelId);
      setModelResults(response.data);
    } catch (error) {
      console.error('Failed to load model results:', error);
      addToast('Failed to load model results', 'error');
    } finally {
      setResultsLoading(false);
    }
  };

  const loadStructure = async () => {
    if (!analysis.modelId) return;
    setStructureLoading(true);
    try {
      const response = await analysisApi.getModelStructure(analysis.modelId);
      setStructure(response.data);
    } catch (error) {
      console.error('Failed to load model structure:', error);
    } finally {
      setStructureLoading(false);
    }
  };

  const handleOpenTreeViz = async () => {
    if (!analysis.modelId) return;
    setTreeVizLoading(true);
    try {
      const response = await analysisApi.getTreeViz(analysis.modelId, treeIndex);
      const html: string = response.data;

      // Same handoff pattern as the full report: route through Electron when
      // available so the viz opens in the user's default browser; fall back
      // to blob URL + window.open in a plain browser.
      if (window.electronAPI?.openReportHtml) {
        const result = await window.electronAPI.openReportHtml(html);
        if (!result.ok) {
          addToast(`Failed to open tree viz: ${result.error || 'unknown error'}`, 'error');
        }
      } else {
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const opened = window.open(url, '_blank', 'noopener,noreferrer');
        if (!opened) {
          addToast('Pop-up blocked — allow pop-ups to view the tree', 'error');
        }
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      }
    } catch (error) {
      addToast('Failed to render tree visualization', 'error');
    } finally {
      setTreeVizLoading(false);
    }
  };

  const handleOpenBayesianNetworkViz = async () => {
    if (!analysis.modelId) return;
    setBnVizLoading(true);
    try {
      const response = await analysisApi.getBayesianNetworkViz(analysis.modelId);
      const html: string = response.data;

      if (window.electronAPI?.openReportHtml) {
        const result = await window.electronAPI.openReportHtml(html);
        if (!result.ok) {
          addToast(`Failed to open network viz: ${result.error || 'unknown error'}`, 'error');
        }
      } else {
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const opened = window.open(url, '_blank', 'noopener,noreferrer');
        if (!opened) {
          addToast('Pop-up blocked — allow pop-ups to view the network', 'error');
        }
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      }
    } catch (error) {
      addToast('Failed to render Bayesian network visualization', 'error');
    } finally {
      setBnVizLoading(false);
    }
  };

  // Top-line metrics from the best model
  const metrics = useMemo(() => {
    if (!modelResults?.metrics) return [];
    return Object.entries(modelResults.metrics)
      .filter(([, v]) => v != null)
      .map(([name, value]) => ({
        name: METRIC_LABELS[name] || name,
        value: value as number,
        color: METRIC_COLORS[name] || '#94a3b8',
      }));
  }, [modelResults]);

  // Parse comparison dict-of-dicts into rows
  const comparisonRows = useMemo(() => {
    if (!modelResults?.comparison) return [];
    const comp = modelResults.comparison;
    const modelKeys = Object.keys(comp['Model'] || {});
    return modelKeys.map((key) => {
      const row: Record<string, number | string> = {};
      for (const col of Object.keys(comp)) {
        row[col] = comp[col][key];
      }
      return row;
    });
  }, [modelResults]);

  const comparisonColumns = useMemo(() => {
    if (!modelResults?.comparison) return [];
    return Object.keys(modelResults.comparison).filter((c) => c !== 'Model');
  }, [modelResults]);

  // Prepare feature importance chart data
  const featureChartData = featureImportance
    ? featureImportance.features.map((name, i) => ({
        name: name.length > 20 ? name.substring(0, 20) + '...' : name,
        fullName: name,
        importance: featureImportance.importances[i],
      })).reverse()
    : [];

  // Derive a compact, type-aware summary of the glassbox structure payload.
  // Only renders rows whose underlying key is actually present, so the card
  // stays useful across linear / tree / additive / rules / bayesian payloads.
  const structureSummary = useMemo<Array<{ label: string; value: string }>>(() => {
    if (!structure?.structure) return [];
    const s = structure.structure as Record<string, unknown>;
    const rows: Array<{ label: string; value: string }> = [];
    const push = (label: string, value: unknown) => {
      if (value == null) return;
      rows.push({ label, value: String(value) });
    };

    push('Structure type', s.structure_type);
    push('Features', s.n_features);
    push('Classes', s.n_classes);

    const st = s.structure_type as string | undefined;
    if (st === 'tree') {
      const tree = s.tree as { root?: unknown } | undefined;
      if (tree?.root) {
        const stats = walkTree(tree.root);
        push('Nodes', stats.nodes);
        push('Leaves', stats.leaves);
        push('Max depth', stats.maxDepth);
      }
      push('Oblique method', s.oblique_method);
    } else if (st === 'tree_ensemble') {
      push('Trees in ensemble', structure.n_trees);
    } else if (st === 'linear') {
      push('Link', s.link);
      push('Solver', s.solver);
      push('Penalty', s.penalty);
    } else if (st === 'additive') {
      push('Terms', s.n_terms);
    } else if (st === 'rules' || st === 'fuzzy_rules') {
      const rules = s.rules as unknown[] | undefined;
      if (Array.isArray(rules)) push('Rules', rules.length);
    } else if (st === 'bayesian_network') {
      const nodes = s.nodes as unknown[] | undefined;
      const edges = s.edges as unknown[] | undefined;
      if (Array.isArray(nodes)) push('Nodes', nodes.length);
      if (Array.isArray(edges)) push('Edges', edges.length);
    } else if (st === 'boxes') {
      const boxes = s.boxes as unknown[] | undefined;
      if (Array.isArray(boxes)) push('Boxes', boxes.length);
    }
    return rows;
  }, [structure]);

  const handleExportResults = async () => {
    if (!analysis.modelId) return;
    // Download the full HTML report as a file
    setReportLoading(true);
    try {
      const response = await analysisApi.getReport(analysis.modelId);
      const blob = new Blob([response.data], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'classification_report.html';
      a.click();
      URL.revokeObjectURL(url);
      addToast('Report downloaded', 'success');
    } catch (error) {
      addToast('Failed to export report', 'error');
    } finally {
      setReportLoading(false);
    }
  };

  const handleViewReport = async () => {
    if (!analysis.modelId) {
      addToast('No model to report on', 'error');
      return;
    }
    setReportLoading(true);
    try {
      const response = await analysisApi.getReport(analysis.modelId);
      const html: string = response.data;

      if (typeof html !== 'string' || html.length === 0) {
        console.error('[handleViewReport] empty response body', response);
        addToast('Report response was empty', 'error');
        return;
      }

      // In Electron: write to a temp .html and hand off to the OS so the
      // report renders in the user's default browser, not in our window.
      // In a plain browser dev session: blob URL + window.open as a fallback.
      if (window.electronAPI?.openReportHtml) {
        const result = await window.electronAPI.openReportHtml(html);
        if (!result.ok) {
          console.error('[handleViewReport] electron openReportHtml failed', result);
          addToast(`Failed to open report: ${result.error || 'unknown error'}`, 'error');
        } else {
          addToast(`Report opened (${Math.round(html.length / 1024)} KB)`, 'success');
        }
      } else {
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const opened = window.open(url, '_blank', 'noopener,noreferrer');
        if (!opened) {
          addToast('Pop-up blocked — allow pop-ups to view the report', 'error');
        }
        // Revoke after the new tab has had time to load.
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      }
    } catch (error) {
      // Surface the actual backend error rather than a generic toast — that
      // way a failing report (e.g., endgame's ClassificationReport choking on
      // a specific model type) doesn't masquerade as a frontend bug.
      // getReport uses responseType: 'text', so on a FastAPI HTTPException the
      // body arrives as a JSON string rather than a parsed object.  Try both.
      const err = error as {
        response?: { data?: unknown; status?: number };
        message?: string;
      };
      let detail: string | null = null;
      const data = err?.response?.data;
      if (typeof data === 'string') {
        try {
          const parsed = JSON.parse(data);
          if (parsed && typeof parsed.detail === 'string') detail = parsed.detail;
        } catch {
          detail = data.slice(0, 500) || null;
        }
      } else if (data && typeof data === 'object' && 'detail' in data) {
        const d = (data as { detail?: unknown }).detail;
        if (typeof d === 'string') detail = d;
      }
      const fallback = err?.response?.status ? `HTTP ${err.response.status}` : err?.message;
      console.error('[handleViewReport] failed', { status: err?.response?.status, data: err?.response?.data, error });
      addToast(`Failed to generate report: ${detail || fallback || 'unknown error'}`, 'error');
    } finally {
      setReportLoading(false);
    }
  };

  const nLabels = modelResults?.class_labels.length || 0;

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="font-display text-3xl font-bold text-pie-text mb-2">
            Analysis Results
          </h1>
          <p className="text-pie-text-muted">
            Review your model performance and feature insights
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="secondary" onClick={handleViewReport} disabled={reportLoading}>
            {reportLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ExternalLink className="w-4 h-4" />}
            {reportLoading ? 'Generating...' : 'View Full Report'}
          </Button>
          <Button variant="primary" onClick={handleExportResults} disabled={reportLoading}>
            <Download className="w-4 h-4" />
            Export Results
          </Button>
        </div>
      </div>

      {resultsLoading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-pie-accent" />
          <span className="ml-3 text-pie-text-muted">Loading results...</span>
        </div>
      ) : (
        <>
          {/* Top Metrics Row.
              NOTE: grid-cols-N must be a static string for Tailwind's JIT
              scanner to emit the rule. Previously this used
              `grid-cols-${count}`, which produced no CSS and silently
              collapsed every card to one-per-row. */}
          <div
            className="grid gap-2 mb-6"
            style={{
              gridTemplateColumns: `repeat(${Math.max(metrics.length, 1)}, minmax(0, 1fr))`,
            }}
          >
            {metrics.map((metric, index) => (
              <motion.div
                key={metric.name}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.04 }}
              >
                <Card variant="glass" padding="sm">
                  <CardContent className="text-center px-2 py-2">
                    <p className="text-xl font-bold leading-tight" style={{ color: metric.color }}>
                      {(metric.value * 100).toFixed(1)}%
                    </p>
                    <p className="text-[11px] uppercase tracking-wide text-pie-text-muted mt-0.5">
                      {metric.name}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Feature Importance */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Layers className="w-5 h-5 text-pie-accent" />
                      Feature Importance
                    </CardTitle>
                    <CardDescription>Top predictive features</CardDescription>
                  </div>
                  <Button variant="ghost" size="sm" onClick={loadFeatureImportance}>
                    <RefreshCw className={clsx('w-4 h-4', loading && 'animate-spin')} />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={featureChartData}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
                      <XAxis type="number" stroke="#8b9dc3" fontSize={12} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        stroke="#8b9dc3"
                        fontSize={11}
                        width={90}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1a2540',
                          border: '1px solid #2a3a5c',
                          borderRadius: '8px'
                        }}
                        labelStyle={{ color: '#e8eff8' }}
                        formatter={(value: number, _name: string, props: { payload?: { fullName?: string } }) => [
                          value.toFixed(4),
                          props.payload?.fullName || 'Importance'
                        ]}
                      />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {featureChartData.map((_, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={`hsl(${15 + index * 8}, 85%, 60%)`}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Selected Features Summary */}
                {analysis.selectedFeatures.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-pie-border">
                    <p className="text-sm text-pie-text-muted">
                      <span className="font-medium text-pie-text">
                        {analysis.selectedFeatures.length}
                      </span>{' '}
                      features selected from original dataset
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Confusion Matrix & Model Info */}
            <div className="space-y-6">
              {/* Best Model Card */}
              {modelResults && (
                <Card variant="elevated">
                  <CardContent className="py-6">
                    <div className="flex items-center gap-4">
                      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-pie-accent to-pie-accent-secondary flex items-center justify-center">
                        <Trophy className="w-7 h-7 text-white" />
                      </div>
                      <div>
                        <p className="text-sm text-pie-text-muted">Best Performing Model</p>
                        <h3 className="text-2xl font-bold text-pie-text">
                          {modelResults.best_model_name}
                        </h3>
                        <p className="text-sm text-pie-accent">
                          {modelResults.metrics['Accuracy'] != null &&
                            `Accuracy: ${(modelResults.metrics['Accuracy']! * 100).toFixed(1)}%`}
                          {modelResults.metrics['AUC'] != null &&
                            ` | AUC: ${(modelResults.metrics['AUC']! * 100).toFixed(1)}%`}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Confusion Matrix */}
              {modelResults && modelResults.confusion_matrix.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="w-5 h-5 text-pie-accent-secondary" />
                      Confusion Matrix
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {/* Column headers */}
                    <div className="mb-2">
                      <p className="text-xs text-pie-text-muted text-center mb-1">Predicted</p>
                      <div
                        className="grid gap-1"
                        style={{ gridTemplateColumns: `80px repeat(${nLabels}, 1fr)` }}
                      >
                        <div />
                        {modelResults.class_labels.map((label) => (
                          <div key={`h-${label}`} className="text-xs text-pie-text-muted text-center truncate px-1">
                            {label}
                          </div>
                        ))}
                      </div>
                    </div>
                    {/* Matrix grid */}
                    <div className="flex gap-2">
                      {/* Row labels */}
                      <div className="flex flex-col items-center justify-center">
                        <p className="text-xs text-pie-text-muted mb-1 [writing-mode:vertical-lr] rotate-180">
                          Actual
                        </p>
                      </div>
                      <div className="flex-1">
                        {modelResults.class_labels.map((actualLabel) => (
                          <div
                            key={`row-${actualLabel}`}
                            className="grid gap-1 mb-1"
                            style={{ gridTemplateColumns: `70px repeat(${nLabels}, 1fr)` }}
                          >
                            <div className="text-xs text-pie-text-muted flex items-center truncate pr-1">
                              {actualLabel}
                            </div>
                            {modelResults.class_labels.map((predictedLabel) => {
                              const cell = modelResults.confusion_matrix.find(
                                (c) => c.actual === actualLabel && c.predicted === predictedLabel
                              );
                              const isCorrect = actualLabel === predictedLabel;
                              return (
                                <div
                                  key={`${actualLabel}-${predictedLabel}`}
                                  className={clsx(
                                    'p-2 rounded-lg text-center',
                                    isCorrect ? 'bg-pie-success/20' : 'bg-pie-error/20'
                                  )}
                                >
                                  <p className="text-lg font-bold text-pie-text">
                                    {cell?.value ?? 0}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Legend */}
                    <div className="flex justify-center gap-6 mt-4 pt-4 border-t border-pie-border">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded bg-pie-success/50" />
                        <span className="text-xs text-pie-text-muted">Correct</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded bg-pie-error/50" />
                        <span className="text-xs text-pie-text-muted">Incorrect</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>

          {/* Model Comparison Table */}
          {comparisonRows.length > 0 && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart2 className="w-5 h-5 text-pie-accent" />
                  Model Comparison
                </CardTitle>
                <CardDescription>Performance metrics across all compared models</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-pie-border">
                        <th className="text-left py-3 px-4 text-pie-text-muted font-medium">Model</th>
                        {comparisonColumns.map((col) => (
                          <th key={col} className="text-right py-3 px-4 text-pie-text-muted font-medium">
                            {METRIC_LABELS[col] || col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonRows.map((row, i) => (
                        <tr
                          key={String(row['Model'])}
                          className={clsx(
                            'border-b border-pie-border/50 transition-colors',
                            i === 0 ? 'bg-pie-accent/10' : 'hover:bg-pie-surface/50'
                          )}
                        >
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2">
                              {i === 0 && <Trophy className="w-4 h-4 text-pie-accent" />}
                              <span className={i === 0 ? 'font-medium text-pie-text' : 'text-pie-text-muted'}>
                                {String(row['Model'])}
                              </span>
                            </div>
                          </td>
                          {comparisonColumns.map((col) => {
                            const val = row[col];
                            const isTime = col === 'TT (Sec)';
                            return (
                              <td key={col} className="text-right py-3 px-4 font-mono text-sm">
                                {typeof val === 'number'
                                  ? isTime
                                    ? `${val.toFixed(1)}s`
                                    : (val * 100).toFixed(1) + '%'
                                  : String(val ?? '-')}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Winning Model Structure (endgame glassbox) */}
          <Card className="mt-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <GitBranch className="w-5 h-5 text-pie-accent-secondary" />
                    Model Structure
                  </CardTitle>
                  <CardDescription>
                    Glassbox structure of the winning model via endgame&apos;s <code className="text-pie-accent">get_structure()</code>
                  </CardDescription>
                </div>
                {(structure?.tree_viz_supported || structure?.bn_viz_supported) && (
                  <div className="flex items-center gap-2">
                    {structure?.tree_viz_supported && structure.n_trees > 1 && (
                      <label className="flex items-center gap-2 text-xs text-pie-text-muted">
                        Tree
                        <input
                          type="number"
                          min={0}
                          max={structure.n_trees - 1}
                          value={treeIndex}
                          onChange={(e) => setTreeIndex(Math.max(0, Math.min(structure.n_trees - 1, Number(e.target.value) || 0)))}
                          className="w-16 px-2 py-1 rounded border border-pie-border bg-pie-surface text-pie-text text-xs focus:outline-none focus:ring-2 focus:ring-pie-accent/50"
                        />
                        <span>/ {structure.n_trees - 1}</span>
                      </label>
                    )}
                    {structure?.tree_viz_supported && (
                      <Button variant="primary" size="sm" onClick={handleOpenTreeViz} disabled={treeVizLoading}>
                        {treeVizLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitBranch className="w-4 h-4" />}
                        {treeVizLoading ? 'Rendering...' : 'Open Tree Visualizer'}
                      </Button>
                    )}
                    {structure?.bn_viz_supported && (
                      <Button variant="primary" size="sm" onClick={handleOpenBayesianNetworkViz} disabled={bnVizLoading}>
                        {bnVizLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Network className="w-4 h-4" />}
                        {bnVizLoading ? 'Rendering...' : 'Open Network Visualizer'}
                      </Button>
                    )}
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {structureLoading && !structure ? (
                <div className="flex items-center justify-center py-6">
                  <Loader2 className="w-6 h-6 animate-spin text-pie-accent" />
                </div>
              ) : structure == null ? (
                <div className="text-sm text-pie-text-muted">No structure available.</div>
              ) : (
                <>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="px-2 py-1 rounded bg-pie-accent/10 text-pie-accent text-xs font-mono">
                      {structure.model_type}
                    </div>
                    {structure.supported ? (
                      <span className="text-xs text-pie-success">Glassbox structure available</span>
                    ) : (
                      <span className="flex items-center gap-1 text-xs text-pie-text-muted">
                        <Info className="w-3 h-3" />
                        {structure.reason || 'Structure not exposed'}
                      </span>
                    )}
                  </div>

                  {structureSummary.length > 0 && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 mb-4">
                      {structureSummary.map(({ label, value }) => (
                        <div key={label} className="px-3 py-2 rounded-lg border border-pie-border bg-pie-surface">
                          <div className="text-[11px] uppercase tracking-wide text-pie-text-muted">{label}</div>
                          <div className="text-sm font-mono text-pie-text mt-0.5">{value}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {!structure.tree_viz_supported &&
                   !structure.bn_viz_supported &&
                   structure.structure?.structure_type != null && (
                    <div className="text-xs text-pie-text-muted mb-3">
                      Interactive visualizer not available for{' '}
                      <span className="font-mono text-pie-text">{String(structure.structure.structure_type)}</span>{' '}
                      models — tree/network visualizers apply to tree-based and Bayesian models only.
                    </div>
                  )}

                  {structure.structure && (
                    <div className="mt-2">
                      <StructureDetailView structure={structure.structure} />

                      <div className="mt-4 pt-3 border-t border-pie-border/50">
                        <button
                          onClick={() => setShowRawStructure((v) => !v)}
                          className="flex items-center gap-1 text-[11px] text-pie-text-muted hover:text-pie-text"
                        >
                          {showRawStructure ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          <Code2 className="w-3 h-3" />
                          {showRawStructure ? 'Hide' : 'Show'} raw structure (JSON) — for developers
                        </button>
                        {showRawStructure && (
                          <pre className="mt-2 p-3 rounded-lg border border-pie-border bg-pie-surface/60 text-[11px] text-pie-text overflow-auto max-h-80">
                            {JSON.stringify(structure.structure, null, 2)}
                          </pre>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* Actions */}
      <div className="mt-8 flex justify-center gap-4">
        <Button variant="secondary" onClick={() => navigate('/ml')}>
          Adjust Parameters
        </Button>
        <Button variant="secondary" onClick={() => navigate('/stats')}>
          <Share2 className="w-4 h-4" />
          Statistical Analysis
        </Button>
        <Button variant="primary" onClick={() => navigate('/')}>
          New Analysis
        </Button>
      </div>

    </div>
  );
}
