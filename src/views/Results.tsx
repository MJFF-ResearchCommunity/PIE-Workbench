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
    if (!analysis.modelId) return;
    setReportLoading(true);
    try {
      const response = await analysisApi.getReport(analysis.modelId);
      const html: string = response.data;

      // In Electron: write to a temp .html and hand off to the OS so the
      // report renders in the user's default browser, not in our window.
      // In a plain browser dev session: blob URL + window.open as a fallback.
      if (window.electronAPI?.openReportHtml) {
        const result = await window.electronAPI.openReportHtml(html);
        if (!result.ok) {
          addToast(`Failed to open report: ${result.error || 'unknown error'}`, 'error');
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
      addToast('Failed to generate report', 'error');
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
                {structure?.tree_viz_supported && (
                  <div className="flex items-center gap-2">
                    {structure.n_trees > 1 && (
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
                    <Button variant="primary" size="sm" onClick={handleOpenTreeViz} disabled={treeVizLoading}>
                      {treeVizLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitBranch className="w-4 h-4" />}
                      {treeVizLoading ? 'Rendering...' : 'Open Tree Visualizer'}
                    </Button>
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

                  {!structure.tree_viz_supported && structure.structure?.structure_type != null && (
                    <div className="text-xs text-pie-text-muted mb-2">
                      Tree visualizer is only available for tree-based models; this model is{' '}
                      <span className="font-mono text-pie-text">{String(structure.structure.structure_type)}</span>.
                    </div>
                  )}

                  {structure.structure && (
                    <div className="mt-2">
                      <button
                        onClick={() => setShowRawStructure((v) => !v)}
                        className="flex items-center gap-1 text-xs text-pie-text-muted hover:text-pie-text"
                      >
                        {showRawStructure ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                        <Code2 className="w-3 h-3" />
                        {showRawStructure ? 'Hide' : 'Show'} raw structure (JSON)
                      </button>
                      {showRawStructure && (
                        <pre className="mt-2 p-3 rounded-lg border border-pie-border bg-pie-surface/60 text-[11px] text-pie-text overflow-auto max-h-80">
                          {JSON.stringify(structure.structure, null, 2)}
                        </pre>
                      )}
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
