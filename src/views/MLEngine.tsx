import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Target,
  Shield,
  Sliders,
  Play,
  ArrowRight,
  Loader2,
  CheckCircle,
  AlertTriangle,
  Zap,
  Settings,
  Search,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  Info,
  X,
  Terminal,
  Sparkles,
  Eye,
  Layers,
  Gauge,
  Square,
} from 'lucide-react';
import Card, { CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '../components/ui/Card';
import Button from '../components/ui/Button';
import Select from '../components/ui/Select';
import Progress from '../components/ui/Progress';
import { useStore } from '../store/useStore';
import { analysisApi, dataApi } from '../services/api';
import { clsx } from 'clsx';

interface ColumnInfo {
  name: string;
  dtype: string;
  is_numeric: boolean;
  is_categorical: boolean;
  unique_count: number;
  source_modality?: string;
}

interface ModelOption {
  id: string;
  name: string;
  family?: string;
  interpretable?: boolean;
  accepts_base_learners?: boolean;
}

// ---------------------------------------------------------------------------
// Model Arena: family hierarchy
// ---------------------------------------------------------------------------

const MODEL_FAMILIES: Record<string, { label: string; order: number }> = {
  gbdt: { label: 'Gradient Boosting', order: 1 },
  tree: { label: 'Trees', order: 2 },
  linear: { label: 'Linear', order: 3 },
  neural: { label: 'Neural Networks', order: 4 },
  bayesian: { label: 'Bayes', order: 5 },
  kernel: { label: 'Kernel', order: 6 },
  rules: { label: 'Rule-Based', order: 7 },
  ensemble: { label: 'Ensemble', order: 8 },
  foundation: { label: 'Foundation', order: 9 },
  other: { label: 'Other', order: 99 },
};

function modelFamilyKey(m: ModelOption): string {
  const key = (m.family || '').toLowerCase();
  return key && key in MODEL_FAMILIES ? key : 'other';
}

interface FSMethodOption {
  id: string;
  name: string;
  description?: string;
  requires_endgame?: boolean;
}

interface LogEntry {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error' | 'progress';
}

interface SuspiciousFeature {
  name: string;
  reason:
    | 'near_target_match'
    | 'known_leakage'
    | 'high_target_correlation'
    | 'identifier'
    | 'near_zero_variance';
  detail: string;
}

// ---------------------------------------------------------------------------
// Leakage Control: Group hierarchy definition
// ---------------------------------------------------------------------------

const MODALITY_GROUPS: Record<string, { label: string; children?: string[] }> = {
  subject_characteristics: { label: 'Subject Characteristics' },
  medical_history: { label: 'Medical History' },
  motor_assessments: { label: 'Motor Assessments' },
  non_motor_assessments: { label: 'Non-Motor Assessments' },
  biospecimen: { label: 'Biospecimen' },
};

/** Derive a top-level group key from a source_modality string. */
function topGroupKey(source: string): string {
  if (source.startsWith('medical_history__')) return 'medical_history';
  if (source.startsWith('biospecimen__')) return 'biospecimen';
  // Direct matches
  if (source in MODALITY_GROUPS) return source;
  return 'unknown';
}

/** Derive a sub-group label from a source_modality string. */
function subGroupKey(source: string): string | null {
  if (source.startsWith('medical_history__')) return source.replace('medical_history__', '');
  if (source.startsWith('biospecimen__')) return source.replace('biospecimen__', '');
  return null;
}

interface GroupNode {
  key: string;
  label: string;
  columns: ColumnInfo[];
  children: Map<string, GroupNode>;
}

// ---------------------------------------------------------------------------
// Tri-state checkbox helper
// ---------------------------------------------------------------------------

type CheckState = 'all' | 'some' | 'none';

function groupCheckState(columns: ColumnInfo[], excluded: Set<string>): CheckState {
  let hasExcluded = false;
  let hasIncluded = false;
  for (const col of columns) {
    if (excluded.has(col.name)) hasExcluded = true;
    else hasIncluded = true;
    if (hasExcluded && hasIncluded) return 'some';
  }
  return hasExcluded ? 'all' : 'none';
}

function allColumnsInGroup(node: GroupNode): ColumnInfo[] {
  const result = [...node.columns];
  for (const child of node.children.values()) {
    result.push(...allColumnsInGroup(child));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function MLEngine() {
  const navigate = useNavigate();
  const { project, data, analysis, setAnalysis, addToast, updateTask, removeTask } = useStore();

  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [fsMethod, setFsMethod] = useState('fdr');
  const [fsParam, setFsParam] = useState(0.05);
  const [leakageFeatures, setLeakageFeatures] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState<'autopilot' | 'expert' | 'automl'>('autopilot');
  const [nModels] = useState(5);
  const [tuneBest, setTuneBest] = useState(false);
  const [timeBudget, setTimeBudget] = useState(30);

  // Model Arena — individual model selection
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([]);
  const [modelSearch, setModelSearch] = useState('');
  const [expandedFamilies, setExpandedFamilies] = useState<Set<string>>(
    () => new Set(['gbdt', 'tree', 'linear', 'bayesian'])
  );
  // Base-learner selections per ensemble meta-method (e.g. "bagging" → {rf, xgboost})
  const [ensembleBaseLearners, setEnsembleBaseLearners] = useState<Map<string, Set<string>>>(
    () => new Map()
  );

  // Feature selection methods — fetched dynamically from backend
  const [fsMethods, setFsMethods] = useState<FSMethodOption[]>([]);

  // Leakage scan state
  const [leakageScanResults, setLeakageScanResults] = useState<SuspiciousFeature[] | null>(null);
  const [leakageScanning, setLeakageScanning] = useState(false);

  // AutoML state
  const [automlPreset, setAutomlPreset] = useState('good_quality');
  const [automlTimeLimit, setAutomlTimeLimit] = useState(30);

  // Post-training state
  const [calibrateMethod, setCalibrateMethod] = useState('conformal');
  const [calibrateStatus, setCalibrateStatus] = useState<'idle' | 'running' | 'done'>('idle');
  type CalibMetrics = { brier: number; log_loss: number; ece: number };
  type CalibDiagnostics = {
    before?: CalibMetrics;
    after?: CalibMetrics;
    delta?: CalibMetrics;
    n_test_samples?: number;
    error?: string;
  };
  const [calibrateDiagnostics, setCalibrateDiagnostics] = useState<CalibDiagnostics | null>(null);
  const [driftStatus, setDriftStatus] = useState<'idle' | 'running' | 'done'>('idle');
  const [driftResult, setDriftResult] = useState<string | null>(null);
  const [ensembleMethod, setEnsembleMethod] = useState('super_learner');
  const [ensembleStatus, setEnsembleStatus] = useState<'idle' | 'running' | 'done'>('idle');

  const [currentStep, setCurrentStep] = useState<'configure' | 'feature_engineering' | 'feature_selection' | 'training' | 'post_training'>('configure');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<{ status: string; progress: number; message: string } | null>(null);
  const [loading, setLoading] = useState(false);

  // Target combobox state
  const [targetSearch, setTargetSearch] = useState('');
  const [targetDropdownOpen, setTargetDropdownOpen] = useState(false);
  const targetRef = useRef<HTMLDivElement>(null);

  // Leakage search state
  const [leakageSearch, setLeakageSearch] = useState('');
  // Expanded groups: set of "topKey" or "topKey/subKey"
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());

  // Processing log console
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [showLogs, setShowLogs] = useState(true);
  const logEndRef = useRef<HTMLDivElement>(null);
  // Track how many backend log entries we've already shown (ref avoids stale closures in setInterval)
  const lastLogIndexRef = useRef(0);
  const transitioningRef = useRef(false);
  // Tracks the last task ID we've taken a terminal action on (completed /
  // failed / cancelled). Stale interval polls and in-flight requests for an
  // already-handled task short-circuit instead of re-running side effects.
  const handledTaskRef = useRef<string | null>(null);
  // Consecutive failed polls. If the backend process dies mid-run, every poll
  // throws (ECONNREFUSED) — after ~10s of failures we assume the backend is
  // gone and surface it to the user instead of spinning forever.
  const pollFailuresRef = useRef(0);
  // Indirection so the polling interval always invokes the freshest
  // pollTaskStatus closure (current taskId / currentStep) without resetting
  // its 2 s cadence on every render. Populated by an effect after
  // pollTaskStatus is declared further down.
  const pollTaskStatusRef = useRef<(() => Promise<void>) | null>(null);

  // Auto-scroll log panel to bottom when new entries arrive
  useEffect(() => {
    if (logEndRef.current && showLogs) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, showLogs]);

  const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, { timestamp, message, type }]);
  }, []);

  // ------------------------------------------------------------------
  // Data loading
  // ------------------------------------------------------------------

  useEffect(() => {
    if (!project || !data.loaded) {
      navigate('/data');
      return;
    }
    loadColumns();
  }, [project, data.loaded, navigate]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (taskId && taskStatus?.status === 'running') {
      pollFailuresRef.current = 0;
      interval = setInterval(() => {
        pollTaskStatusRef.current?.();
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [taskId, taskStatus?.status]);

  // Fetch available models when taskType changes (for expert mode)
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await analysisApi.getAvailableModels(taskType);
        const models: ModelOption[] = response.data.models;
        setAvailableModels(models);
        setSelectedModels(new Set());
      } catch {
        // Fallback: keep whatever we had
      }
    };
    fetchModels();
  }, [taskType]);

  // Fetch available feature selection methods from backend
  useEffect(() => {
    const fetchFSMethods = async () => {
      try {
        const response = await analysisApi.getFeatureSelectionMethods();
        const methods: FSMethodOption[] = response.data.methods;
        setFsMethods(methods);
      } catch {
        // Fallback: hardcoded basics
        setFsMethods([
          { id: 'fdr', name: 'False Discovery Rate (FDR)' },
          { id: 'k_best', name: 'K-Best Features' },
          { id: 'select_from_model', name: 'Model-Based' },
          { id: 'rfe', name: 'Recursive Feature Elimination' },
        ]);
      }
    };
    fetchFSMethods();
  }, []);

  const loadColumns = async () => {
    if (!data.cacheKey) return;

    try {
      const response = await dataApi.getColumns(data.cacheKey);
      const cols: ColumnInfo[] = response.data.columns;
      setColumns(cols);

      // Auto-select COHORT if it exists and is a valid target
      const cohort = cols.find(
        (c) => c.name === 'COHORT' && (c.is_categorical || (c.is_numeric && c.unique_count <= 20))
      );
      if (cohort && !targetColumn) {
        handleTargetChange(cohort.name);
      }
    } catch (error) {
      console.error('Failed to load columns:', error);
    }
  };

  // ------------------------------------------------------------------
  // Target combobox — click outside closes dropdown
  // ------------------------------------------------------------------

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (targetRef.current && !targetRef.current.contains(e.target as Node)) {
        setTargetDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleTargetChange = async (value: string) => {
    setTargetColumn(value);
    setTargetSearch(value);
    setTargetDropdownOpen(false);

    if (data.cacheKey && value) {
      try {
        const response = await analysisApi.suggestTaskType(data.cacheKey, value);
        setTaskType(response.data.suggestion as 'classification' | 'regression');
        addToast(`Suggested task type: ${response.data.suggestion}`, 'info');
      } catch (error) {
        console.error('Failed to suggest task type:', error);
      }
    }
  };

  // Eligible target columns (same filter as before)
  const targetCandidates = useMemo(
    () => columns.filter((c) => c.is_categorical || (c.is_numeric && c.unique_count <= 20)),
    [columns]
  );

  // Filtered list for the dropdown — max 50
  const filteredTargets = useMemo(() => {
    const q = targetSearch.toLowerCase();
    if (!q) return targetCandidates.slice(0, 50);
    return targetCandidates.filter((c) => c.name.toLowerCase().includes(q)).slice(0, 50);
  }, [targetCandidates, targetSearch]);

  // ------------------------------------------------------------------
  // Leakage: hierarchical groups
  // ------------------------------------------------------------------

  const featureColumns = useMemo(
    () => columns.filter((c) => c.name !== targetColumn),
    [columns, targetColumn]
  );

  const groupTree = useMemo(() => {
    const roots = new Map<string, GroupNode>();
    for (const col of featureColumns) {
      const source = col.source_modality || 'unknown';
      const topKey = topGroupKey(source);
      const sub = subGroupKey(source);

      if (!roots.has(topKey)) {
        const info = MODALITY_GROUPS[topKey] || { label: topKey };
        roots.set(topKey, { key: topKey, label: info.label, columns: [], children: new Map() });
      }
      const topNode = roots.get(topKey)!;

      if (sub) {
        if (!topNode.children.has(sub)) {
          topNode.children.set(sub, { key: `${topKey}/${sub}`, label: sub, columns: [], children: new Map() });
        }
        topNode.children.get(sub)!.columns.push(col);
      } else {
        topNode.columns.push(col);
      }
    }
    return roots;
  }, [featureColumns]);

  // Search in leakage panel — show up to 200 matches
  const leakageSearchResults = useMemo(() => {
    if (!leakageSearch) return null;
    const q = leakageSearch.toLowerCase();
    return featureColumns.filter((c) => c.name.toLowerCase().includes(q)).slice(0, 200);
  }, [featureColumns, leakageSearch]);

  const toggleLeakageFeature = (column: string) => {
    setLeakageFeatures((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(column)) newSet.delete(column);
      else newSet.add(column);
      return newSet;
    });
  };

  const toggleGroupLeakage = (node: GroupNode) => {
    const allCols = allColumnsInGroup(node);
    const state = groupCheckState(allCols, leakageFeatures);
    setLeakageFeatures((prev) => {
      const newSet = new Set(prev);
      if (state === 'all') {
        // Un-exclude all
        for (const c of allCols) newSet.delete(c.name);
      } else {
        // Exclude all
        for (const c of allCols) newSet.add(c.name);
      }
      return newSet;
    });
  };

  const toggleExpanded = (key: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  // ------------------------------------------------------------------
  // Model selection helpers
  // ------------------------------------------------------------------

  // Reasonable default base learners for new ensemble selections — a small mix
  // of strong tree-based + linear models, filtered to those actually available.
  const DEFAULT_BASE_LEARNERS = ['rf', 'lr', 'xgboost', 'xgb', 'lgbm', 'lightgbm'];

  const toggleModel = (modelId: string) => {
    const model = availableModels.find((m) => m.id === modelId);
    const isEnsembleMeta = !!model?.accepts_base_learners;

    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }
      return next;
    });

    // Seed default base learners on first check of an ensemble meta-method
    if (isEnsembleMeta) {
      setEnsembleBaseLearners((prev) => {
        const next = new Map(prev);
        if (next.has(modelId)) {
          // Was selected — wipe the config on uncheck
          if (selectedModels.has(modelId)) next.delete(modelId);
        } else if (!selectedModels.has(modelId)) {
          // Fresh check — seed defaults from available non-ensemble models
          const pool = availableModels.filter((m) => !m.accepts_base_learners);
          const seeded = new Set(
            DEFAULT_BASE_LEARNERS.filter((id) => pool.some((m) => m.id === id))
          );
          next.set(modelId, seeded);
        }
        return next;
      });
    }
  };

  const toggleBaseLearner = (ensembleId: string, learnerId: string) => {
    setEnsembleBaseLearners((prev) => {
      const next = new Map(prev);
      const cur = new Set(next.get(ensembleId) ?? []);
      if (cur.has(learnerId)) cur.delete(learnerId);
      else cur.add(learnerId);
      next.set(ensembleId, cur);
      return next;
    });
  };

  const toggleFamilyExpanded = (family: string) => {
    setExpandedFamilies((prev) => {
      const next = new Set(prev);
      if (next.has(family)) next.delete(family);
      else next.add(family);
      return next;
    });
  };

  const toggleFamilySelection = (modelsInFamily: ModelOption[]) => {
    const ids = modelsInFamily.map((m) => m.id);
    const allSelected = ids.every((id) => selectedModels.has(id));
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (allSelected) {
        for (const id of ids) next.delete(id);
      } else {
        for (const id of ids) next.add(id);
      }
      return next;
    });
    // Also seed/clear base-learner configs for any ensemble methods in this family
    const ensembleIdsInFamily = modelsInFamily
      .filter((m) => m.accepts_base_learners)
      .map((m) => m.id);
    if (ensembleIdsInFamily.length > 0) {
      setEnsembleBaseLearners((prev) => {
        const next = new Map(prev);
        if (allSelected) {
          for (const id of ensembleIdsInFamily) next.delete(id);
        } else {
          const pool = availableModels.filter((m) => !m.accepts_base_learners);
          const seeded = new Set(
            DEFAULT_BASE_LEARNERS.filter((id) => pool.some((m) => m.id === id))
          );
          for (const id of ensembleIdsInFamily) {
            if (!next.has(id)) next.set(id, new Set(seeded));
          }
        }
        return next;
      });
    }
  };

  // Group models by family for the Model Arena
  const filteredModels = useMemo(() => {
    const q = modelSearch.trim().toLowerCase();
    if (!q) return availableModels;
    return availableModels.filter(
      (m) => m.name.toLowerCase().includes(q) || m.id.toLowerCase().includes(q)
    );
  }, [availableModels, modelSearch]);

  const modelGroups = useMemo(() => {
    const buckets = new Map<string, ModelOption[]>();
    for (const m of filteredModels) {
      const key = modelFamilyKey(m);
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key)!.push(m);
    }
    for (const list of buckets.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }
    return Array.from(buckets.entries()).sort(
      ([a], [b]) => (MODEL_FAMILIES[a]?.order ?? 99) - (MODEL_FAMILIES[b]?.order ?? 99)
    );
  }, [filteredModels]);

  const selectAllModels = () => setSelectedModels(new Set(availableModels.map((m) => m.id)));
  const clearAllModels = () => setSelectedModels(new Set());

  // ------------------------------------------------------------------
  // Pipeline execution
  // ------------------------------------------------------------------

  const runFeatureEngineering = async () => {
    if (!data.cacheKey) return;

    setLoading(true);
    setCurrentStep('feature_engineering');
    setLogs([]);
    lastLogIndexRef.current = 0;
    handledTaskRef.current = null;
    transitioningRef.current = false;
    addLog('Starting ML analysis pipeline...', 'info');
    addLog(`Target: ${targetColumn}, Method: ${fsMethod}, Leakage exclusions: ${leakageFeatures.size}`, 'info');

    try {
      const response = await analysisApi.featureEngineering({
        cache_key: data.cacheKey,
        target_column: targetColumn,
        scale_numeric: true,
        one_hot_encode: true,
      });

      addLog(`Feature engineering task started (${response.data.task_id.slice(0, 8)}...)`, 'info');
      setTaskId(response.data.task_id);
      setTaskStatus({ status: 'running', progress: 0, message: 'Engineering features...' });
      updateTask(response.data.task_id, { status: 'running', progress: 0, message: 'Engineering features...' });
    } catch (error) {
      addLog('Failed to start feature engineering', 'error');
      addToast('Failed to start feature engineering', 'error');
      setLoading(false);
      setCurrentStep('configure');
    }
  };

  const runFeatureSelection = async (cacheKey: string) => {
    setCurrentStep('feature_selection');
    // NOTE: transitioningRef stays true here on purpose. Resetting it before
    // the new task ID has propagated lets stale polls of the just-completed
    // FE task re-trigger this transition (observed: "Starting feature
    // selection..." logged 5x in a single tick). pollTaskStatus resets the
    // ref once it sees the new task in the 'running' state.

    try {
      const response = await analysisApi.featureSelection({
        cache_key: cacheKey,
        target_column: targetColumn,
        method: fsMethod,
        param_value: fsParam,
        leakage_features: Array.from(leakageFeatures),
      });

      setTaskId(response.data.task_id);
      setTaskStatus({ status: 'running', progress: 0, message: 'Selecting features...' });
      updateTask(response.data.task_id, { status: 'running', progress: 0, message: 'Selecting features...' });
    } catch (error) {
      transitioningRef.current = false;
      addToast('Failed to start feature selection', 'error');
      setLoading(false);
      setCurrentStep('configure');
    }
  };

  const runModelTraining = async (trainKey: string, testKey: string) => {
    setCurrentStep('training');
    // transitioningRef stays true until pollTaskStatus sees the new task
    // running — see runFeatureSelection for the full rationale.

    try {
      let response;

      if (mode === 'automl') {
        response = await analysisApi.autoML({
          train_cache_key: trainKey,
          test_cache_key: testKey,
          target_column: targetColumn,
          time_limit: automlTimeLimit * 60,
          presets: automlPreset,
        });
      } else {
        const trainRequest: Record<string, unknown> = {
          train_cache_key: trainKey,
          test_cache_key: testKey,
          target_column: targetColumn,
          task_type: taskType,
          n_models: nModels,
          tune_best: tuneBest,
          time_budget_minutes: timeBudget,
        };
        if (mode === 'expert' && selectedModels.size > 0) {
          trainRequest.models_to_compare = Array.from(selectedModels);

          // For any selected ensemble meta-methods, attach the chosen base learners
          const ensembleConfigs: Record<string, string[]> = {};
          for (const id of selectedModels) {
            const model = availableModels.find((m) => m.id === id);
            if (model?.accepts_base_learners) {
              const learners = ensembleBaseLearners.get(id);
              if (learners && learners.size > 0) {
                ensembleConfigs[id] = Array.from(learners);
              }
            }
          }
          if (Object.keys(ensembleConfigs).length > 0) {
            trainRequest.ensemble_configs = ensembleConfigs;
          }
        }
        response = await analysisApi.train(trainRequest as any);
      }

      setTaskId(response.data.task_id);
      setTaskStatus({ status: 'running', progress: 0, message: mode === 'automl' ? 'Running AutoML...' : 'Training models...' });
      updateTask(response.data.task_id, { status: 'running', progress: 0, message: mode === 'automl' ? 'Running AutoML...' : 'Training models...' });
    } catch (error: any) {
      transitioningRef.current = false;
      const detail = error?.response?.data?.detail || error?.message || String(error);
      addToast(`Failed to start model training: ${detail}`, 'error');
      addLog(`Training error: ${detail}`, 'error');
      console.error('Training request failed:', error);
      setLoading(false);
      setCurrentStep('configure');
    }
  };

  const pollTaskStatus = useCallback(async () => {
    if (!taskId) return;
    const polledTaskId = taskId;

    try {
      const response = await analysisApi.getTaskStatus(polledTaskId);
      const status = response.data;
      pollFailuresRef.current = 0;

      // If taskId rotated while this request was in flight, OR we've already
      // acted on this task's terminal state, drop the result on the floor.
      if (polledTaskId !== taskId) return;
      if (
        handledTaskRef.current === polledTaskId &&
        status.status !== 'running' &&
        status.status !== 'pending'
      ) {
        return;
      }

      // Ingest any new log entries from the backend
      const backendLogs: { timestamp: string; message: string }[] = status.logs || [];
      if (backendLogs.length > lastLogIndexRef.current) {
        const newEntries = backendLogs.slice(lastLogIndexRef.current);
        for (const entry of newEntries) {
          let type: LogEntry['type'] = 'progress';
          const msg = entry.message || '';
          if (msg.includes('[WARNING]') || msg.includes('[ERROR]') || msg.startsWith('WARNING') || msg.startsWith('Failed')) {
            type = 'error';
          } else if (msg.includes('[INFO]') || msg.startsWith('  ')) {
            type = 'info';
          } else if (msg.includes('completed') || msg.includes('successfully') || msg.includes('complete')) {
            type = 'success';
          }
          addLog(msg, type);
        }
        lastLogIndexRef.current = backendLogs.length;
      }

      setTaskStatus(status);
      updateTask(polledTaskId, { status: status.status, progress: status.progress * 100, message: status.message });

      // Once we're successfully polling a fresh task, re-arm the transition
      // gate. This is intentionally NOT done in runFeatureSelection /
      // runModelTraining — doing it there created a window where stale polls
      // of the just-completed predecessor could re-fire the transition.
      if (status.status === 'running') {
        transitioningRef.current = false;
      }

      if (status.status === 'completed') {
        if (transitioningRef.current) return;
        transitioningRef.current = true;
        handledTaskRef.current = polledTaskId;
        removeTask(polledTaskId);

        if (currentStep === 'feature_engineering') {
          setAnalysis({ engineeredCacheKey: status.result.cache_key });
          addLog('Feature engineering completed!', 'success');
          addLog('Starting feature selection...', 'info');
          await runFeatureSelection(status.result.cache_key);
        } else if (currentStep === 'feature_selection') {
          setAnalysis({
            trainCacheKey: status.result.train_cache_key,
            testCacheKey: status.result.test_cache_key,
            selectedFeatures: status.result.selected_feature_names || [],
          });
          addLog(`Feature selection completed — ${status.result.selected_features} features selected`, 'success');
          addLog('Starting model training...', 'info');
          await runModelTraining(status.result.train_cache_key, status.result.test_cache_key);
        } else if (currentStep === 'training') {
          setAnalysis({ modelId: status.result.model_id });
          addLog('Model training completed!', 'success');
          addToast('Model training completed! Configure post-training options or view results.', 'success');
          setLoading(false);
          setCurrentStep('post_training');
        }
      } else if (status.status === 'failed') {
        if (handledTaskRef.current === polledTaskId) return;
        handledTaskRef.current = polledTaskId;
        transitioningRef.current = false;
        addLog(`Error: ${status.error}`, 'error');
        setLoading(false);
        setCurrentStep('configure');
        removeTask(polledTaskId);
        addToast(`Failed: ${status.error}`, 'error');
      } else if (status.status === 'cancelled') {
        if (handledTaskRef.current === polledTaskId) return;
        handledTaskRef.current = polledTaskId;
        transitioningRef.current = false;
        setLoading(false);
        setCurrentStep('configure');
        removeTask(polledTaskId);
      }
    } catch (error) {
      console.error('Failed to poll task status:', error);
      pollFailuresRef.current += 1;
      // 5 failures × 2s interval = ~10s of silence. At that point the backend
      // is almost certainly dead (crashed native extension, OOM, or RLIMIT_AS
      // trip) rather than briefly slow. Stop polling and tell the user, so
      // they aren't stranded watching a frozen progress bar.
      if (pollFailuresRef.current >= 5 && handledTaskRef.current !== polledTaskId) {
        handledTaskRef.current = polledTaskId;
        transitioningRef.current = false;
        addLog(
          'Lost connection to backend (5 failed polls). The Python process likely crashed — check the terminal for a traceback.',
          'error',
        );
        addToast(
          'Backend stopped responding. Training aborted — check the terminal for a crash traceback.',
          'error',
        );
        setLoading(false);
        setCurrentStep('configure');
        removeTask(polledTaskId);
        setTaskId(null);
        setTaskStatus(null);
      }
    }
  }, [taskId, currentStep, setAnalysis, addLog, addToast, updateTask, removeTask, navigate, runFeatureSelection, runModelTraining]);

  // Keep the ref consumed by the polling interval pointed at the freshest
  // pollTaskStatus closure. Defined here (not at the interval site) because
  // pollTaskStatus is declared further down.
  useEffect(() => {
    pollTaskStatusRef.current = pollTaskStatus;
  }, [pollTaskStatus]);

  const handleStop = async () => {
    if (!taskId) return;
    try {
      await analysisApi.cancelTask(taskId);
      addLog('Pipeline stopped by user', 'error');
      addToast('Pipeline stopped', 'info');
    } catch {
      // Even if the cancel request fails, reset the UI
    }
    setLoading(false);
    setCurrentStep('configure');
    setTaskId(null);
    setTaskStatus(null);
    transitioningRef.current = false;
    removeTask(taskId);
  };

  const handleLeakageScan = async () => {
    if (!data.cacheKey || !targetColumn) return;
    setLeakageScanning(true);
    setLeakageScanResults(null);
    try {
      const response = await analysisApi.detectLeakage({
        cache_key: data.cacheKey,
        target_column: targetColumn,
      });
      const results: SuspiciousFeature[] = response.data.suspicious_features;
      setLeakageScanResults(results);
      addToast(`Leakage scan complete: ${results.length} suspicious features found (scanned ${response.data.total_scanned} in ${response.data.scan_time_seconds}s)`, results.length > 0 ? 'warning' : 'success');
    } catch (error) {
      addToast('Leakage scan failed', 'error');
    } finally {
      setLeakageScanning(false);
    }
  };

  const handleExcludeAllDetected = () => {
    if (!leakageScanResults) return;
    setLeakageFeatures((prev) => {
      const newSet = new Set(prev);
      for (const f of leakageScanResults) {
        newSet.add(f.name);
      }
      return newSet;
    });
    addToast(`Excluded ${leakageScanResults.length} detected features`, 'info');
  };

  const handleCalibrate = async () => {
    if (!analysis.modelId) return;
    setCalibrateStatus('running');
    setCalibrateDiagnostics(null);
    try {
      const response = await analysisApi.calibrate({ model_id: analysis.modelId, method: calibrateMethod });
      const calTaskId = response.data.task_id;
      // Poll for calibration completion
      const poll = setInterval(async () => {
        const s = await analysisApi.getTaskStatus(calTaskId);
        if (s.data.status === 'completed') {
          clearInterval(poll);
          setAnalysis({ calibratedModelId: s.data.result.model_id, modelId: s.data.result.model_id });
          setCalibrateDiagnostics(s.data.result.diagnostics ?? null);
          setCalibrateStatus('done');
          const diag = s.data.result.diagnostics;
          if (diag?.before && diag?.after) {
            const deltaBrier = diag.after.brier - diag.before.brier;
            const direction = deltaBrier < 0 ? 'improved' : deltaBrier > 0 ? 'worsened' : 'unchanged';
            addToast(
              `Calibrated (${calibrateMethod}) — Brier ${direction} by ${Math.abs(deltaBrier).toFixed(4)}`,
              deltaBrier <= 0 ? 'success' : 'info',
            );
          } else {
            addToast(`Model calibrated with ${calibrateMethod}`, 'success');
          }
        } else if (s.data.status === 'failed') {
          clearInterval(poll);
          setCalibrateStatus('idle');
          addToast(`Calibration failed: ${s.data.error}`, 'error');
        }
      }, 2000);
    } catch {
      setCalibrateStatus('idle');
      addToast('Calibration failed', 'error');
    }
  };

  const handleDriftCheck = async () => {
    if (!analysis.trainCacheKey || !analysis.testCacheKey) return;
    setDriftStatus('running');
    try {
      const response = await analysisApi.validateDrift({
        train_cache_key: analysis.trainCacheKey,
        test_cache_key: analysis.testCacheKey,
      });
      const driftTaskId = response.data.task_id;
      const poll = setInterval(async () => {
        const s = await analysisApi.getTaskStatus(driftTaskId);
        if (s.data.status === 'completed') {
          clearInterval(poll);
          const result = s.data.result.drift_result;
          setDriftResult(result);
          setAnalysis({ driftResult: result });
          setDriftStatus('done');
          addToast('Drift validation complete', 'success');
        } else if (s.data.status === 'failed') {
          clearInterval(poll);
          setDriftStatus('idle');
          addToast(`Drift check failed: ${s.data.error}`, 'error');
        }
      }, 2000);
    } catch {
      setDriftStatus('idle');
      addToast('Drift check failed', 'error');
    }
  };

  const handleCreateEnsemble = async () => {
    if (!analysis.modelId) return;
    setEnsembleStatus('running');
    try {
      const response = await analysisApi.createEnsemble({
        model_id: analysis.modelId,
        method: ensembleMethod,
      });
      const ensTaskId = response.data.task_id;
      const poll = setInterval(async () => {
        const s = await analysisApi.getTaskStatus(ensTaskId);
        if (s.data.status === 'completed') {
          clearInterval(poll);
          setAnalysis({ ensembleModelId: s.data.result.model_id, modelId: s.data.result.model_id });
          setEnsembleStatus('done');
          addToast(`Ensemble (${ensembleMethod}) created`, 'success');
        } else if (s.data.status === 'failed') {
          clearInterval(poll);
          setEnsembleStatus('idle');
          addToast(`Ensemble failed: ${s.data.error}`, 'error');
        }
      }, 2000);
    } catch {
      setEnsembleStatus('idle');
      addToast('Ensemble creation failed', 'error');
    }
  };

  const handleRunAnalysis = () => {
    if (!targetColumn) {
      addToast('Please select a target column', 'error');
      return;
    }
    runFeatureEngineering();
  };

  const steps = [
    { id: 'configure', label: 'Configure', icon: Settings },
    { id: 'feature_engineering', label: 'Engineering', icon: Sliders },
    { id: 'feature_selection', label: 'Selection', icon: Shield },
    { id: 'training', label: 'Training', icon: Brain },
    { id: 'post_training', label: 'Enhance', icon: Sparkles },
  ];

  // ------------------------------------------------------------------
  // Render helpers
  // ------------------------------------------------------------------

  /** Tri-state checkbox indicator */
  const TriCheckbox = ({ state, onClick }: { state: CheckState; onClick: () => void }) => (
    <button
      onClick={onClick}
      className={clsx(
        'w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-colors',
        state === 'all'
          ? 'bg-pie-warning border-pie-warning'
          : state === 'some'
          ? 'bg-pie-warning/40 border-pie-warning'
          : 'border-pie-border hover:border-pie-text-muted'
      )}
    >
      {state === 'all' && <X className="w-3 h-3 text-white" />}
      {state === 'some' && <div className="w-2 h-0.5 bg-white rounded" />}
    </button>
  );

  /** Render a single group node row */
  const renderGroupNode = (node: GroupNode, depth: number = 0) => {
    const allCols = allColumnsInGroup(node);
    const state = groupCheckState(allCols, leakageFeatures);
    const isExpanded = expandedGroups.has(node.key);
    const hasChildren = node.children.size > 0 || node.columns.length > 0;
    const childNodes = Array.from(node.children.values()).sort((a, b) => a.label.localeCompare(b.label));

    return (
      <div key={node.key}>
        <div
          className={clsx(
            'flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-pie-surface/50 cursor-pointer select-none',
            depth > 0 && 'ml-4'
          )}
        >
          <button onClick={() => toggleExpanded(node.key)} className="flex-shrink-0 text-pie-text-muted hover:text-pie-text">
            {hasChildren ? (
              isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />
            ) : (
              <span className="w-4" />
            )}
          </button>
          <TriCheckbox state={state} onClick={() => toggleGroupLeakage(node)} />
          <span
            className="text-sm text-pie-text truncate flex-1"
            onClick={() => toggleExpanded(node.key)}
          >
            {node.label}
          </span>
          <span className="text-xs text-pie-text-muted flex-shrink-0">
            {allCols.length} col{allCols.length !== 1 ? 's' : ''}
          </span>
        </div>

        {isExpanded && (
          <div className={clsx(depth > 0 ? 'ml-8' : 'ml-4')}>
            {/* Sub-groups */}
            {childNodes.map((child) => renderGroupNode(child, depth + 1))}

            {/* Direct columns of this group */}
            {node.columns.map((col) => (
              <div
                key={col.name}
                className="flex items-center gap-2 py-1 px-2 ml-4"
              >
                <button
                  onClick={() => toggleLeakageFeature(col.name)}
                  className={clsx(
                    'w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-colors',
                    leakageFeatures.has(col.name)
                      ? 'bg-pie-warning border-pie-warning'
                      : 'border-pie-border hover:border-pie-text-muted'
                  )}
                >
                  {leakageFeatures.has(col.name) && <X className="w-3 h-3 text-white" />}
                </button>
                <span className="text-xs text-pie-text-muted truncate">{col.name}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-6 max-w-[1600px] mx-auto">
      {/* Compact header: title + inline progress + Run Analysis (top-right) */}
      <div className="flex items-center gap-6 mb-5 flex-wrap">
        <div className="min-w-0">
          <h1 className="font-display text-2xl font-bold text-pie-text leading-tight">
            ML Engine
          </h1>
          <p className="text-xs text-pie-text-muted">
            Configure and train machine learning models on your data
          </p>
        </div>

        {/* Inline compact progress steps */}
        <div className="flex items-center flex-1 justify-center min-w-0">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.id;
            const isCompleted = steps.findIndex((s) => s.id === currentStep) > index;

            return (
              <div key={step.id} className="flex items-center">
                <div className="flex items-center gap-1.5">
                  <div
                    className={clsx(
                      'w-7 h-7 rounded-full flex items-center justify-center transition-all flex-shrink-0',
                      isCompleted ? 'bg-pie-success text-white' :
                      isActive ? 'bg-pie-accent text-white' :
                      'bg-pie-surface text-pie-text-muted'
                    )}
                  >
                    {isCompleted ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : (
                      <Icon className="w-3.5 h-3.5" />
                    )}
                  </div>
                  <span className={clsx(
                    'text-xs whitespace-nowrap hidden xl:inline',
                    isActive ? 'text-pie-text font-medium' : 'text-pie-text-muted'
                  )}>
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className={clsx(
                    'w-4 xl:w-6 h-0.5 mx-1.5 flex-shrink-0',
                    isCompleted ? 'bg-pie-success' : 'bg-pie-border'
                  )} />
                )}
              </div>
            );
          })}
        </div>

        {/* Primary action: Run Analysis (top-right, prominent) */}
        {currentStep === 'configure' && !loading && (
          <Button
            variant="primary"
            size="lg"
            onClick={handleRunAnalysis}
            disabled={!targetColumn}
          >
            <Play className="w-4 h-4" />
            Run Analysis
            <ArrowRight className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Processing State */}
      {loading && taskStatus && (
        <Card className="mb-6">
          <CardContent className="py-8">
            <div className="text-center mb-6">
              <Loader2 className="w-12 h-12 animate-spin text-pie-accent mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-pie-text mb-2">
                {currentStep === 'feature_engineering' && 'Engineering Features...'}
                {currentStep === 'feature_selection' && 'Selecting Features...'}
                {currentStep === 'training' && (mode === 'automl' ? 'Running AutoML...' : 'Training Models...')}
              </h3>
              <p className="text-pie-text-muted">{taskStatus.message}</p>
            </div>
            <Progress
              value={taskStatus.progress * 100}
              variant="gradient"
              showLabel
            />
            <div className="mt-6 text-center">
              <Button
                variant="danger"
                onClick={handleStop}
              >
                <Square className="w-4 h-4 mr-2 fill-current" />
                Stop
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Configuration Panel: 2-column — left has Target/FS/Leakage, right is Model Arena (tall) */}
      <AnimatePresence mode="wait">
        {!loading && currentStep === 'configure' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid grid-cols-1 lg:grid-cols-[minmax(0,3fr)_minmax(380px,2fr)] gap-5 items-start"
          >
            {/* =================================================== */}
            {/* LEFT COLUMN                                         */}
            {/* =================================================== */}
            <div className="space-y-5 min-w-0">
              {/* Top row: Target + Feature Selection side-by-side */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {/* Target Variable */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Target className="w-5 h-5 text-pie-accent" />
                      Target Variable
                    </CardTitle>
                    <CardDescription>Select what you want to predict</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div ref={targetRef} className="relative">
                      <label className="block text-sm font-medium text-pie-text mb-1.5">
                        Target Column
                      </label>
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-pie-text-muted pointer-events-none" />
                        <input
                          type="text"
                          value={targetSearch}
                          onChange={(e) => {
                            setTargetSearch(e.target.value);
                            setTargetDropdownOpen(true);
                            if (!e.target.value) setTargetColumn('');
                          }}
                          onFocus={() => setTargetDropdownOpen(true)}
                          placeholder="Search columns..."
                          className="w-full pl-9 pr-3 py-2 rounded-lg border border-pie-border bg-pie-surface text-pie-text text-sm placeholder:text-pie-text-muted focus:outline-none focus:ring-2 focus:ring-pie-accent/50 focus:border-pie-accent"
                        />
                        {targetColumn && (
                          <button
                            onClick={() => {
                              setTargetColumn('');
                              setTargetSearch('');
                            }}
                            className="absolute right-2 top-1/2 -translate-y-1/2 text-pie-text-muted hover:text-pie-text"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        )}
                      </div>

                      {targetDropdownOpen && filteredTargets.length > 0 && (
                        <div className="absolute z-50 mt-1 w-full max-h-60 overflow-y-auto rounded-lg border border-pie-border bg-pie-card shadow-lg">
                          {filteredTargets.map((col) => (
                            <button
                              key={col.name}
                              onClick={() => handleTargetChange(col.name)}
                              className={clsx(
                                'w-full text-left px-3 py-2 text-sm hover:bg-pie-surface transition-colors',
                                col.name === targetColumn
                                  ? 'bg-pie-accent/10 text-pie-accent'
                                  : 'text-pie-text'
                              )}
                            >
                              <span className="font-medium">{col.name}</span>
                              <span className="text-pie-text-muted ml-2">({col.unique_count} unique)</span>
                            </button>
                          ))}
                          {targetSearch && filteredTargets.length === 50 && (
                            <div className="px-3 py-2 text-xs text-pie-text-muted text-center border-t border-pie-border">
                              Showing first 50 matches — refine your search
                            </div>
                          )}
                        </div>
                      )}
                      {targetDropdownOpen && filteredTargets.length === 0 && targetSearch && (
                        <div className="absolute z-50 mt-1 w-full rounded-lg border border-pie-border bg-pie-card shadow-lg px-3 py-4 text-sm text-pie-text-muted text-center">
                          No matching columns
                        </div>
                      )}
                    </div>

                    {targetColumn && (
                      <div className="p-3 rounded-lg bg-pie-surface">
                        <div className="flex items-center gap-2 mb-1">
                          {taskType === 'classification' ? (
                            <div className="px-2 py-1 rounded bg-blue-500/20 text-blue-400 text-xs font-medium">
                              Classification
                            </div>
                          ) : (
                            <div className="px-2 py-1 rounded bg-green-500/20 text-green-400 text-xs font-medium">
                              Regression
                            </div>
                          )}
                        </div>
                        <p className="text-xs text-pie-text-muted">
                          Detected task type based on target column
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Feature Selection */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sliders className="w-5 h-5 text-pie-accent-secondary" />
                      Feature Selection
                    </CardTitle>
                    <CardDescription>Configure feature selection method</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Select
                      label="Method"
                      value={fsMethod}
                      onChange={(e) => setFsMethod(e.target.value)}
                      options={[
                        { value: 'none', label: 'None (Skip Feature Selection)' },
                        ...fsMethods.map((m) => ({ value: m.id, label: m.name })),
                      ]}
                    />

                    {fsMethod === 'none' ? (
                      <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-500/10 text-blue-400 text-xs">
                        <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>
                          All features passed directly to training without reduction.
                        </span>
                      </div>
                    ) : ['fdr', 'k_best', 'rfe', 'select_from_model', 'mrmr', 'relief'].includes(fsMethod) ? (
                      <div className="space-y-2">
                        <label className="text-sm font-medium text-pie-text">
                          {fsMethod === 'fdr' ? 'Alpha Level' : 'Feature Fraction'}
                        </label>
                        <input
                          type="range"
                          min={fsMethod === 'fdr' ? 0.01 : 0.1}
                          max={fsMethod === 'fdr' ? 0.2 : 0.9}
                          step={0.01}
                          value={fsParam}
                          onChange={(e) => setFsParam(parseFloat(e.target.value))}
                          className="w-full"
                        />
                        <div className="flex justify-between text-xs text-pie-text-muted">
                          <span>{fsMethod === 'fdr' ? '0.01' : '10%'}</span>
                          <span className="font-medium text-pie-accent">
                            {fsMethod === 'fdr' ? fsParam.toFixed(2) : `${(fsParam * 100).toFixed(0)}%`}
                          </span>
                          <span>{fsMethod === 'fdr' ? '0.20' : '90%'}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start gap-2 p-3 rounded-lg bg-purple-500/10 text-purple-400 text-xs">
                        <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                        <span>
                          {fsMethods.find((m) => m.id === fsMethod)?.description ||
                            'This method automatically determines the optimal feature subset.'}
                        </span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Leakage Control */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="w-5 h-5 text-pie-warning" />
                    Leakage Control
                  </CardTitle>
                  <CardDescription>
                    Select features to exclude that might leak target information
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-3 mb-3">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleLeakageScan}
                      disabled={!targetColumn || !data.cacheKey || leakageScanning}
                    >
                      {leakageScanning ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Eye className="w-4 h-4" />
                      )}
                      {leakageScanning ? 'Scanning...' : 'Scan for Leakage'}
                    </Button>
                    {!targetColumn && (
                      <span className="text-xs text-pie-text-muted">Set a target column first</span>
                    )}
                  </div>

                  {leakageScanResults && leakageScanResults.length > 0 && (
                    <div className="mb-4 rounded-lg border border-pie-warning/30 bg-pie-warning/5 p-3 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-pie-warning flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4" />
                          {leakageScanResults.length} suspicious feature{leakageScanResults.length !== 1 ? 's' : ''} detected
                        </span>
                        <Button variant="secondary" size="sm" onClick={handleExcludeAllDetected}>
                          <Shield className="w-3 h-3" />
                          Exclude All Detected
                        </Button>
                      </div>
                      {(['near_target_match', 'known_leakage', 'high_target_correlation', 'identifier', 'near_zero_variance'] as const).map((reason) => {
                        const group = leakageScanResults.filter((f) => f.reason === reason);
                        if (group.length === 0) return null;
                        const labels: Record<string, string> = {
                          near_target_match: 'Near-Perfect Target Predictor',
                          known_leakage: 'Known Leakage',
                          high_target_correlation: 'High Target Correlation',
                          identifier: 'Identifier Columns',
                          near_zero_variance: 'Near-Zero Variance',
                        };
                        const colors: Record<string, string> = {
                          near_target_match: 'text-red-500',
                          known_leakage: 'text-red-400',
                          high_target_correlation: 'text-orange-400',
                          identifier: 'text-blue-400',
                          near_zero_variance: 'text-gray-400',
                        };
                        return (
                          <div key={reason}>
                            <div className={clsx('text-xs font-semibold uppercase mb-1', colors[reason])}>
                              {labels[reason]} ({group.length})
                            </div>
                            <div className="space-y-0.5">
                              {group.map((f) => (
                                <div key={f.name} className="flex items-center gap-2 text-xs">
                                  <button
                                    onClick={() => toggleLeakageFeature(f.name)}
                                    className={clsx(
                                      'px-1.5 py-0.5 rounded text-xs transition-colors',
                                      leakageFeatures.has(f.name)
                                        ? 'bg-pie-warning/20 text-pie-warning'
                                        : 'bg-pie-surface text-pie-text-muted hover:text-pie-text'
                                    )}
                                  >
                                    {leakageFeatures.has(f.name) ? 'Excluded' : 'Exclude'}
                                  </button>
                                  <span className="text-pie-text font-mono">{f.name}</span>
                                  <span className="text-pie-text-muted truncate">{f.detail}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {leakageScanResults && leakageScanResults.length === 0 && (
                    <div className="mb-4 rounded-lg border border-pie-success/30 bg-pie-success/5 p-3 flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-pie-success" />
                      <span className="text-sm text-pie-success">No suspicious features detected</span>
                    </div>
                  )}

                  <div className="relative mb-3">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-pie-text-muted pointer-events-none" />
                    <input
                      type="text"
                      value={leakageSearch}
                      onChange={(e) => setLeakageSearch(e.target.value)}
                      placeholder="Search columns..."
                      className="w-full pl-9 pr-3 py-2 rounded-lg border border-pie-border bg-pie-surface text-pie-text text-sm placeholder:text-pie-text-muted focus:outline-none focus:ring-2 focus:ring-pie-accent/50 focus:border-pie-accent"
                    />
                    {leakageSearch && (
                      <button
                        onClick={() => setLeakageSearch('')}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-pie-text-muted hover:text-pie-text"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>

                  <div className="max-h-72 overflow-y-auto">
                    {leakageSearchResults ? (
                      leakageSearchResults.length === 0 ? (
                        <div className="text-sm text-pie-text-muted text-center py-4">No matching columns</div>
                      ) : (
                        <div className="space-y-0.5">
                          {leakageSearchResults.map((col) => (
                            <div key={col.name} className="flex items-center gap-2 py-1 px-2">
                              <button
                                onClick={() => toggleLeakageFeature(col.name)}
                                className={clsx(
                                  'w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-colors',
                                  leakageFeatures.has(col.name)
                                    ? 'bg-pie-warning border-pie-warning'
                                    : 'border-pie-border hover:border-pie-text-muted'
                                )}
                              >
                                {leakageFeatures.has(col.name) && <X className="w-3 h-3 text-white" />}
                              </button>
                              <span className="text-sm text-pie-text truncate flex-1">{col.name}</span>
                              <span className="text-xs text-pie-text-muted flex-shrink-0">
                                {col.source_modality || 'unknown'}
                              </span>
                            </div>
                          ))}
                          {leakageSearchResults.length === 200 && (
                            <div className="text-xs text-pie-text-muted text-center py-2 border-t border-pie-border">
                              Showing first 200 matches — refine your search
                            </div>
                          )}
                        </div>
                      )
                    ) : (
                      <div className="space-y-0.5">
                        {Array.from(groupTree.entries())
                          .sort(([a], [b]) => a.localeCompare(b))
                          .map(([, node]) => renderGroupNode(node, 0))}
                      </div>
                    )}
                  </div>

                  {leakageFeatures.size > 0 && (
                    <p className="mt-3 text-sm text-pie-warning flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      {leakageFeatures.size} feature{leakageFeatures.size > 1 ? 's' : ''} will be excluded
                    </p>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* =================================================== */}
            {/* RIGHT COLUMN — Model Arena (tall, hierarchical)     */}
            {/* =================================================== */}
            <Card className="lg:sticky lg:top-4 flex flex-col max-h-[calc(100vh-7rem)]">
              <CardHeader className="flex-shrink-0">
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-pie-accent" />
                  Model Arena
                </CardTitle>
                <CardDescription>Configure model training</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col gap-4 min-h-0 overflow-hidden">
                {/* Mode toggle */}
                <div className="flex rounded-lg bg-pie-surface p-1 flex-shrink-0">
                  <button
                    onClick={() => setMode('autopilot')}
                    className={clsx(
                      'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all',
                      mode === 'autopilot'
                        ? 'bg-pie-accent text-white'
                        : 'text-pie-text-muted hover:text-pie-text'
                    )}
                  >
                    <Zap className="w-4 h-4 inline mr-1" />
                    Auto-Pilot
                  </button>
                  <button
                    onClick={() => setMode('expert')}
                    className={clsx(
                      'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all',
                      mode === 'expert'
                        ? 'bg-pie-accent text-white'
                        : 'text-pie-text-muted hover:text-pie-text'
                    )}
                  >
                    <Settings className="w-4 h-4 inline mr-1" />
                    Expert
                  </button>
                  <button
                    onClick={() => setMode('automl')}
                    className={clsx(
                      'flex-1 py-2 px-3 rounded-md text-sm font-medium transition-all',
                      mode === 'automl'
                        ? 'bg-pie-accent text-white'
                        : 'text-pie-text-muted hover:text-pie-text'
                    )}
                  >
                    <Sparkles className="w-4 h-4 inline mr-1" />
                    AutoML
                  </button>
                </div>

                {mode === 'autopilot' && (
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-500/10 text-blue-400 text-xs flex-shrink-0">
                    <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <span>
                      Auto-Pilot compares a curated selection of {nModels} strong baseline models
                      within the configured time budget.
                    </span>
                  </div>
                )}

                {mode === 'automl' && (
                  <div className="space-y-4 flex-shrink-0">
                    <Select
                      label="Preset"
                      value={automlPreset}
                      onChange={(e) => setAutomlPreset(e.target.value)}
                      options={[
                        { value: 'good_quality', label: 'Good Quality (fastest)' },
                        { value: 'high_quality', label: 'High Quality' },
                        { value: 'best_quality', label: 'Best Quality (slowest)' },
                      ]}
                    />
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-pie-text">
                        Time Limit: {automlTimeLimit} min
                      </label>
                      <input
                        type="range"
                        min={5}
                        max={120}
                        step={5}
                        value={automlTimeLimit}
                        onChange={(e) => setAutomlTimeLimit(parseInt(e.target.value))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-pie-text-muted">
                        <span>5 min</span>
                        <span>120 min</span>
                      </div>
                    </div>
                    <div className="flex items-start gap-2 p-3 rounded-lg bg-purple-500/10 text-purple-400 text-xs">
                      <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>
                        AutoML will automatically try many model architectures and hyperparameter
                        configurations within the time budget to find the best performer.
                      </span>
                    </div>
                  </div>
                )}

                {mode === 'expert' && (
                  <>
                    {/* Summary + bulk actions */}
                    <div className="flex items-center justify-between flex-shrink-0">
                      <label className="text-sm font-medium text-pie-text">
                        Models to Compare
                        <span className="text-pie-text-muted font-normal ml-1">
                          ({selectedModels.size}/{availableModels.length})
                        </span>
                      </label>
                      <div className="flex items-center gap-2 text-xs">
                        <button
                          onClick={selectAllModels}
                          className="text-pie-accent hover:underline"
                          disabled={availableModels.length === 0}
                        >
                          All
                        </button>
                        <span className="text-pie-text-muted">·</span>
                        <button
                          onClick={clearAllModels}
                          className="text-pie-text-muted hover:text-pie-text"
                          disabled={selectedModels.size === 0}
                        >
                          Clear
                        </button>
                      </div>
                    </div>

                    {/* Search */}
                    <div className="relative flex-shrink-0">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-pie-text-muted pointer-events-none" />
                      <input
                        type="text"
                        value={modelSearch}
                        onChange={(e) => setModelSearch(e.target.value)}
                        placeholder="Search models..."
                        className="w-full pl-9 pr-3 py-2 rounded-lg border border-pie-border bg-pie-surface text-pie-text text-sm placeholder:text-pie-text-muted focus:outline-none focus:ring-2 focus:ring-pie-accent/50 focus:border-pie-accent"
                      />
                      {modelSearch && (
                        <button
                          onClick={() => setModelSearch('')}
                          className="absolute right-2 top-1/2 -translate-y-1/2 text-pie-text-muted hover:text-pie-text"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>

                    {/* Hierarchical family tree (tall, flex-1 scrollable) */}
                    <div className="flex-1 min-h-0 overflow-y-auto rounded-lg border border-pie-border bg-pie-surface/40 p-2">
                      {modelGroups.length === 0 ? (
                        <div className="text-sm text-pie-text-muted text-center py-6">
                          {availableModels.length === 0 ? 'Loading models...' : 'No matching models'}
                        </div>
                      ) : (
                        <div className="space-y-1">
                          {modelGroups.map(([familyKey, familyModels]) => {
                            // When searching, auto-expand all groups with matches
                            const searching = modelSearch.trim().length > 0;
                            const isExpanded = searching || expandedFamilies.has(familyKey);
                            const familyLabel = MODEL_FAMILIES[familyKey]?.label ?? familyKey;
                            const selectedInFamily = familyModels.filter((m) =>
                              selectedModels.has(m.id)
                            ).length;
                            const allSelected =
                              selectedInFamily === familyModels.length && familyModels.length > 0;
                            const someSelected =
                              selectedInFamily > 0 && selectedInFamily < familyModels.length;

                            return (
                              <div key={familyKey}>
                                <div className="flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-pie-card/50 cursor-pointer select-none">
                                  <button
                                    onClick={() => toggleFamilyExpanded(familyKey)}
                                    className="flex-shrink-0 text-pie-text-muted hover:text-pie-text"
                                  >
                                    {isExpanded ? (
                                      <ChevronDown className="w-4 h-4" />
                                    ) : (
                                      <ChevronRight className="w-4 h-4" />
                                    )}
                                  </button>
                                  <button
                                    onClick={() => toggleFamilySelection(familyModels)}
                                    className={clsx(
                                      'w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-colors',
                                      allSelected
                                        ? 'bg-pie-accent border-pie-accent'
                                        : someSelected
                                        ? 'bg-pie-accent/40 border-pie-accent'
                                        : 'border-pie-border hover:border-pie-text-muted'
                                    )}
                                    aria-label={`Toggle all ${familyLabel} models`}
                                  >
                                    {allSelected && <CheckCircle className="w-3 h-3 text-white" />}
                                    {someSelected && <div className="w-2 h-0.5 bg-white rounded" />}
                                  </button>
                                  <span
                                    className="text-sm font-medium text-pie-text flex-1 truncate"
                                    onClick={() => toggleFamilyExpanded(familyKey)}
                                  >
                                    {familyLabel}
                                  </span>
                                  <span className="text-xs text-pie-text-muted flex-shrink-0">
                                    {selectedInFamily}/{familyModels.length}
                                  </span>
                                </div>

                                {isExpanded && (
                                  <div className="ml-6 mt-0.5 space-y-0.5">
                                    {familyModels.map((model) => {
                                      const isChecked = selectedModels.has(model.id);
                                      const isMetaEnsemble = !!model.accepts_base_learners;
                                      const baseLearnerPool = availableModels.filter(
                                        (m) => !m.accepts_base_learners
                                      );
                                      const selectedLearners =
                                        ensembleBaseLearners.get(model.id) ?? new Set<string>();

                                      return (
                                        <div key={model.id}>
                                          <label
                                            className="flex items-center gap-2 py-1 px-2 rounded hover:bg-pie-card/50 cursor-pointer"
                                          >
                                            <input
                                              type="checkbox"
                                              checked={isChecked}
                                              onChange={() => toggleModel(model.id)}
                                              className="rounded border-pie-border flex-shrink-0"
                                            />
                                            <span className="text-sm text-pie-text truncate flex-1">
                                              {model.name}
                                            </span>
                                            {isMetaEnsemble && (
                                              <span
                                                className="text-[10px] px-1.5 py-0.5 rounded bg-pie-accent/15 text-pie-accent flex-shrink-0"
                                                title="Meta-ensemble — wraps base learners you pick"
                                              >
                                                meta
                                              </span>
                                            )}
                                            {model.interpretable && (
                                              <span
                                                className="text-[10px] px-1.5 py-0.5 rounded bg-pie-success/15 text-pie-success flex-shrink-0"
                                                title="Interpretable model"
                                              >
                                                glass-box
                                              </span>
                                            )}
                                          </label>

                                          {/* Base-learner picker for checked meta-ensembles */}
                                          {isChecked && isMetaEnsemble && (
                                            <div className="ml-7 mt-1 mb-2 p-2 rounded-md border border-pie-accent/30 bg-pie-accent/5">
                                              <div className="flex items-center justify-between mb-1.5">
                                                <span className="text-[11px] font-medium text-pie-accent uppercase tracking-wide">
                                                  Base learners
                                                  <span className="ml-1 text-pie-text-muted normal-case tracking-normal font-normal">
                                                    ({selectedLearners.size} selected)
                                                  </span>
                                                </span>
                                                <div className="flex items-center gap-2 text-[11px]">
                                                  <button
                                                    onClick={() =>
                                                      setEnsembleBaseLearners((prev) => {
                                                        const next = new Map(prev);
                                                        next.set(
                                                          model.id,
                                                          new Set(baseLearnerPool.map((m) => m.id))
                                                        );
                                                        return next;
                                                      })
                                                    }
                                                    className="text-pie-accent hover:underline"
                                                  >
                                                    All
                                                  </button>
                                                  <span className="text-pie-text-muted">·</span>
                                                  <button
                                                    onClick={() =>
                                                      setEnsembleBaseLearners((prev) => {
                                                        const next = new Map(prev);
                                                        next.set(model.id, new Set());
                                                        return next;
                                                      })
                                                    }
                                                    className="text-pie-text-muted hover:text-pie-text"
                                                  >
                                                    Clear
                                                  </button>
                                                </div>
                                              </div>
                                              {selectedLearners.size === 0 && (
                                                <div className="text-[11px] text-pie-warning mb-1">
                                                  No base learners selected — this ensemble won't train.
                                                </div>
                                              )}
                                              <div className="max-h-40 overflow-y-auto grid grid-cols-2 gap-x-2 gap-y-0.5">
                                                {baseLearnerPool.map((learner) => (
                                                  <label
                                                    key={learner.id}
                                                    className="flex items-center gap-1.5 py-0.5 px-1 rounded hover:bg-pie-card/60 cursor-pointer"
                                                  >
                                                    <input
                                                      type="checkbox"
                                                      checked={selectedLearners.has(learner.id)}
                                                      onChange={() =>
                                                        toggleBaseLearner(model.id, learner.id)
                                                      }
                                                      className="rounded border-pie-border flex-shrink-0 w-3 h-3"
                                                    />
                                                    <span className="text-[11px] text-pie-text truncate">
                                                      {learner.name}
                                                    </span>
                                                  </label>
                                                ))}
                                              </div>
                                            </div>
                                          )}
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>

                    {/* Tune + time budget (below the scroll area) */}
                    <div className="space-y-3 flex-shrink-0">
                      <div className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          id="tuneBest"
                          checked={tuneBest}
                          onChange={(e) => setTuneBest(e.target.checked)}
                          className="rounded border-pie-border"
                        />
                        <label htmlFor="tuneBest" className="text-sm text-pie-text">
                          Tune best model
                        </label>
                      </div>

                      <div className="space-y-2">
                        <label className="text-sm font-medium text-pie-text">
                          Time Budget: {timeBudget} min
                        </label>
                        <input
                          type="range"
                          min={5}
                          max={120}
                          step={5}
                          value={timeBudget}
                          onChange={(e) => setTimeBudget(parseInt(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ============================================================ */}
      {/* Post-Training Options Panel                                  */}
      {/* ============================================================ */}
      <AnimatePresence mode="wait">
        {!loading && currentStep === 'post_training' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-pie-accent" />
                  Post-Training Enhancement
                </CardTitle>
                <CardDescription>
                  Calibrate, validate, or ensemble your trained model before viewing results
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-6">
                  {/* Column 1 — Calibrate Model */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Gauge className="w-4 h-4 text-pie-accent-secondary" />
                      <span className="text-sm font-semibold text-pie-text">Calibrate Model</span>
                      <span className={clsx(
                        'text-xs px-2 py-0.5 rounded-full',
                        calibrateStatus === 'done' ? 'bg-pie-success/20 text-pie-success' :
                        calibrateStatus === 'running' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-pie-surface text-pie-text-muted'
                      )}>
                        {calibrateStatus === 'done' ? 'Calibrated' : calibrateStatus === 'running' ? 'Running...' : 'Uncalibrated'}
                      </span>
                    </div>
                    <Select
                      label="Method"
                      value={calibrateMethod}
                      onChange={(e) => setCalibrateMethod(e.target.value)}
                      options={[
                        { value: 'conformal', label: 'Conformal' },
                        { value: 'temperature_scaling', label: 'Temperature Scaling' },
                        { value: 'venn_abers', label: 'Venn-Abers' },
                        { value: 'platt', label: 'Platt Scaling' },
                        { value: 'isotonic', label: 'Isotonic Regression' },
                      ]}
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleCalibrate}
                      disabled={calibrateStatus === 'running' || !analysis.modelId}
                      className="w-full"
                    >
                      {calibrateStatus === 'running' ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Gauge className="w-4 h-4" />
                      )}
                      {calibrateStatus === 'done' ? 'Re-Calibrate' : 'Calibrate'}
                    </Button>
                    {calibrateStatus === 'done' && calibrateDiagnostics?.before && calibrateDiagnostics?.after && (
                      <div className="rounded-lg border border-pie-border bg-pie-surface/60 p-2 text-xs">
                        <div className="grid grid-cols-3 gap-1 mb-1 text-pie-text-muted font-medium">
                          <span />
                          <span className="text-right">Before</span>
                          <span className="text-right">After</span>
                        </div>
                        {(['brier', 'log_loss', 'ece'] as const).map((k) => {
                          const before = calibrateDiagnostics.before![k];
                          const after = calibrateDiagnostics.after![k];
                          const delta = after - before;
                          const improved = delta < 0;
                          const label = k === 'log_loss' ? 'Log loss' : k === 'ece' ? 'ECE' : 'Brier';
                          return (
                            <div key={k} className="grid grid-cols-3 gap-1 items-center">
                              <span className="text-pie-text-muted" title={
                                k === 'brier' ? 'Brier score — mean squared error of predicted probabilities. Lower is better.' :
                                k === 'log_loss' ? 'Log loss — penalizes confident wrong predictions. Lower is better.' :
                                'Expected Calibration Error — gap between confidence and accuracy. Lower is better.'
                              }>{label}</span>
                              <span className="text-right font-mono text-pie-text">{before.toFixed(4)}</span>
                              <span className={clsx(
                                'text-right font-mono',
                                improved ? 'text-pie-success' : delta > 0 ? 'text-pie-warning' : 'text-pie-text'
                              )}>
                                {after.toFixed(4)}
                                <span className="ml-1 text-[10px]">
                                  {improved ? '▼' : delta > 0 ? '▲' : '='}
                                </span>
                              </span>
                            </div>
                          );
                        })}
                        {calibrateDiagnostics.n_test_samples != null && (
                          <div className="mt-1 pt-1 border-t border-pie-border text-[10px] text-pie-text-muted">
                            {calibrateDiagnostics.n_test_samples.toLocaleString()} test samples
                          </div>
                        )}
                      </div>
                    )}
                    {calibrateStatus === 'done' && calibrateDiagnostics?.error && (
                      <div className="rounded-lg bg-pie-warning/10 text-pie-warning p-2 text-xs">
                        Diagnostics unavailable: {calibrateDiagnostics.error}
                      </div>
                    )}
                  </div>

                  {/* Column 2 — Drift Validation */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Eye className="w-4 h-4 text-pie-accent-secondary" />
                      <span className="text-sm font-semibold text-pie-text">Drift Validation</span>
                    </div>
                    <p className="text-xs text-pie-text-muted">
                      Check if your train and test distributions are consistent using adversarial validation.
                    </p>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleDriftCheck}
                      disabled={driftStatus === 'running' || !analysis.trainCacheKey || !analysis.testCacheKey}
                      className="w-full"
                    >
                      {driftStatus === 'running' ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Eye className="w-4 h-4" />
                      )}
                      {driftStatus === 'running' ? 'Checking...' : 'Check for Drift'}
                    </Button>
                    {driftStatus === 'done' && driftResult && (
                      <div className={clsx(
                        'p-2 rounded-lg text-xs flex items-center gap-2',
                        driftResult.toLowerCase().includes('drift') || driftResult.toLowerCase().includes('high')
                          ? 'bg-pie-warning/10 text-pie-warning'
                          : 'bg-pie-success/10 text-pie-success'
                      )}>
                        {driftResult.toLowerCase().includes('drift') || driftResult.toLowerCase().includes('high')
                          ? <AlertTriangle className="w-3 h-3" />
                          : <CheckCircle className="w-3 h-3" />
                        }
                        {driftResult}
                      </div>
                    )}
                  </div>

                  {/* Column 3 — Create Ensemble */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-2">
                      <Layers className="w-4 h-4 text-pie-accent-secondary" />
                      <span className="text-sm font-semibold text-pie-text">Create Ensemble</span>
                      {ensembleStatus === 'done' && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-pie-success/20 text-pie-success">
                          Created
                        </span>
                      )}
                    </div>
                    <Select
                      label="Method"
                      value={ensembleMethod}
                      onChange={(e) => setEnsembleMethod(e.target.value)}
                      options={[
                        { value: 'super_learner', label: 'Super Learner' },
                        { value: 'bma', label: 'Bayesian Model Avg.' },
                        { value: 'blending', label: 'Blending' },
                        { value: 'bagging', label: 'Bagging' },
                        { value: 'boosting', label: 'Boosting' },
                      ]}
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleCreateEnsemble}
                      disabled={ensembleStatus === 'running' || !analysis.modelId}
                      className="w-full"
                    >
                      {ensembleStatus === 'running' ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Layers className="w-4 h-4" />
                      )}
                      {ensembleStatus === 'running' ? 'Building...' : 'Build Ensemble'}
                    </Button>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <Button
                  variant="primary"
                  className="ml-auto"
                  onClick={() => navigate('/results')}
                >
                  <ArrowRight className="w-4 h-4" />
                  View Results
                </Button>
              </CardFooter>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ============================================================ */}
      {/* Terminal Log Panel                                           */}
      {/* ============================================================ */}
      {logs.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-6"
        >
          <Card className="overflow-hidden">
            <button
              onClick={() => setShowLogs(!showLogs)}
              className="w-full px-4 py-3 flex items-center justify-between bg-pie-surface hover:bg-pie-card transition-colors border-b border-pie-border"
            >
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-pie-accent" />
                <span className="text-sm font-medium text-pie-text">Processing Log</span>
                {logs.length > 0 && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-pie-accent/20 text-pie-accent">
                    {logs.length} entries
                  </span>
                )}
              </div>
              {showLogs ? (
                <ChevronUp className="w-4 h-4 text-pie-text-muted" />
              ) : (
                <ChevronDown className="w-4 h-4 text-pie-text-muted" />
              )}
            </button>

            {showLogs && (
              <div className="bg-[#0d1117] p-4 font-mono text-sm max-h-64 overflow-y-auto">
                {logs.length === 0 ? (
                  <p className="text-gray-500">Waiting for operations...</p>
                ) : (
                  <div className="space-y-1">
                    {logs.map((log, index) => (
                      <div key={index} className="flex gap-3">
                        <span className="text-gray-600 flex-shrink-0">[{log.timestamp}]</span>
                        <span
                          className={clsx(
                            log.type === 'success' && 'text-green-400',
                            log.type === 'error' && 'text-red-400',
                            log.type === 'progress' && 'text-blue-400',
                            log.type === 'info' && 'text-gray-300'
                          )}
                        >
                          {log.type === 'progress' && '\u27F3 '}
                          {log.type === 'success' && '\u2713 '}
                          {log.type === 'error' && '\u2717 '}
                          {log.message}
                        </span>
                      </div>
                    ))}
                    <div ref={logEndRef} />
                  </div>
                )}
                {loading && (
                  <div className="flex items-center gap-2 mt-2 text-blue-400">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    <span>
                      {taskStatus?.message
                        ? `Working: ${taskStatus.message}`
                        : 'Processing...'}
                    </span>
                  </div>
                )}
              </div>
            )}
          </Card>
        </motion.div>
      )}
    </div>
  );
}
