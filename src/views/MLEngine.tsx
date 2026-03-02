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
  reason: 'known_leakage' | 'high_target_correlation' | 'identifier' | 'near_zero_variance';
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
      interval = setInterval(pollTaskStatus, 2000);
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
        // Default: select all models
        setSelectedModels(new Set(models.map((m) => m.id)));
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

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) next.delete(modelId);
      else next.add(modelId);
      return next;
    });
  };

  // ------------------------------------------------------------------
  // Pipeline execution
  // ------------------------------------------------------------------

  const runFeatureEngineering = async () => {
    if (!data.cacheKey) return;

    setLoading(true);
    setCurrentStep('feature_engineering');
    setLogs([]);
    lastLogIndexRef.current = 0;
    addLog('Starting ML analysis pipeline...', 'info');
    addLog(`Target: ${targetColumn}, Method: ${fsMethod}, Leakage exclusions: ${leakageFeatures.size}`, 'info');

    try {
      const response = await analysisApi.featureEngineering({
        cache_key: data.cacheKey,
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
      addToast('Failed to start feature selection', 'error');
      setLoading(false);
      setCurrentStep('configure');
    }
  };

  const runModelTraining = async (trainKey: string, testKey: string) => {
    setCurrentStep('training');

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
        }
        response = await analysisApi.train(trainRequest as any);
      }

      setTaskId(response.data.task_id);
      setTaskStatus({ status: 'running', progress: 0, message: mode === 'automl' ? 'Running AutoML...' : 'Training models...' });
      updateTask(response.data.task_id, { status: 'running', progress: 0, message: mode === 'automl' ? 'Running AutoML...' : 'Training models...' });
    } catch (error) {
      addToast('Failed to start model training', 'error');
      setLoading(false);
      setCurrentStep('configure');
    }
  };

  const pollTaskStatus = useCallback(async () => {
    if (!taskId) return;

    try {
      const response = await analysisApi.getTaskStatus(taskId);
      const status = response.data;

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
      updateTask(taskId, { status: status.status, progress: status.progress * 100, message: status.message });

      if (status.status === 'completed') {
        removeTask(taskId);

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
        addLog(`Error: ${status.error}`, 'error');
        setLoading(false);
        setCurrentStep('configure');
        removeTask(taskId);
        addToast(`Failed: ${status.error}`, 'error');
      }
    } catch (error) {
      console.error('Failed to poll task status:', error);
    }
  }, [taskId, currentStep, setAnalysis, addLog, addToast, updateTask, removeTask, navigate, runFeatureSelection, runModelTraining]);

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
    try {
      const response = await analysisApi.calibrate({ model_id: analysis.modelId, method: calibrateMethod });
      const calTaskId = response.data.task_id;
      // Poll for calibration completion
      const poll = setInterval(async () => {
        const s = await analysisApi.getTaskStatus(calTaskId);
        if (s.data.status === 'completed') {
          clearInterval(poll);
          setAnalysis({ calibratedModelId: s.data.result.model_id, modelId: s.data.result.model_id });
          setCalibrateStatus('done');
          addToast(`Model calibrated with ${calibrateMethod}`, 'success');
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
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold text-pie-text mb-2">
          ML Engine
        </h1>
        <p className="text-pie-text-muted">
          Configure and train machine learning models on your data
        </p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between max-w-2xl mx-auto">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = currentStep === step.id;
            const isCompleted = steps.findIndex((s) => s.id === currentStep) > index;

            return (
              <div key={step.id} className="flex items-center">
                <div className="flex flex-col items-center">
                  <div
                    className={clsx(
                      'w-10 h-10 rounded-full flex items-center justify-center transition-all',
                      isCompleted ? 'bg-pie-success text-white' :
                      isActive ? 'bg-pie-accent text-white' :
                      'bg-pie-surface text-pie-text-muted'
                    )}
                  >
                    {isCompleted ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                  </div>
                  <span className={clsx(
                    'text-xs mt-2',
                    isActive ? 'text-pie-text' : 'text-pie-text-muted'
                  )}>
                    {step.label}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className={clsx(
                    'w-16 h-0.5 mx-2',
                    isCompleted ? 'bg-pie-success' : 'bg-pie-border'
                  )} />
                )}
              </div>
            );
          })}
        </div>
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
          </CardContent>
        </Card>
      )}

      {/* Configuration Panel */}
      <AnimatePresence mode="wait">
        {!loading && currentStep === 'configure' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="grid grid-cols-3 gap-6"
          >
            {/* ============================================================ */}
            {/* 1. Target Variable — Searchable Combobox                     */}
            {/* ============================================================ */}
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

                  {/* Dropdown */}
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
                    <div className="flex items-center gap-2 mb-2">
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
                    <p className="text-sm text-pie-text-muted">
                      Detected task type based on target column
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* ============================================================ */}
            {/* 2. Feature Selection — with "None (Skip)" option             */}
            {/* ============================================================ */}
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
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-500/10 text-blue-400 text-sm">
                    <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <span>
                      All features will be passed directly to model training without reduction.
                      Useful when your feature set is already curated.
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
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-purple-500/10 text-purple-400 text-sm">
                    <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <span>
                      {fsMethods.find((m) => m.id === fsMethod)?.description ||
                        'This method automatically determines the optimal feature subset.'}
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* ============================================================ */}
            {/* 3. Model Arena — Individual Model Selection                  */}
            {/* ============================================================ */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-pie-accent" />
                  Model Arena
                </CardTitle>
                <CardDescription>Configure model training</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Mode Toggle */}
                <div className="flex rounded-lg bg-pie-surface p-1">
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

                {mode === 'automl' && (
                  <div className="space-y-4">
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
                    <div className="flex items-start gap-2 p-3 rounded-lg bg-purple-500/10 text-purple-400 text-sm">
                      <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>
                        AutoML will automatically try many model architectures and hyperparameter
                        configurations within the time budget to find the best performer.
                      </span>
                    </div>
                  </div>
                )}

                {mode === 'expert' && (
                  <div className="space-y-4">
                    {/* Model checkbox list */}
                    <div>
                      <label className="block text-sm font-medium text-pie-text mb-2">
                        Models to Compare ({selectedModels.size}/{availableModels.length})
                      </label>
                      <div className="max-h-40 overflow-y-auto space-y-1 p-2 rounded-lg border border-pie-border bg-pie-surface">
                        {availableModels.map((model) => (
                          <label
                            key={model.id}
                            className="flex items-center gap-2 py-1 px-1 rounded hover:bg-pie-card/50 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              checked={selectedModels.has(model.id)}
                              onChange={() => toggleModel(model.id)}
                              className="rounded border-pie-border"
                            />
                            <span className="text-sm text-pie-text">{model.name}</span>
                          </label>
                        ))}
                      </div>
                    </div>

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
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ============================================================ */}
      {/* 4. Leakage Control — Hierarchical Collapsible Groups         */}
      {/* ============================================================ */}
      {!loading && currentStep === 'configure' && (
        <Card className="mt-6">
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
            {/* Leakage Scan Button */}
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

            {/* Leakage Scan Results */}
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
                {(['known_leakage', 'high_target_correlation', 'identifier', 'near_zero_variance'] as const).map((reason) => {
                  const group = leakageScanResults.filter((f) => f.reason === reason);
                  if (group.length === 0) return null;
                  const labels: Record<string, string> = {
                    known_leakage: 'Known Leakage',
                    high_target_correlation: 'High Target Correlation',
                    identifier: 'Identifier Columns',
                    near_zero_variance: 'Near-Zero Variance',
                  };
                  const colors: Record<string, string> = {
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

            {/* Search bar */}
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

            <div className="max-h-64 overflow-y-auto">
              {leakageSearchResults ? (
                /* Search mode: flat filtered list with group context */
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
                /* Browse mode: hierarchical group tree */
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
          <CardFooter>
            <Button
              variant="primary"
              className="ml-auto"
              onClick={handleRunAnalysis}
              disabled={!targetColumn}
            >
              <Play className="w-4 h-4" />
              Run Analysis
              <ArrowRight className="w-4 h-4" />
            </Button>
          </CardFooter>
        </Card>
      )}

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
