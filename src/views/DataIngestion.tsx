import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Database, 
  Check, 
  X, 
  ArrowRight, 
  Loader2, 
  FileSpreadsheet,
  AlertCircle,
  RefreshCw,
  Eye,
  Terminal,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import Button from '../components/ui/Button';
import Progress from '../components/ui/Progress';
import { useStore } from '../store/useStore';
import { dataApi } from '../services/api';
import { clsx } from 'clsx';

interface Modality {
  id: string;
  name: string;
  description: string;
  available: boolean;
  file_count?: number;
}

interface ColumnInfo {
  name: string;
  dtype: string;
  null_count: number;
  null_pct: number;
}

interface LogEntry {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error' | 'progress';
}

export default function DataIngestion() {
  const navigate = useNavigate();
  const { project, setData, data, addToast, updateTask, removeTask } = useStore();
  
  const [modalities, setModalities] = useState<Modality[]>([]);
  const [selectedModalities, setSelectedModalities] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<{ status: string; progress: number; message: string } | null>(null);
  const [previewData, setPreviewData] = useState<{ columns: ColumnInfo[]; row_count: number } | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [showLogs, setShowLogs] = useState(true);
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log panel to bottom when new entries arrive
  useEffect(() => {
    if (logEndRef.current && showLogs) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, showLogs]);

  const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, type }]);
  }, []);

  useEffect(() => {
    if (!project) {
      navigate('/');
      return;
    }
    detectModalities();
  }, [project, navigate]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (taskId && taskStatus?.status === 'running') {
      interval = setInterval(pollTaskStatus, 2000);
    }
    return () => clearInterval(interval);
  }, [taskId, taskStatus?.status]);

  const detectModalities = async () => {
    if (!project?.dataPath) return;
    
    setDetecting(true);
    addLog(`Scanning data directory: ${project.dataPath}`, 'info');
    try {
      // Get modality definitions
      const defsResponse = await dataApi.getModalities();
      const definitions = defsResponse.data.modalities;
      
      // Detect available modalities in the data path
      addLog('Detecting available modalities...', 'info');
      const detectResponse = await dataApi.detectModalities(project.dataPath);
      const detected = detectResponse.data.modalities;
      
      // Merge definitions with detected info
      const merged = definitions.map((def: Modality) => {
        const found = detected.find((d: { id: string }) => d.id === def.id);
        return {
          ...def,
          available: found?.available || false,
          file_count: found?.file_count || 0,
        };
      });
      
      setModalities(merged);
      
      // Auto-select available modalities
      const availableIds = merged.filter((m: Modality) => m.available).map((m: Modality) => m.id);
      setSelectedModalities(new Set(availableIds));
      
      // Log detected modalities
      const availableCount = merged.filter((m: Modality) => m.available).length;
      addLog(`Found ${availableCount} available modalities`, 'success');
      merged.forEach((m: Modality) => {
        if (m.available) {
          addLog(`  ✓ ${m.name}: ${m.file_count} files`, 'success');
        }
      });
    } catch (error) {
      addLog('Failed to detect modalities', 'error');
      addToast('Failed to detect modalities. Check the data path.', 'error');
      console.error(error);
    } finally {
      setDetecting(false);
    }
  };

  const toggleModality = (id: string) => {
    const newSelected = new Set(selectedModalities);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedModalities(newSelected);
  };

  const handleLoadData = async () => {
    if (!project?.dataPath || selectedModalities.size === 0) {
      addToast('Please select at least one modality', 'error');
      return;
    }

    setLoading(true);
    setLogs([]); // Clear previous logs
    lastLogIndexRef.current = 0; // Reset backend log cursor
    addLog('Starting data loading process...', 'info');
    addLog(`Selected modalities: ${Array.from(selectedModalities).join(', ')}`, 'info');
    
    try {
      const response = await dataApi.load({
        data_path: project.dataPath,
        modalities: Array.from(selectedModalities),
        merge_output: true,
        clean_data: true,
      });

      addLog(`Task started with ID: ${response.data.task_id.slice(0, 8)}...`, 'info');
      setTaskId(response.data.task_id);
      setTaskStatus({ status: 'running', progress: 0, message: 'Starting...' });
      updateTask(response.data.task_id, { status: 'running', progress: 0, message: 'Starting...' });
    } catch (error) {
      addLog('Failed to start data loading task', 'error');
      addToast('Failed to start data loading', 'error');
      setLoading(false);
      console.error(error);
    }
  };

  // Track how many backend log entries we've already shown.
  // Using a ref instead of state to avoid stale closures in the setInterval callback —
  // useState would capture the initial value (0) in the closure, causing ALL logs to be
  // re-added on every poll, leading to unbounded memory growth and eventual OOM crash.
  const lastLogIndexRef = useRef(0);

  const pollTaskStatus = useCallback(async () => {
    if (!taskId) return;

    try {
      const response = await dataApi.getStatus(taskId);
      const status = response.data;

      // Ingest any new log entries from the backend
      const backendLogs: { timestamp: string; message: string }[] = status.logs || [];
      if (backendLogs.length > lastLogIndexRef.current) {
        const newEntries = backendLogs.slice(lastLogIndexRef.current);
        for (const entry of newEntries) {
          // Determine log type from message content
          let type: LogEntry['type'] = 'progress';
          if (entry.message.includes('[WARNING]') || entry.message.includes('[ERROR]') || entry.message.startsWith('WARNING')) {
            type = 'error';
          } else if (entry.message.includes('[INFO]') || entry.message.startsWith('  Found') || entry.message.startsWith('    -')) {
            type = 'info';
          } else if (entry.message.includes('loaded:') || entry.message.includes('successfully') || entry.message.includes('complete')) {
            type = 'success';
          } else if (entry.message.startsWith('Failed') || entry.message.startsWith('PIE-clean import failed')) {
            type = 'error';
          }
          addLog(entry.message, type);
        }
        lastLogIndexRef.current = backendLogs.length;
      }

      setTaskStatus(status);
      updateTask(taskId, { status: status.status, progress: status.progress * 100, message: status.message });

      if (status.status === 'completed') {
        setLoading(false);
        addLog('Data loading completed successfully!', 'success');
        setData({
          loaded: true,
          cacheKey: status.result.cache_key,
          shape: status.result.shape || null,
          columns: status.result.columns || [],
          modalities: Array.from(selectedModalities),
        });
        removeTask(taskId);
        addToast('Data loaded successfully!', 'success');

        // Load preview
        addLog('Loading data preview...', 'info');
        await loadPreview(status.result.cache_key);
        addLog('Preview loaded', 'success');
      } else if (status.status === 'failed') {
        setLoading(false);
        addLog(`Error: ${status.error}`, 'error');
        removeTask(taskId);
        addToast(`Data loading failed: ${status.error}`, 'error');
      }
    } catch (error) {
      console.error('Failed to poll task status:', error);
    }
  }, [taskId, selectedModalities, setData, addToast, updateTask, removeTask, addLog]);

  const loadPreview = async (cacheKey: string) => {
    try {
      const response = await dataApi.getColumns(cacheKey);
      setPreviewData({
        columns: response.data.columns,
        row_count: response.data.total_rows,
      });
    } catch (error) {
      console.error('Failed to load preview:', error);
    }
  };

  const handleContinue = () => {
    navigate('/ml');
  };

  if (!project) {
    return null;
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold text-pie-text mb-2">
          Data Ingestion
        </h1>
        <p className="text-pie-text-muted">
          Load and visualize your PPMI data. Select the modalities you want to include in your analysis.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Left Panel - Modality Selection */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5 text-pie-accent" />
                  Data Modalities
                </CardTitle>
                <CardDescription>Select the data types to include</CardDescription>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={detectModalities}
                disabled={detecting}
              >
                <RefreshCw className={clsx('w-4 h-4', detecting && 'animate-spin')} />
              </Button>
            </div>
          </CardHeader>

          <CardContent>
            {detecting ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-pie-accent" />
              </div>
            ) : (
              <div className="space-y-3">
                {modalities.map((modality) => (
                  <motion.button
                    key={modality.id}
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.99 }}
                    onClick={() => modality.available && toggleModality(modality.id)}
                    disabled={!modality.available}
                    className={clsx(
                      'w-full p-4 rounded-lg border text-left transition-all duration-200',
                      modality.available
                        ? selectedModalities.has(modality.id)
                          ? 'bg-pie-accent/10 border-pie-accent/50'
                          : 'bg-pie-surface border-pie-border hover:border-pie-text-muted'
                        : 'bg-pie-surface/50 border-pie-border/50 opacity-50 cursor-not-allowed'
                    )}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={clsx(
                          'w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 mt-0.5',
                          selectedModalities.has(modality.id)
                            ? 'bg-pie-accent border-pie-accent'
                            : 'border-pie-border'
                        )}
                      >
                        {selectedModalities.has(modality.id) && (
                          <Check className="w-3 h-3 text-white" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-pie-text">
                            {modality.name}
                          </span>
                          {modality.available ? (
                            <span className="text-xs px-2 py-0.5 rounded-full bg-pie-success/20 text-pie-success">
                              {modality.file_count} files
                            </span>
                          ) : (
                            <span className="text-xs px-2 py-0.5 rounded-full bg-pie-error/20 text-pie-error">
                              Not found
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-pie-text-muted mt-1">
                          {modality.description}
                        </p>
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            )}

            {/* Load Button */}
            <div className="mt-6 pt-6 border-t border-pie-border">
              {taskStatus && taskStatus.status === 'running' ? (
                <div className="space-y-3">
                  <Progress
                    value={taskStatus.progress * 100}
                    variant="gradient"
                    showLabel
                  />
                  <p className="text-sm text-pie-text-muted text-center">
                    {taskStatus.message}
                  </p>
                </div>
              ) : (
                <Button
                  variant="primary"
                  className="w-full"
                  onClick={handleLoadData}
                  disabled={selectedModalities.size === 0 || loading}
                  loading={loading}
                >
                  <FileSpreadsheet className="w-4 h-4" />
                  Merge & Process Data
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Right Panel - Data Preview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="w-5 h-5 text-pie-accent-secondary" />
              Data Health Check
            </CardTitle>
            <CardDescription>
              Preview your loaded data and check for missing values
            </CardDescription>
          </CardHeader>

          <CardContent>
            {data.loaded && previewData ? (
              <div className="space-y-4">
                {/* Summary Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-lg bg-pie-surface">
                    <p className="text-2xl font-bold text-pie-accent">
                      {previewData.row_count.toLocaleString()}
                    </p>
                    <p className="text-sm text-pie-text-muted">Total Rows</p>
                  </div>
                  <div className="p-4 rounded-lg bg-pie-surface">
                    <p className="text-2xl font-bold text-pie-accent-secondary">
                      {previewData.columns.length}
                    </p>
                    <p className="text-sm text-pie-text-muted">Columns</p>
                  </div>
                </div>

                {/* Column List with Missingness */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium text-pie-text">Column Missingness</h4>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowPreview(!showPreview)}
                    >
                      {showPreview ? 'Hide' : 'Show All'}
                    </Button>
                  </div>
                  
                  <div className={clsx(
                    'space-y-1 overflow-y-auto',
                    showPreview ? 'max-h-96' : 'max-h-48'
                  )}>
                    {previewData.columns
                      .sort((a, b) => b.null_pct - a.null_pct)
                      .slice(0, showPreview ? undefined : 10)
                      .map((col) => (
                        <div
                          key={col.name}
                          className="flex items-center gap-3 p-2 rounded bg-pie-surface/50"
                        >
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-pie-text truncate">{col.name}</p>
                          </div>
                          <div className="w-24">
                            <div className="h-2 bg-pie-bg rounded-full overflow-hidden">
                              <div
                                className={clsx(
                                  'h-full rounded-full',
                                  col.null_pct > 50 ? 'bg-pie-error' :
                                  col.null_pct > 20 ? 'bg-pie-warning' :
                                  'bg-pie-success'
                                )}
                                style={{ width: `${100 - col.null_pct}%` }}
                              />
                            </div>
                          </div>
                          <span className={clsx(
                            'text-xs font-mono w-12 text-right',
                            col.null_pct > 50 ? 'text-pie-error' :
                            col.null_pct > 20 ? 'text-pie-warning' :
                            'text-pie-success'
                          )}>
                            {col.null_pct.toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Continue Button */}
                <Button
                  variant="primary"
                  className="w-full mt-4"
                  onClick={handleContinue}
                >
                  Continue to ML Engine
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <AlertCircle className="w-12 h-12 text-pie-text-muted mb-4" />
                <p className="text-pie-text-muted">
                  Load data to see the health check preview
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Path Info */}
      <div className="mt-6 p-4 rounded-lg bg-pie-surface/50 border border-pie-border/50">
        <p className="text-sm text-pie-text-muted">
          <span className="font-medium text-pie-text">Data Path:</span>{' '}
          <code className="font-mono text-pie-accent-secondary">{project.dataPath}</code>
        </p>
      </div>

      {/* Terminal Log Panel */}
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
                      <span className={clsx(
                        log.type === 'success' && 'text-green-400',
                        log.type === 'error' && 'text-red-400',
                        log.type === 'progress' && 'text-blue-400',
                        log.type === 'info' && 'text-gray-300'
                      )}>
                        {log.type === 'progress' && '⟳ '}
                        {log.type === 'success' && '✓ '}
                        {log.type === 'error' && '✗ '}
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
    </div>
  );
}
