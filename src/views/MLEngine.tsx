import { useState, useEffect, useCallback } from 'react';
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
  Settings
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
}

export default function MLEngine() {
  const navigate = useNavigate();
  const { project, data, analysis, setAnalysis, addToast, updateTask, removeTask } = useStore();
  
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [targetColumn, setTargetColumn] = useState('');
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [fsMethod, setFsMethod] = useState('fdr');
  const [fsParam, setFsParam] = useState(0.05);
  const [leakageFeatures, setLeakageFeatures] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState<'autopilot' | 'expert'>('autopilot');
  const [nModels, setNModels] = useState(5);
  const [tuneBest, setTuneBest] = useState(false);
  const [timeBudget, setTimeBudget] = useState(30);
  
  const [currentStep, setCurrentStep] = useState<'configure' | 'feature_engineering' | 'feature_selection' | 'training'>('configure');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<{ status: string; progress: number; message: string } | null>(null);
  const [loading, setLoading] = useState(false);

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

  const loadColumns = async () => {
    if (!data.cacheKey) return;
    
    try {
      const response = await dataApi.getColumns(data.cacheKey);
      setColumns(response.data.columns);
    } catch (error) {
      console.error('Failed to load columns:', error);
    }
  };

  const handleTargetChange = async (value: string) => {
    setTargetColumn(value);
    
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

  const toggleLeakageFeature = (column: string) => {
    const newSet = new Set(leakageFeatures);
    if (newSet.has(column)) {
      newSet.delete(column);
    } else {
      newSet.add(column);
    }
    setLeakageFeatures(newSet);
  };

  const runFeatureEngineering = async () => {
    if (!data.cacheKey) return;
    
    setLoading(true);
    setCurrentStep('feature_engineering');
    
    try {
      const response = await analysisApi.featureEngineering({
        cache_key: data.cacheKey,
        scale_numeric: true,
        one_hot_encode: true,
      });
      
      setTaskId(response.data.task_id);
      setTaskStatus({ status: 'running', progress: 0, message: 'Engineering features...' });
      updateTask(response.data.task_id, { status: 'running', progress: 0, message: 'Engineering features...' });
    } catch (error) {
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
      const response = await analysisApi.train({
        train_cache_key: trainKey,
        test_cache_key: testKey,
        target_column: targetColumn,
        task_type: taskType,
        n_models: nModels,
        tune_best: tuneBest,
        time_budget_minutes: timeBudget,
      });
      
      setTaskId(response.data.task_id);
      setTaskStatus({ status: 'running', progress: 0, message: 'Training models...' });
      updateTask(response.data.task_id, { status: 'running', progress: 0, message: 'Training models...' });
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
      
      setTaskStatus(status);
      updateTask(taskId, { status: status.status, progress: status.progress * 100, message: status.message });

      if (status.status === 'completed') {
        removeTask(taskId);
        
        if (currentStep === 'feature_engineering') {
          setAnalysis({ engineeredCacheKey: status.result.cache_key });
          addToast('Feature engineering completed!', 'success');
          await runFeatureSelection(status.result.cache_key);
        } else if (currentStep === 'feature_selection') {
          setAnalysis({
            trainCacheKey: status.result.train_cache_key,
            testCacheKey: status.result.test_cache_key,
            selectedFeatures: status.result.selected_feature_names || [],
          });
          addToast(`Selected ${status.result.selected_features} features!`, 'success');
          await runModelTraining(status.result.train_cache_key, status.result.test_cache_key);
        } else if (currentStep === 'training') {
          setAnalysis({ modelId: status.result.model_id });
          addToast('Model training completed!', 'success');
          setLoading(false);
          navigate('/results');
        }
      } else if (status.status === 'failed') {
        setLoading(false);
        setCurrentStep('configure');
        removeTask(taskId);
        addToast(`Failed: ${status.error}`, 'error');
      }
    } catch (error) {
      console.error('Failed to poll task status:', error);
    }
  }, [taskId, currentStep, setAnalysis, addToast, updateTask, removeTask, navigate, runFeatureSelection, runModelTraining]);

  const handleRunAnalysis = () => {
    if (!targetColumn) {
      addToast('Please select a target column', 'error');
      return;
    }
    runFeatureEngineering();
  };

  const targetOptions = columns
    .filter((c) => c.is_categorical || (c.is_numeric && c.unique_count <= 20))
    .map((c) => ({
      value: c.name,
      label: `${c.name} (${c.unique_count} unique)`,
    }));

  const featureColumns = columns.filter((c) => c.name !== targetColumn);

  const steps = [
    { id: 'configure', label: 'Configure', icon: Settings },
    { id: 'feature_engineering', label: 'Engineering', icon: Sliders },
    { id: 'feature_selection', label: 'Selection', icon: Shield },
    { id: 'training', label: 'Training', icon: Brain },
  ];

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
                {currentStep === 'training' && 'Training Models...'}
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
            {/* Target Selection */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-pie-accent" />
                  Target Variable
                </CardTitle>
                <CardDescription>Select what you want to predict</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Select
                  label="Target Column"
                  value={targetColumn}
                  onChange={(e) => handleTargetChange(e.target.value)}
                  options={targetOptions}
                  placeholder="Select target..."
                />
                
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

            {/* Feature Selection Config */}
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
                    { value: 'fdr', label: 'False Discovery Rate (FDR)' },
                    { value: 'k_best', label: 'K-Best Features' },
                    { value: 'rfe', label: 'Recursive Feature Elimination' },
                  ]}
                />
                
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
              </CardContent>
            </Card>

            {/* Model Arena */}
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
                </div>

                {mode === 'expert' && (
                  <div className="space-y-4">
                    <Select
                      label="Models to Compare"
                      value={nModels.toString()}
                      onChange={(e) => setNModels(parseInt(e.target.value))}
                      options={[
                        { value: '3', label: '3 models' },
                        { value: '5', label: '5 models' },
                        { value: '10', label: '10 models' },
                      ]}
                    />
                    
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

      {/* Leakage Control Panel */}
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
            <div className="grid grid-cols-4 gap-2 max-h-48 overflow-y-auto">
              {featureColumns.map((col) => (
                <button
                  key={col.name}
                  onClick={() => toggleLeakageFeature(col.name)}
                  className={clsx(
                    'px-3 py-2 rounded-lg text-sm text-left transition-all',
                    leakageFeatures.has(col.name)
                      ? 'bg-pie-warning/20 text-pie-warning border border-pie-warning/50'
                      : 'bg-pie-surface text-pie-text-muted hover:text-pie-text'
                  )}
                >
                  <span className="truncate block">{col.name}</span>
                </button>
              ))}
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
    </div>
  );
}
