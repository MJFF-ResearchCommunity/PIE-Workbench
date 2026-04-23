import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8100/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000, // 30 seconds for normal requests
  headers: {
    'Content-Type': 'application/json',
  },
});

// No timeout for ML pipeline endpoints — user cancels manually via Stop button
const longApi = axios.create({
  baseURL: API_BASE,
  timeout: 0,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Project API
export const projectApi = {
  create: (config: ProjectConfig) => api.post('/project/create', config),
  getCurrent: () => api.get('/project/current'),
  save: (filePath: string) => api.post(`/project/save?file_path=${encodeURIComponent(filePath)}`),
  load: (filePath: string) => api.post(`/project/load?file_path=${encodeURIComponent(filePath)}`),
  updateState: (updates: Record<string, unknown>) => api.post('/project/update_state', updates),
  getDiseaseContexts: () => api.get('/project/disease_contexts'),
  getRecent: () => api.get('/project/recent'),
};

// Data API
export const dataApi = {
  getModalities: () => api.get('/data/modalities'),
  detectModalities: (dataPath: string) => api.post(`/data/detect_modalities?data_path=${encodeURIComponent(dataPath)}`),
  preview: (dataPath: string, modality?: string, limit?: number) => 
    api.post(`/data/preview?data_path=${encodeURIComponent(dataPath)}&modality=${modality || ''}&limit=${limit || 50}`),
  load: (request: DataLoadRequest) => longApi.post('/data/load', request),
  getStatus: (taskId: string) => api.get(`/data/status/${taskId}`),
  getColumns: (cacheKey: string) => api.get(`/data/columns?cache_key=${cacheKey}`),
  getMissingnessHeatmap: (cacheKey: string, sampleSize?: number) => 
    api.post(`/data/missingness_heatmap?cache_key=${cacheKey}&sample_size=${sampleSize || 100}`),
};

// Analysis API
export const analysisApi = {
  getTaskTypes: () => api.get('/analysis/task_types'),
  getFeatureSelectionMethods: () => api.get('/analysis/feature_selection_methods'),
  getAvailableModels: (taskType: string) => api.get(`/analysis/available_models?task_type=${taskType}`),
  suggestTaskType: (cacheKey: string, targetColumn: string) => 
    api.post(`/analysis/suggest_task_type?cache_key=${cacheKey}&target_column=${targetColumn}`),
  featureEngineering: (request: FeatureEngineeringRequest) => longApi.post('/analysis/feature_engineering', request),
  featureSelection: (request: FeatureSelectionRequest) => longApi.post('/analysis/feature_selection', request),
  train: (request: TrainModelRequest) => longApi.post('/analysis/train', request),
  runPipeline: (request: PipelineRequest) => longApi.post('/analysis/run_pipeline', request),
  getTaskStatus: (taskId: string) => api.get(`/analysis/task/${taskId}`),
  cancelTask: (taskId: string) => api.post(`/analysis/task/${taskId}/cancel`),
  getFeatureImportance: (modelId: string, topN?: number) =>
    api.get(`/analysis/model/${modelId}/feature_importance?top_n=${topN || 20}`),
  getModelResults: (modelId: string) => api.get(`/analysis/model/${modelId}/results`),
  getReport: (modelId: string) =>
    longApi.get(`/analysis/model/${modelId}/report`, { responseType: 'text' }),
  getModelStructure: (modelId: string) =>
    api.get(`/analysis/model/${modelId}/structure`),
  getTreeViz: (modelId: string, treeIndex: number = 0) =>
    longApi.get(`/analysis/model/${modelId}/tree_viz?tree_index=${treeIndex}`, { responseType: 'text' }),
  getBayesianNetworkViz: (modelId: string) =>
    longApi.get(`/analysis/model/${modelId}/bn_viz`, { responseType: 'text' }),
  autoML: (request: AutoMLRequest) => longApi.post('/analysis/auto_ml', request),
  calibrate: (request: CalibrateRequest) => api.post('/analysis/calibrate', request),
  validateDrift: (request: DriftValidationRequest) => api.post('/analysis/validate_drift', request),
  detectLeakage: (request: DetectLeakageRequest) => longApi.post('/analysis/detect_leakage', request),
  createEnsemble: (request: EnsembleRequest) => api.post('/analysis/create_ensemble', request),
};

// Statistics API
export const statsApi = {
  autoTest: (request: StatTestRequest) => api.post('/statistics/auto_test', request),
  ttest: (cacheKey: string, variable: string, groupingVariable: string) =>
    api.post(`/statistics/ttest?cache_key=${cacheKey}&variable=${variable}&grouping_variable=${groupingVariable}`),
  anova: (cacheKey: string, variable: string, groupingVariable: string) =>
    api.post(`/statistics/anova?cache_key=${cacheKey}&variable=${variable}&grouping_variable=${groupingVariable}`),
  correlation: (request: CorrelationRequest) => api.post('/statistics/correlation', request),
  scatter: (request: ScatterRequest) => api.post('/statistics/scatter', request),
  survival: (request: SurvivalAnalysisRequest) => api.post('/statistics/survival', request),
  descriptive: (cacheKey: string, variables: string[]) =>
    api.post(`/statistics/descriptive?cache_key=${cacheKey}`, variables),

  // --- Describe ---
  describeSummary: (cacheKey: string, variables: string[]) =>
    api.post('/statistics/describe/summary', { cache_key: cacheKey, variables }),
  describeNormality: (cacheKey: string, variable: string, test: string = 'shapiro') =>
    api.post('/statistics/describe/normality', { cache_key: cacheKey, variable, test }),
  describeMissingness: (cacheKey: string, variables: string[]) =>
    api.post('/statistics/describe/missingness', { cache_key: cacheKey, variables }),

  // --- Compare ---
  compareTwoGroup: (payload: { cache_key: string; variable: string; grouping_variable: string; test?: string }) =>
    api.post('/statistics/compare/two_group', payload),
  compareMultiGroup: (payload: { cache_key: string; variable: string; grouping_variable: string; test?: string; posthoc?: string | null }) =>
    api.post('/statistics/compare/multi_group', payload),
  compareCategorical: (payload: { cache_key: string; variable_a: string; variable_b: string; test?: string }) =>
    api.post('/statistics/compare/categorical', payload),

  // --- Correlate ---
  correlatePartial: (payload: { cache_key: string; x: string; y: string; covariates: string[]; method?: string }) =>
    api.post('/statistics/correlate/partial', payload),
  correlateMatrix: (payload: { cache_key: string; variables: string[]; method?: string }) =>
    api.post('/statistics/correlate/matrix', payload),

  // --- Regress ---
  regressLinear: (payload: { cache_key: string; outcome: string; predictors: string[]; standardize?: boolean }) =>
    api.post('/statistics/regress/linear', payload),
  regressLogistic: (payload: { cache_key: string; outcome: string; predictors: string[] }) =>
    api.post('/statistics/regress/logistic', payload),
  regressAncova: (payload: { cache_key: string; outcome: string; group: string; covariates: string[] }) =>
    api.post('/statistics/regress/ancova', payload),

  // --- Longitudinal ---
  longitudinalLMM: (payload: { cache_key: string; outcome: string; fixed_effects: string[]; group: string; random_slopes?: string[] | null }) =>
    api.post('/statistics/longitudinal/mixed_model', payload),
  longitudinalChange: (payload: { cache_key: string; subject: string; time: string; outcome: string; baseline_time?: unknown }) =>
    api.post('/statistics/longitudinal/change_from_baseline', payload),

  // --- Survive (new) ---
  surviveKM: (payload: { cache_key: string; time: string; event: string; group?: string | null }) =>
    api.post('/statistics/survive/km', payload),
  surviveLogrank: (payload: { cache_key: string; time: string; event: string; group: string }) =>
    api.post('/statistics/survive/logrank', payload),
  surviveCox: (payload: { cache_key: string; time: string; event: string; covariates: string[] }) =>
    api.post('/statistics/survive/cox', payload),

  // --- Multitest ---
  multitestAdjust: (payload: { p_values: number[]; method?: string; alpha?: number }) =>
    api.post('/statistics/multitest/adjust', payload),

  // --- PD Helpers ---
  pdLedd: (doses_mg: Record<string, number>) =>
    api.post('/statistics/pd/ledd', { doses_mg }),
  pdLeddFactors: () => api.get('/statistics/pd/ledd_factors'),
  pdUpdrs: (payload: { cache_key: string; part1_cols?: string[]; part2_cols?: string[]; part3_cols?: string[]; part4_cols?: string[] }) =>
    api.post('/statistics/pd/updrs', payload),
  pdHoehnYahr: (payload: { cache_key: string; variable: string }) =>
    api.post('/statistics/pd/hoehn_yahr', payload),
};

// Health check
export const healthCheck = () => api.get('/health');

// Types
export interface ProjectConfig {
  name: string;
  disease_context: string;
  data_path: string;
  output_path?: string;
  target_column?: string;
  leakage_features?: string[];
}

export interface DataLoadRequest {
  data_path: string;
  modalities?: string[];
  merge_output?: boolean;
  clean_data?: boolean;
}

export interface FeatureEngineeringRequest {
  cache_key: string;
  target_column?: string;
  scale_numeric?: boolean;
  one_hot_encode?: boolean;
  max_categories?: number;
  min_frequency?: number;
}

export interface FeatureSelectionRequest {
  cache_key: string;
  target_column: string;
  method?: string;
  param_value?: number;
  leakage_features?: string[];
  test_size?: number;
}

export interface TrainModelRequest {
  train_cache_key: string;
  test_cache_key: string;
  target_column: string;
  task_type?: string;
  models_to_compare?: string[];
  n_models?: number;
  tune_best?: boolean;
  time_budget_minutes?: number;
}

export interface PipelineRequest {
  data_path: string;
  output_dir: string;
  target_column: string;
  modalities?: string[];
  leakage_features_path?: string;
  fs_method?: string;
  fs_param?: number;
  n_models?: number;
  tune_best?: boolean;
  generate_plots?: boolean;
  budget_minutes?: number;
}

export interface StatTestRequest {
  cache_key: string;
  x_variable: string;
  y_variable: string;
  grouping_variable?: string;
}

export interface CorrelationRequest {
  cache_key: string;
  variables: string[];
  method?: string;
}

export interface ScatterRequest {
  cache_key: string;
  x_variable: string;
  y_variable: string;
  method?: 'pearson' | 'spearman' | 'kendall';
  max_points?: number;
}

export interface SurvivalAnalysisRequest {
  cache_key: string;
  time_variable: string;
  event_variable: string;
  grouping_variable?: string;
}

export interface AutoMLRequest {
  train_cache_key: string;
  test_cache_key: string;
  target_column: string;
  time_limit?: number;
  presets?: string;
}

export interface CalibrateRequest {
  model_id: string;
  method?: string;
}

export interface DriftValidationRequest {
  train_cache_key: string;
  test_cache_key: string;
}

export interface DetectLeakageRequest {
  cache_key: string;
  target_column: string;
}

export interface EnsembleRequest {
  model_id: string;
  method?: string;
}

export default api;
