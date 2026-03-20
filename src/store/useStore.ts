import { create } from 'zustand';
import { projectApi } from '../services/api';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
}

interface ProjectState {
  name: string;
  diseaseContext: string;
  dataPath: string;
  outputPath: string;
  targetColumn: string;
  leakageFeatures: string[];
}

interface DataState {
  loaded: boolean;
  cacheKey: string | null;
  shape: [number, number] | null;
  columns: string[];
  modalities: string[];
}

interface AnalysisState {
  engineeredCacheKey: string | null;
  trainCacheKey: string | null;
  testCacheKey: string | null;
  modelId: string | null;
  selectedFeatures: string[];
  calibratedModelId: string | null;
  ensembleModelId: string | null;
  driftResult: string | null;
}

interface AppStore {
  // Navigation
  currentStep: 'project_hub' | 'data_ingestion' | 'ml_engine' | 'stats_lab' | 'results';
  setCurrentStep: (step: AppStore['currentStep']) => void;
  
  // Project state
  project: ProjectState | null;
  setProject: (project: ProjectState | null) => void;
  updateProject: (updates: Partial<ProjectState>) => void;
  
  // Data state
  data: DataState;
  setData: (data: Partial<DataState>) => void;
  
  // Analysis state
  analysis: AnalysisState;
  setAnalysis: (analysis: Partial<AnalysisState>) => void;
  
  // UI state
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  
  // Task tracking
  runningTasks: Map<string, { status: string; progress: number; message: string }>;
  updateTask: (taskId: string, updates: { status: string; progress: number; message: string }) => void;
  removeTask: (taskId: string) => void;
  
  // Toasts
  toasts: Toast[];
  addToast: (message: string, type: Toast['type']) => void;
  removeToast: (id: string) => void;
  
  // Persistence
  saveProject: () => void;

  // Reset
  reset: () => void;
}

const initialDataState: DataState = {
  loaded: false,
  cacheKey: null,
  shape: null,
  columns: [],
  modalities: [],
};

const initialAnalysisState: AnalysisState = {
  engineeredCacheKey: null,
  trainCacheKey: null,
  testCacheKey: null,
  modelId: null,
  selectedFeatures: [],
  calibratedModelId: null,
  ensembleModelId: null,
  driftResult: null,
};

export const useStore = create<AppStore>((set, get) => ({
  // Navigation
  currentStep: 'project_hub',
  setCurrentStep: (step) => set({ currentStep: step }),
  
  // Project
  project: null,
  setProject: (project) => {
    set({ project });
    if (project) setTimeout(() => get().saveProject(), 0);
  },
  updateProject: (updates) => {
    set((state) => ({
      project: state.project ? { ...state.project, ...updates } : null,
    }));
    setTimeout(() => get().saveProject(), 0);
  },

  // Data
  data: initialDataState,
  setData: (data) => {
    set((state) => ({
      data: { ...state.data, ...data },
    }));
    setTimeout(() => get().saveProject(), 0);
  },

  // Analysis
  analysis: initialAnalysisState,
  setAnalysis: (analysis) => {
    set((state) => ({
      analysis: { ...state.analysis, ...analysis },
    }));
    setTimeout(() => get().saveProject(), 0);
  },
  
  // UI
  sidebarCollapsed: false,
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  
  // Tasks
  runningTasks: new Map(),
  updateTask: (taskId, updates) => set((state) => {
    const newTasks = new Map(state.runningTasks);
    newTasks.set(taskId, updates);
    return { runningTasks: newTasks };
  }),
  removeTask: (taskId) => set((state) => {
    const newTasks = new Map(state.runningTasks);
    newTasks.delete(taskId);
    return { runningTasks: newTasks };
  }),
  
  // Toasts
  toasts: [],
  addToast: (message, type) => {
    const id = Math.random().toString(36).substring(7);
    set((state) => ({
      toasts: [...state.toasts, { id, message, type }],
    }));
    // Auto-remove after 5 seconds
    setTimeout(() => {
      get().removeToast(id);
    }, 5000);
  },
  removeToast: (id) => set((state) => ({
    toasts: state.toasts.filter((t) => t.id !== id),
  })),
  
  // Persistence — sync frontend state to backend, which auto-saves to .pie
  saveProject: () => {
    const state = get();
    if (!state.project) return;
    projectApi.updateState({
      config: {
        target_column: state.project.targetColumn,
        leakage_features: state.project.leakageFeatures,
      },
      data: {
        loaded: state.data.loaded,
        cache_key: state.data.cacheKey,
        shape: state.data.shape,
        columns: state.data.columns,
        modalities: state.data.modalities,
      },
      analysis: {
        engineered_cache_key: state.analysis.engineeredCacheKey,
        train_cache_key: state.analysis.trainCacheKey,
        test_cache_key: state.analysis.testCacheKey,
        model_id: state.analysis.modelId,
        selected_features: state.analysis.selectedFeatures,
        calibrated_model_id: state.analysis.calibratedModelId,
        ensemble_model_id: state.analysis.ensembleModelId,
        drift_result: state.analysis.driftResult,
      },
      current_step: state.currentStep,
    }).catch(() => {}); // Best-effort
  },

  // Reset
  reset: () => set({
    currentStep: 'project_hub',
    project: null,
    data: initialDataState,
    analysis: initialAnalysisState,
    runningTasks: new Map(),
  }),
}));

export default useStore;
