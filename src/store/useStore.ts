import { create } from 'zustand';

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
  setProject: (project) => set({ project }),
  updateProject: (updates) => set((state) => ({
    project: state.project ? { ...state.project, ...updates } : null,
  })),
  
  // Data
  data: initialDataState,
  setData: (data) => set((state) => ({
    data: { ...state.data, ...data },
  })),
  
  // Analysis
  analysis: initialAnalysisState,
  setAnalysis: (analysis) => set((state) => ({
    analysis: { ...state.analysis, ...analysis },
  })),
  
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
