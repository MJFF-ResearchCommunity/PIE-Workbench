import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  FolderOpen, 
  Plus, 
  ArrowRight, 
  Database,
  Brain,
  BarChart3,
  Sparkles
} from 'lucide-react';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';
import { useStore } from '../store/useStore';
import { projectApi } from '../services/api';

// Declare Electron API types
declare global {
  interface Window {
    electronAPI?: {
      selectDirectory: () => Promise<string | null>;
      selectFile: (options?: { filters?: Array<{ name: string; extensions: string[] }> }) => Promise<string | null>;
    };
  }
}

export default function ProjectHub() {
  const navigate = useNavigate();
  const { setProject, addToast } = useStore();
  
  const [mode, setMode] = useState<'landing' | 'create' | 'open'>('landing');
  const [diseaseContexts, setDiseaseContexts] = useState<Array<{ id: string; name: string; available: boolean }>>([
    { id: 'parkinsons', name: "Parkinson's Disease (PPMI)", available: true },
    { id: 'alzheimers', name: "Alzheimer's Disease", available: false },
  ]);
  const [loading, setLoading] = useState(false);
  
  // Form state
  const [projectName, setProjectName] = useState('');
  const [diseaseContext, setDiseaseContext] = useState('parkinsons');
  const [dataPath, setDataPath] = useState('');
  
  // Check if running in Electron
  const isElectron = Boolean(window.electronAPI);

  useEffect(() => {
    loadDiseaseContexts();
  }, []);

  const loadDiseaseContexts = async () => {
    try {
      const response = await projectApi.getDiseaseContexts();
      setDiseaseContexts(response.data.contexts);
    } catch {
      // Use default if API unavailable
      setDiseaseContexts([
        { id: 'parkinsons', name: "Parkinson's Disease (PPMI)", available: true },
        { id: 'alzheimers', name: "Alzheimer's Disease", available: false },
      ]);
    }
  };

  const handleSelectDirectory = async () => {
    if (window.electronAPI) {
      try {
        const path = await window.electronAPI.selectDirectory();
        if (path) setDataPath(path);
      } catch (error) {
        console.error('Failed to open directory picker:', error);
        addToast('Failed to open file browser. Please enter the path manually.', 'error');
      }
    } else {
      // Fallback for browser development - prompt user
      addToast('File browser not available in web mode. Please enter the path manually.', 'info');
      // Focus the input field
      const input = document.querySelector('input[placeholder="/path/to/PPMI"]') as HTMLInputElement;
      if (input) {
        input.focus();
        input.select();
      }
    }
  };

  const handleCreateProject = async () => {
    if (!projectName || !dataPath) {
      addToast('Please fill in all required fields', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await projectApi.create({
        name: projectName,
        disease_context: diseaseContext,
        data_path: dataPath,
      });

      setProject({
        name: projectName,
        diseaseContext,
        dataPath,
        outputPath: response.data.project.config.output_path,
        targetColumn: '',
        leakageFeatures: [],
      });

      addToast('Project created successfully!', 'success');
      navigate('/data');
    } catch (error) {
      addToast('Failed to create project. Make sure the backend is running.', 'error');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenProject = async () => {
    if (window.electronAPI) {
      const path = await window.electronAPI.selectFile({
        filters: [{ name: 'PIE Project', extensions: ['pie', 'json'] }],
      });
      if (path) {
        try {
          setLoading(true);
          const response = await projectApi.load(path);
          setProject({
            name: response.data.project.config.name,
            diseaseContext: response.data.project.config.disease_context,
            dataPath: response.data.project.config.data_path,
            outputPath: response.data.project.config.output_path,
            targetColumn: response.data.project.config.target_column || '',
            leakageFeatures: response.data.project.config.leakage_features || [],
          });
          addToast('Project loaded successfully!', 'success');
          navigate('/data');
        } catch {
          addToast('Failed to load project', 'error');
        } finally {
          setLoading(false);
        }
      }
    }
  };

  const features = [
    {
      icon: Database,
      title: 'Data Ingestion',
      description: 'Load and visualize PPMI data with automatic modality detection',
    },
    {
      icon: Brain,
      title: 'ML Engine',
      description: 'Train and compare machine learning models with visual pipelines',
    },
    {
      icon: BarChart3,
      title: 'Statistics Lab',
      description: 'Perform statistical tests and survival analysis',
    },
  ];

  if (mode === 'landing') {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-8">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="w-24 h-24 mx-auto mb-6 rounded-2xl overflow-hidden shadow-lg shadow-pie-accent/20">
            <img src="/icon.png" alt="PIE Workbench" className="w-full h-full object-cover" />
          </div>
          <h1 className="font-display text-5xl font-bold mb-4">
            <span className="gradient-text">PIE Workbench</span>
          </h1>
          <p className="text-xl text-pie-text-muted max-w-2xl mx-auto">
            The Parkinson's Insight Engine GUI — A powerful, visual interface for 
            clinical research and machine learning on PPMI data.
          </p>
        </motion.div>

        {/* Action Cards */}
        <div className="flex gap-6 mb-16">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <Card
              variant="elevated"
              className="w-72 cursor-pointer hover:border-pie-accent/50 hover:shadow-pie-accent/10"
              onClick={() => setMode('create')}
            >
              <CardContent className="text-center py-8">
                <div className="w-14 h-14 mx-auto mb-4 rounded-xl bg-pie-accent/20 flex items-center justify-center">
                  <Plus className="w-7 h-7 text-pie-accent" />
                </div>
                <h3 className="font-display text-xl font-semibold text-pie-text mb-2">
                  New Analysis
                </h3>
                <p className="text-sm text-pie-text-muted">
                  Start a fresh project with your data
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.3 }}
          >
            <Card
              variant="elevated"
              className="w-72 cursor-pointer hover:border-pie-accent-secondary/50 hover:shadow-pie-accent-secondary/10"
              onClick={handleOpenProject}
            >
              <CardContent className="text-center py-8">
                <div className="w-14 h-14 mx-auto mb-4 rounded-xl bg-pie-accent-secondary/20 flex items-center justify-center">
                  <FolderOpen className="w-7 h-7 text-pie-accent-secondary" />
                </div>
                <h3 className="font-display text-xl font-semibold text-pie-text mb-2">
                  Open Project
                </h3>
                <p className="text-sm text-pie-text-muted">
                  Continue with an existing project
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.4 }}
          className="grid grid-cols-3 gap-6 max-w-4xl"
        >
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={index}
                className="p-6 rounded-xl bg-pie-surface/50 border border-pie-border/50 text-center"
              >
                <Icon className="w-8 h-8 mx-auto mb-3 text-pie-accent-secondary" />
                <h4 className="font-medium text-pie-text mb-2">{feature.title}</h4>
                <p className="text-sm text-pie-text-muted">{feature.description}</p>
              </div>
            );
          })}
        </motion.div>

        {/* Version info */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-12 text-sm text-pie-text-muted flex items-center gap-2"
        >
          <Sparkles className="w-4 h-4" />
          Version 1.0.0 — Powered by PIE & PIE-clean
        </motion.p>
      </div>
    );
  }

  if (mode === 'create') {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="w-full max-w-xl"
        >
          <Card variant="elevated" padding="lg">
            <CardHeader>
              <CardTitle className="text-2xl font-display">Create New Analysis</CardTitle>
              <CardDescription>
                Configure your project settings to get started
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              <Input
                label="Project Name"
                placeholder="e.g., UPDRS Prediction Study"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
              />

              <Select
                label="Disease Context"
                value={diseaseContext}
                onChange={(e) => setDiseaseContext(e.target.value)}
                options={diseaseContexts.map((ctx) => ({
                  value: ctx.id,
                  label: ctx.name,
                  disabled: !ctx.available,
                }))}
              />

              <div className="space-y-2">
                <label className="block text-sm font-medium text-pie-text">
                  Data Directory
                </label>
                <div className="flex gap-3">
                  <Input
                    placeholder="/path/to/PPMI"
                    value={dataPath}
                    onChange={(e) => setDataPath(e.target.value)}
                    className="flex-1"
                  />
                  {isElectron ? (
                    <Button
                      variant="secondary"
                      onClick={handleSelectDirectory}
                      className="flex-shrink-0"
                    >
                      <FolderOpen className="w-4 h-4" />
                      Browse
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      onClick={handleSelectDirectory}
                      className="flex-shrink-0 opacity-60"
                      title="File browser only available in desktop app"
                    >
                      <FolderOpen className="w-4 h-4" />
                      Browse
                    </Button>
                  )}
                </div>
                <p className="text-sm text-pie-text-muted">
                  {isElectron 
                    ? 'Select the folder containing your raw PPMI data'
                    : 'Enter the full path to your PPMI data folder (e.g., /home/user/PPMI)'
                  }
                </p>
              </div>

              <div className="flex gap-3 pt-4">
                <Button
                  variant="ghost"
                  onClick={() => setMode('landing')}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  onClick={handleCreateProject}
                  loading={loading}
                  className="flex-1"
                >
                  Create Project
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  return null;
}
