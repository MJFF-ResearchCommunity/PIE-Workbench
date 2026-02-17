import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Trophy, 
  Download, 
  BarChart2,
  Target,
  Layers,
  ExternalLink,
  FileJson,
  Share2,
  RefreshCw
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
  PieChart,
  Pie
} from 'recharts';

interface FeatureImportance {
  features: string[];
  importances: number[];
}

interface ComparisonResult {
  Model: string;
  Accuracy: number;
  AUC: number;
  Recall: number;
  Prec: number;
  F1: number;
}

export default function Results() {
  const navigate = useNavigate();
  const { project, analysis, addToast } = useStore();
  
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!project || !analysis.modelId) {
      navigate('/ml');
      return;
    }
    loadFeatureImportance();
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

  // Sample metrics for display (in real implementation, these would come from the API)
  const metrics = [
    { name: 'Accuracy', value: 0.87, color: '#ff6b4a' },
    { name: 'AUC', value: 0.92, color: '#4ecdc4' },
    { name: 'Precision', value: 0.85, color: '#fbbf24' },
    { name: 'Recall', value: 0.83, color: '#4ade80' },
    { name: 'F1 Score', value: 0.84, color: '#a78bfa' },
  ];

  // Prepare feature importance chart data
  const featureChartData = featureImportance
    ? featureImportance.features.map((name, i) => ({
        name: name.length > 20 ? name.substring(0, 20) + '...' : name,
        fullName: name,
        importance: featureImportance.importances[i],
      })).reverse()
    : [];

  // Sample confusion matrix
  const confusionMatrix = [
    { predicted: 'PD', actual: 'PD', value: 145 },
    { predicted: 'PD', actual: 'Control', value: 12 },
    { predicted: 'Control', actual: 'PD', value: 18 },
    { predicted: 'Control', actual: 'Control', value: 125 },
  ];

  const handleExportResults = () => {
    addToast('Exporting results...', 'info');
    // In real implementation, this would download the results
  };

  const handleViewReport = () => {
    if (project?.outputPath) {
      window.open(`file://${project.outputPath}/classification/classification_report.html`, '_blank');
    }
  };

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
          <Button variant="secondary" onClick={handleViewReport}>
            <ExternalLink className="w-4 h-4" />
            View Full Report
          </Button>
          <Button variant="primary" onClick={handleExportResults}>
            <Download className="w-4 h-4" />
            Export Results
          </Button>
        </div>
      </div>

      {/* Top Metrics Row */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card variant="glass" padding="sm">
              <CardContent className="text-center py-4">
                <p className="text-3xl font-bold" style={{ color: metric.color }}>
                  {(metric.value * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-pie-text-muted mt-1">{metric.name}</p>
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
          <Card variant="elevated">
            <CardContent className="py-6">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-pie-accent to-pie-accent-secondary flex items-center justify-center">
                  <Trophy className="w-7 h-7 text-white" />
                </div>
                <div>
                  <p className="text-sm text-pie-text-muted">Best Performing Model</p>
                  <h3 className="text-2xl font-bold text-pie-text">Random Forest</h3>
                  <p className="text-sm text-pie-accent">AUC: 0.92 | Accuracy: 87%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Confusion Matrix */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5 text-pie-accent-secondary" />
                Confusion Matrix
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2">
                {confusionMatrix.map((cell, i) => (
                  <div
                    key={i}
                    className={clsx(
                      'p-4 rounded-lg text-center',
                      cell.predicted === cell.actual
                        ? 'bg-pie-success/20'
                        : 'bg-pie-error/20'
                    )}
                  >
                    <p className="text-2xl font-bold text-pie-text">{cell.value}</p>
                    <p className="text-xs text-pie-text-muted mt-1">
                      Pred: {cell.predicted} / True: {cell.actual}
                    </p>
                  </div>
                ))}
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
        </div>
      </div>

      {/* Model Comparison Table */}
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
                  <th className="text-right py-3 px-4 text-pie-text-muted font-medium">Accuracy</th>
                  <th className="text-right py-3 px-4 text-pie-text-muted font-medium">AUC</th>
                  <th className="text-right py-3 px-4 text-pie-text-muted font-medium">Precision</th>
                  <th className="text-right py-3 px-4 text-pie-text-muted font-medium">Recall</th>
                  <th className="text-right py-3 px-4 text-pie-text-muted font-medium">F1</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { model: 'Random Forest', accuracy: 0.87, auc: 0.92, precision: 0.85, recall: 0.83, f1: 0.84 },
                  { model: 'XGBoost', accuracy: 0.85, auc: 0.90, precision: 0.83, recall: 0.82, f1: 0.82 },
                  { model: 'LightGBM', accuracy: 0.84, auc: 0.89, precision: 0.82, recall: 0.81, f1: 0.81 },
                  { model: 'Logistic Regression', accuracy: 0.78, auc: 0.84, precision: 0.76, recall: 0.75, f1: 0.75 },
                  { model: 'SVM', accuracy: 0.76, auc: 0.82, precision: 0.74, recall: 0.73, f1: 0.73 },
                ].map((row, i) => (
                  <tr 
                    key={row.model}
                    className={clsx(
                      'border-b border-pie-border/50 transition-colors',
                      i === 0 ? 'bg-pie-accent/10' : 'hover:bg-pie-surface/50'
                    )}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {i === 0 && <Trophy className="w-4 h-4 text-pie-accent" />}
                        <span className={i === 0 ? 'font-medium text-pie-text' : 'text-pie-text-muted'}>
                          {row.model}
                        </span>
                      </div>
                    </td>
                    <td className="text-right py-3 px-4 font-mono text-sm">
                      {(row.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="text-right py-3 px-4 font-mono text-sm">
                      {(row.auc * 100).toFixed(1)}%
                    </td>
                    <td className="text-right py-3 px-4 font-mono text-sm">
                      {(row.precision * 100).toFixed(1)}%
                    </td>
                    <td className="text-right py-3 px-4 font-mono text-sm">
                      {(row.recall * 100).toFixed(1)}%
                    </td>
                    <td className="text-right py-3 px-4 font-mono text-sm">
                      {(row.f1 * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

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
