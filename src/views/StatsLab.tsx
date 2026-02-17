import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  ArrowRight,
  Plus,
  Activity,
  TrendingUp,
  Clock,
  Loader2
} from 'lucide-react';
import Card, { CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import Button from '../components/ui/Button';
import Select from '../components/ui/Select';
import { useStore } from '../store/useStore';
import { statsApi, dataApi } from '../services/api';
import { clsx } from 'clsx';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

interface ColumnInfo {
  name: string;
  dtype: string;
  is_numeric: boolean;
  is_categorical: boolean;
}

interface StatResult {
  test_name: string;
  description: string;
  p_value: number;
  significant: boolean;
  interpretation: string;
  statistic?: number;
  correlation?: number;
  group_statistics?: Record<string, unknown>;
}

interface SurvivalCurve {
  group: string;
  timeline: number[];
  survival: number[];
  median_survival: number | null;
}

export default function StatsLab() {
  const navigate = useNavigate();
  const { project, data, addToast } = useStore();
  
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [activeTab, setActiveTab] = useState<'comparison' | 'correlation' | 'survival'>('comparison');
  
  // Comparison state
  const [xVariable, setXVariable] = useState('');
  const [yVariable, setYVariable] = useState('');
  const [statResult, setStatResult] = useState<StatResult | null>(null);
  
  // Survival state
  const [timeVariable, setTimeVariable] = useState('');
  const [eventVariable, setEventVariable] = useState('');
  const [groupVariable, setGroupVariable] = useState('');
  const [survivalCurves, setSurvivalCurves] = useState<SurvivalCurve[]>([]);
  const [survivalStats, setSurvivalStats] = useState<{ logrank?: { p_value: number; significant: boolean } } | null>(null);
  
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!project || !data.loaded) {
      navigate('/data');
      return;
    }
    loadColumns();
  }, [project, data.loaded, navigate]);

  const loadColumns = async () => {
    if (!data.cacheKey) return;
    
    try {
      const response = await dataApi.getColumns(data.cacheKey);
      setColumns(response.data.columns);
    } catch (error) {
      console.error('Failed to load columns:', error);
    }
  };

  const runStatisticalTest = async () => {
    if (!data.cacheKey || !xVariable || !yVariable) {
      addToast('Please select both variables', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await statsApi.autoTest({
        cache_key: data.cacheKey,
        x_variable: xVariable,
        y_variable: yVariable,
      });
      setStatResult(response.data);
    } catch (error) {
      addToast('Failed to run statistical test', 'error');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const runSurvivalAnalysis = async () => {
    if (!data.cacheKey || !timeVariable || !eventVariable) {
      addToast('Please select time and event variables', 'error');
      return;
    }

    setLoading(true);
    try {
      const response = await statsApi.survival({
        cache_key: data.cacheKey,
        time_variable: timeVariable,
        event_variable: eventVariable,
        grouping_variable: groupVariable || undefined,
      });
      setSurvivalCurves(response.data.curves);
      setSurvivalStats(response.data.statistics);
    } catch (error) {
      addToast('Failed to run survival analysis', 'error');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const numericColumns = columns.filter((c) => c.is_numeric);
  const categoricalColumns = columns.filter((c) => c.is_categorical);
  const allColumnOptions = columns.map((c) => ({
    value: c.name,
    label: `${c.name} (${c.is_numeric ? 'numeric' : 'categorical'})`,
  }));

  const tabs = [
    { id: 'comparison' as const, label: 'Group Comparison', icon: BarChart3 },
    { id: 'correlation' as const, label: 'Correlation', icon: TrendingUp },
    { id: 'survival' as const, label: 'Survival Analysis', icon: Clock },
  ];

  // Prepare survival chart data
  const survivalChartData = survivalCurves.length > 0 
    ? survivalCurves[0].timeline.map((time, i) => {
        const point: Record<string, number> = { time };
        survivalCurves.forEach((curve) => {
          point[curve.group] = curve.survival[i];
        });
        return point;
      })
    : [];

  return (
    <div className="p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="font-display text-3xl font-bold text-pie-text mb-2">
          Statistical Workbench
        </h1>
        <p className="text-pie-text-muted">
          Perform statistical tests and visualize relationships in your data
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 mb-6">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg transition-all',
                activeTab === tab.id
                  ? 'bg-pie-accent text-white'
                  : 'bg-pie-surface text-pie-text-muted hover:text-pie-text'
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="grid grid-cols-2 gap-6">
        {/* Configuration Panel */}
        <Card>
          <CardHeader>
            <CardTitle>
              {activeTab === 'comparison' && 'Compare Groups'}
              {activeTab === 'correlation' && 'Correlation Analysis'}
              {activeTab === 'survival' && 'Survival Analysis'}
            </CardTitle>
            <CardDescription>
              {activeTab === 'comparison' && 'Drag variables to compare across groups'}
              {activeTab === 'correlation' && 'Analyze relationships between numeric variables'}
              {activeTab === 'survival' && 'Configure Kaplan-Meier analysis'}
            </CardDescription>
          </CardHeader>

          <CardContent>
            {activeTab === 'comparison' && (
              <div className="space-y-6">
                {/* Drop Zones */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-pie-text">X-Axis (Groups)</label>
                    <Select
                      value={xVariable}
                      onChange={(e) => setXVariable(e.target.value)}
                      options={allColumnOptions}
                      placeholder="Select grouping variable..."
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-pie-text">Y-Axis (Values)</label>
                    <Select
                      value={yVariable}
                      onChange={(e) => setYVariable(e.target.value)}
                      options={allColumnOptions}
                      placeholder="Select outcome variable..."
                    />
                  </div>
                </div>

                <Button
                  variant="primary"
                  className="w-full"
                  onClick={runStatisticalTest}
                  disabled={!xVariable || !yVariable || loading}
                  loading={loading}
                >
                  <Activity className="w-4 h-4" />
                  Run Auto Test
                </Button>
              </div>
            )}

            {activeTab === 'correlation' && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <Select
                    label="Variable 1"
                    value={xVariable}
                    onChange={(e) => setXVariable(e.target.value)}
                    options={numericColumns.map((c) => ({ value: c.name, label: c.name }))}
                    placeholder="Select variable..."
                  />
                  <Select
                    label="Variable 2"
                    value={yVariable}
                    onChange={(e) => setYVariable(e.target.value)}
                    options={numericColumns.map((c) => ({ value: c.name, label: c.name }))}
                    placeholder="Select variable..."
                  />
                </div>

                <Button
                  variant="primary"
                  className="w-full"
                  onClick={runStatisticalTest}
                  disabled={!xVariable || !yVariable || loading}
                  loading={loading}
                >
                  <TrendingUp className="w-4 h-4" />
                  Calculate Correlation
                </Button>
              </div>
            )}

            {activeTab === 'survival' && (
              <div className="space-y-4">
                <Select
                  label="Time Variable"
                  value={timeVariable}
                  onChange={(e) => setTimeVariable(e.target.value)}
                  options={numericColumns.map((c) => ({ value: c.name, label: c.name }))}
                  placeholder="Select time variable..."
                />
                
                <Select
                  label="Event Variable (1=event, 0=censored)"
                  value={eventVariable}
                  onChange={(e) => setEventVariable(e.target.value)}
                  options={allColumnOptions}
                  placeholder="Select event variable..."
                />
                
                <Select
                  label="Grouping Variable (optional)"
                  value={groupVariable}
                  onChange={(e) => setGroupVariable(e.target.value)}
                  options={[{ value: '', label: 'None' }, ...categoricalColumns.map((c) => ({ value: c.name, label: c.name }))]}
                />

                <Button
                  variant="primary"
                  className="w-full"
                  onClick={runSurvivalAnalysis}
                  disabled={!timeVariable || !eventVariable || loading}
                  loading={loading}
                >
                  <Clock className="w-4 h-4" />
                  Run Kaplan-Meier Analysis
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Panel */}
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
          </CardHeader>

          <CardContent>
            {loading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-pie-accent" />
              </div>
            )}

            {!loading && activeTab !== 'survival' && statResult && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-4"
              >
                {/* Test Info */}
                <div className="p-4 rounded-lg bg-pie-surface">
                  <h4 className="font-semibold text-pie-text mb-2">{statResult.test_name}</h4>
                  <p className="text-sm text-pie-text-muted">{statResult.description}</p>
                </div>

                {/* P-value Display */}
                <div className={clsx(
                  'p-4 rounded-lg border-2',
                  statResult.significant
                    ? 'bg-pie-success/10 border-pie-success/50'
                    : 'bg-pie-surface border-pie-border'
                )}>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-pie-text-muted">P-value</span>
                    <span className={clsx(
                      'text-2xl font-mono font-bold',
                      statResult.significant ? 'text-pie-success' : 'text-pie-text'
                    )}>
                      {statResult.p_value.toFixed(4)}
                    </span>
                  </div>
                  {statResult.significant && (
                    <p className="text-sm text-pie-success mt-2">✓ Statistically significant (p &lt; 0.05)</p>
                  )}
                </div>

                {/* Interpretation */}
                <div className="p-4 rounded-lg bg-pie-card">
                  <h5 className="font-medium text-pie-text mb-2">Interpretation</h5>
                  <p className="text-sm text-pie-text-muted">{statResult.interpretation}</p>
                </div>
              </motion.div>
            )}

            {!loading && activeTab === 'survival' && survivalCurves.length > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-4"
              >
                {/* Kaplan-Meier Curve */}
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={survivalChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a3a5c" />
                      <XAxis 
                        dataKey="time" 
                        stroke="#8b9dc3"
                        fontSize={12}
                        label={{ value: 'Time', position: 'bottom', fill: '#8b9dc3' }}
                      />
                      <YAxis 
                        stroke="#8b9dc3"
                        fontSize={12}
                        domain={[0, 1]}
                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                        label={{ value: 'Survival', angle: -90, position: 'left', fill: '#8b9dc3' }}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1a2540', border: '1px solid #2a3a5c' }}
                        labelStyle={{ color: '#e8eff8' }}
                      />
                      <Legend />
                      {survivalCurves.map((curve, i) => (
                        <Line
                          key={curve.group}
                          type="stepAfter"
                          dataKey={curve.group}
                          stroke={i === 0 ? '#ff6b4a' : '#4ecdc4'}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Survival Stats */}
                <div className="grid grid-cols-2 gap-4">
                  {survivalCurves.map((curve) => (
                    <div key={curve.group} className="p-4 rounded-lg bg-pie-surface">
                      <h5 className="font-medium text-pie-text mb-1">{curve.group}</h5>
                      <p className="text-sm text-pie-text-muted">
                        Median survival: {curve.median_survival !== null 
                          ? `${curve.median_survival.toFixed(1)} units`
                          : 'Not reached'}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Log-rank test */}
                {survivalStats?.logrank && (
                  <div className={clsx(
                    'p-4 rounded-lg border-2',
                    survivalStats.logrank.significant
                      ? 'bg-pie-success/10 border-pie-success/50'
                      : 'bg-pie-surface border-pie-border'
                  )}>
                    <h5 className="font-medium text-pie-text mb-2">Log-rank Test</h5>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-pie-text-muted">P-value</span>
                      <span className="text-xl font-mono font-bold">
                        {survivalStats.logrank.p_value.toFixed(4)}
                      </span>
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {!loading && !statResult && survivalCurves.length === 0 && (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Plus className="w-12 h-12 text-pie-text-muted mb-4" />
                <p className="text-pie-text-muted">
                  Select variables and run a test to see results
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
