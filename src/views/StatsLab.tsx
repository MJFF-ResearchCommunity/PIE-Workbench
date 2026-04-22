import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  BarChart3,
  ScatterChart as ScatterIcon,
  TrendingUp,
  Activity,
  Clock,
  Sparkles,
  Pill,
  FileText,
} from 'lucide-react';
import { clsx } from 'clsx';
import { useStore } from '../store/useStore';
import { dataApi } from '../services/api';
import DescribeTab from './stats/DescribeTab';
import CompareTab from './stats/CompareTab';
import CorrelateTab from './stats/CorrelateTab';
import RegressTab from './stats/RegressTab';
import LongitudinalTab from './stats/LongitudinalTab';
import SurviveTab from './stats/SurviveTab';
import MultitestTab from './stats/MultitestTab';
import PDHelpersTab from './stats/PDHelpersTab';
import { ColumnInfo } from './stats/shared/types';

type TabId =
  | 'describe'
  | 'compare'
  | 'correlate'
  | 'regress'
  | 'longitudinal'
  | 'survive'
  | 'multitest'
  | 'pd';

const TABS: { id: TabId; label: string; icon: any; description: string }[] = [
  { id: 'describe', label: 'Describe', icon: FileText, description: 'Summary stats, normality, missingness' },
  { id: 'compare', label: 'Compare', icon: BarChart3, description: 'Group comparisons with effect sizes and post-hoc' },
  { id: 'correlate', label: 'Correlate', icon: ScatterIcon, description: 'Pairwise, partial, and matrix-wide correlations' },
  { id: 'regress', label: 'Regress', icon: TrendingUp, description: 'Linear, logistic, and ANCOVA with diagnostics' },
  { id: 'longitudinal', label: 'Longitudinal', icon: Activity, description: 'Mixed-effects models + change-from-baseline' },
  { id: 'survive', label: 'Survive', icon: Clock, description: 'Kaplan-Meier and Cox proportional hazards' },
  { id: 'multitest', label: 'Multitest', icon: Sparkles, description: 'FDR / Bonferroni / Holm correction' },
  { id: 'pd', label: 'PD Helpers', icon: Pill, description: 'LEDD, UPDRS, Hoehn & Yahr utilities' },
];

export default function StatsLab() {
  const navigate = useNavigate();
  const { project, data } = useStore();
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [activeTab, setActiveTab] = useState<TabId>('describe');

  useEffect(() => {
    if (!project || !data.loaded) {
      navigate('/data');
      return;
    }
    if (data.cacheKey) {
      dataApi.getColumns(data.cacheKey).then((r) => setColumns(r.data.columns)).catch(() => undefined);
    }
  }, [project, data.loaded, data.cacheKey, navigate]);

  const cacheKey = data.cacheKey ?? '';
  const activeMeta = useMemo(() => TABS.find((t) => t.id === activeTab)!, [activeTab]);

  return (
    <div className="p-6 max-w-[1600px] mx-auto">
      {/* Compact header */}
      <div className="mb-5">
        <h1 className="font-display text-2xl font-bold text-pie-text leading-tight">Statistics Lab</h1>
        <p className="text-sm text-pie-text-muted">
          Classical statistical tools — built for researchers without ML training.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex flex-wrap gap-1 mb-4 border-b border-pie-border pb-1">
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={clsx(
                'px-4 py-2.5 rounded-t-md text-sm font-medium transition-all flex items-center gap-2',
                active
                  ? 'bg-pie-accent text-white'
                  : 'text-pie-text-muted hover:text-pie-text hover:bg-pie-surface'
              )}
              title={t.description}
            >
              <Icon className="w-4 h-4" />
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="mb-4 text-sm text-pie-text-muted">
        <span className="text-pie-text font-medium">{activeMeta.label}:</span> {activeMeta.description}
      </div>

      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        {activeTab === 'describe' && <DescribeTab cacheKey={cacheKey} columns={columns} />}
        {activeTab === 'compare' && <CompareTab cacheKey={cacheKey} columns={columns} />}
        {activeTab === 'correlate' && <CorrelateTab cacheKey={cacheKey} columns={columns} />}
        {activeTab === 'regress' && <RegressTab cacheKey={cacheKey} columns={columns} />}
        {activeTab === 'longitudinal' && <LongitudinalTab cacheKey={cacheKey} columns={columns} />}
        {activeTab === 'survive' && <SurviveTab cacheKey={cacheKey} columns={columns} />}
        {activeTab === 'multitest' && <MultitestTab cacheKey={cacheKey} columns={columns} />}
        {activeTab === 'pd' && <PDHelpersTab cacheKey={cacheKey} columns={columns} />}
      </motion.div>
    </div>
  );
}
