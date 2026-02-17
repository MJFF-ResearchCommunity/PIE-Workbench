import { ReactNode } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Home, 
  Database, 
  Brain, 
  BarChart3, 
  FileText,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { clsx } from 'clsx';

interface LayoutProps {
  children: ReactNode;
}

const navItems = [
  { path: '/', icon: Home, label: 'Project Hub', step: 'project_hub' as const },
  { path: '/data', icon: Database, label: 'Data Ingestion', step: 'data_ingestion' as const },
  { path: '/ml', icon: Brain, label: 'ML Engine', step: 'ml_engine' as const },
  { path: '/stats', icon: BarChart3, label: 'Statistics', step: 'stats_lab' as const },
  { path: '/results', icon: FileText, label: 'Results', step: 'results' as const },
];

export default function Layout({ children }: LayoutProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { sidebarCollapsed, toggleSidebar, project, runningTasks } = useStore();

  const hasActiveTasks = runningTasks.size > 0;

  return (
    <div className="flex min-h-screen bg-pie-bg">
      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: sidebarCollapsed ? 72 : 240 }}
        className="fixed left-0 top-0 h-full bg-pie-surface border-r border-pie-border z-50 flex flex-col"
      >
        {/* Logo */}
        <div className="p-4 border-b border-pie-border flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl overflow-hidden flex-shrink-0">
            <img src="/icon.png" alt="PIE Workbench" className="w-full h-full object-cover" />
          </div>
          <AnimatePresence>
            {!sidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                className="overflow-hidden"
              >
                <h1 className="font-display text-xl font-semibold text-pie-text whitespace-nowrap">
                  PIE Workbench
                </h1>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            
            return (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={clsx(
                  'w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-all duration-200',
                  isActive
                    ? 'bg-pie-accent/20 text-pie-accent'
                    : 'text-pie-text-muted hover:bg-pie-card hover:text-pie-text'
                )}
              >
                <Icon className={clsx('w-5 h-5 flex-shrink-0', isActive && 'text-pie-accent')} />
                <AnimatePresence>
                  {!sidebarCollapsed && (
                    <motion.span
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: 'auto' }}
                      exit={{ opacity: 0, width: 0 }}
                      className="font-medium whitespace-nowrap overflow-hidden"
                    >
                      {item.label}
                    </motion.span>
                  )}
                </AnimatePresence>
              </button>
            );
          })}
        </nav>

        {/* Running Tasks Indicator */}
        {hasActiveTasks && (
          <div className="p-3 border-t border-pie-border">
            <div className="flex items-center gap-2 px-3 py-2 bg-pie-accent/10 rounded-lg">
              <div className="w-2 h-2 bg-pie-accent rounded-full animate-pulse" />
              <AnimatePresence>
                {!sidebarCollapsed && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-sm text-pie-accent"
                  >
                    {runningTasks.size} task{runningTasks.size > 1 ? 's' : ''} running
                  </motion.span>
                )}
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* Project Info */}
        {project && !sidebarCollapsed && (
          <div className="p-3 border-t border-pie-border">
            <div className="px-3 py-2 bg-pie-card rounded-lg">
              <p className="text-xs text-pie-text-muted">Current Project</p>
              <p className="text-sm font-medium text-pie-text truncate">{project.name}</p>
            </div>
          </div>
        )}

        {/* Collapse Toggle */}
        <button
          onClick={toggleSidebar}
          className="absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 bg-pie-card border border-pie-border rounded-full flex items-center justify-center text-pie-text-muted hover:text-pie-text hover:bg-pie-surface transition-colors"
        >
          {sidebarCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </motion.aside>

      {/* Main Content */}
      <main
        className={clsx(
          'flex-1 transition-all duration-300',
          sidebarCollapsed ? 'ml-[72px]' : 'ml-[240px]'
        )}
      >
        <div className="min-h-screen">
          {/* Background pattern */}
          <div className="fixed inset-0 bg-mesh opacity-30 pointer-events-none" />
          
          {/* Content */}
          <div className="relative z-10">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                {children}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  );
}
