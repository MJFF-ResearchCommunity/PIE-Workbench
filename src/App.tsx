import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from './components/ui/Toaster';
import Layout from './components/Layout';
import ProjectHub from './views/ProjectHub';
import DataIngestion from './views/DataIngestion';
import MLEngine from './views/MLEngine';
import StatsLab from './views/StatsLab';
import Results from './views/Results';

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<ProjectHub />} />
          <Route path="/data" element={<DataIngestion />} />
          <Route path="/ml" element={<MLEngine />} />
          <Route path="/stats" element={<StatsLab />} />
          <Route path="/results" element={<Results />} />
        </Routes>
      </Layout>
      <Toaster />
    </BrowserRouter>
  );
}

export default App;
