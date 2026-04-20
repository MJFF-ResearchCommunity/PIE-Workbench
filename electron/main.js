const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0a0f1a',
    show: false,
  });

  // Load the app
  const isDev = !app.isPackaged;
  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startPythonBackend() {
  const isDev = !app.isPackaged;
  const backendDir = isDev 
    ? path.join(__dirname, '..', 'backend')
    : path.join(process.resourcesPath, 'backend');
  
  // Use venv Python in both dev and production
  const pythonPath = path.join(backendDir, 'venv', 'bin', 'python');
  
  // Set up PYTHONPATH to include the lib directory
  const libPath = isDev 
    ? path.join(__dirname, '..', 'lib')
    : path.join(process.resourcesPath, 'lib');

  pythonProcess = spawn(pythonPath, ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', '8100'], {
    cwd: backendDir,
    env: { 
      ...process.env, 
      PYTHONPATH: `${libPath}/PIE:${libPath}/PIE-clean:${process.env.PYTHONPATH || ''}`
    }
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
  });
}

// IPC handlers
ipcMain.handle('select-directory', async () => {
  try {
    if (!mainWindow) {
      console.error('mainWindow is null');
      return null;
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
      title: 'Select Data Directory'
    });
    console.log('Directory dialog result:', result);
    return result.canceled ? null : result.filePaths[0] || null;
  } catch (error) {
    console.error('Error showing directory dialog:', error);
    return null;
  }
});

ipcMain.handle('select-file', async (event, options) => {
  try {
    if (!mainWindow) {
      console.error('mainWindow is null');
      return null;
    }
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openFile'],
      filters: options?.filters || [],
      title: 'Select File'
    });
    console.log('File dialog result:', result);
    return result.canceled ? null : result.filePaths[0] || null;
  } catch (error) {
    console.error('Error showing file dialog:', error);
    return null;
  }
});

// Write report HTML to a temp file and open it in the user's default browser.
// Used by the Results view's "View Full Report" button so the report renders
// in a real browser window instead of an in-app modal iframe.
ipcMain.handle('open-report-html', async (event, html) => {
  try {
    if (typeof html !== 'string' || !html) {
      return { ok: false, error: 'No report content to open' };
    }
    const filename = `pie-report-${Date.now()}.html`;
    const filePath = path.join(os.tmpdir(), filename);
    fs.writeFileSync(filePath, html, 'utf8');
    await shell.openExternal(`file://${filePath}`);
    return { ok: true, path: filePath };
  } catch (error) {
    console.error('Failed to open report externally:', error);
    return { ok: false, error: String(error?.message || error) };
  }
});

ipcMain.handle('save-file', async (event, options) => {
  try {
    if (!mainWindow) {
      console.error('mainWindow is null');
      return null;
    }
    const result = await dialog.showSaveDialog(mainWindow, {
      filters: options?.filters || [],
      title: 'Save File'
    });
    console.log('Save dialog result:', result);
    return result.canceled ? null : result.filePath || null;
  } catch (error) {
    console.error('Error showing save dialog:', error);
    return null;
  }
});

app.whenReady().then(() => {
  startPythonBackend();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});
