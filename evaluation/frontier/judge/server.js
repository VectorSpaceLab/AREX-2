import express from 'express';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

import { SubmissionManager } from './src/utils.js';
import { ProblemManager } from './src/problem_manager.js';
import { JudgeEngine } from './src/judge_engine.js';
import { createApiRoutes } from './src/router.js';

// Initialize configuration
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const config = {
    problemsRoot: path.join(__dirname, 'problems'),
    dataRoot: path.join(__dirname, 'data'),
    submissionsRoot: process.env.SUBMISSIONS_DIR || path.join(__dirname, 'submissions'),
    bucketSize: +(process.env.SUB_BUCKET || 100),
    gjAddr: process.env.GJ_ADDR || 'http://127.0.0.1:5050',
    workers: +(process.env.JUDGE_WORKERS || 4),
    testlibPath: process.env.TESTLIB_INSIDE || '/lib/testlib',
    port: process.env.PORT || 8081
};

// Create directories
await fs.mkdir(config.dataRoot, { recursive: true });
await fs.mkdir(config.submissionsRoot, { recursive: true });

// Initialize modules
const submissionManager = new SubmissionManager(
    config.dataRoot, 
    config.submissionsRoot, 
    config.bucketSize
);

const problemManager = new ProblemManager({
    problemsRoot: config.problemsRoot,
    gjAddr: config.gjAddr,
    testlibPath: config.testlibPath
});

const judgeEngine = new JudgeEngine({
    problemsRoot: config.problemsRoot,
    gjAddr: config.gjAddr,
    submissionManager,
    testlibPath: config.testlibPath,
    workers: config.workers
});

// Create Express application
const app = express();
app.use(express.json({ limit: '10mb' }));

// Register routes
const apiRoutes = createApiRoutes(judgeEngine, problemManager, submissionManager);
app.use('/', apiRoutes);

// Start server
app.listen(config.port, () => {
    console.log(`LightCPVerifier listening on port ${config.port} (modular architecture)`);
});

export { judgeEngine, problemManager, submissionManager };