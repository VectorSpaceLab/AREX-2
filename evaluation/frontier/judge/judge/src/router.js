import express from 'express';
import tar from 'tar';
import { emptyDir } from './utils.js';
import { uploadZip, upload } from './upload.js';

export function createApiRoutes(judgeEngine, problemManager, submissionManager) {
    const router = express.Router();
    router.use(express.json({ limit: '10mb' }));
    router.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Submit code
    router.post('/submit', upload.single('code'), async (req, res) => {
        try {
            const pid = (req.body && req.body.pid) || (req.query && req.query.pid);
            const langRaw = (req.body && req.body.lang) || (req.query && req.query.lang);

            let code = null;

            // (a) multipart file field: code=@file
            if (req.file && req.file.buffer) {
                code = req.file.buffer.toString('utf8');
            }

            // (b) text field: multipart text / x-www-form-urlencoded / JSON
            if (!code && typeof req.body?.code === 'string') {
                code = req.body.code;
            }

            // (c) base64 in JSON as fallback
            if (!code && typeof req.body?.codeBase64 === 'string') {
                try {
                    code = Buffer.from(req.body.codeBase64, 'base64').toString('utf8');
                } catch { /* ignore */ }
            }

            // (d) if express.text is enabled and Content-Type is text/*, entire body is source code
            if (!code && typeof req.body === 'string' && req.is && req.is('text/*')) {
                code = req.body;
            }

            // 3) Language normalization (adjust according to your judge's enum)
            const langMap = {
                'c++': 'cpp', 'cpp': 'cpp', 'cxx': 'cpp', 'g++': 'cpp', 'cpp17': 'cpp', 'gnu++17': 'cpp',
                'python': 'py', 'py': 'py', 'python3': 'py', 'py3': 'py'
            };
            const lang = langMap[(langRaw || '').toLowerCase()] || langRaw;

            // 4) Validation
            if (!pid || !lang || !code) {
                return res.status(400).json({
                    error: 'pid/lang/code required',
                    hint: 'please try json as the easist way.'
                });
            }

            // 5) Submit
            const sid = await judgeEngine.submit(pid, lang, code);
            return res.json({ sid });
        } catch (error) {
            return res.status(500).json({ error: 'Submit failed', message: error.message });
        }
    });

    // Get result
    router.get('/result/:sid', async (req, res) => {
        const sid = parseInt(req.params.sid, 10);
        if (Number.isNaN(sid)) {
            return res.status(400).json({ error: 'sid must be number' });
        }

        try {
            const result = await judgeEngine.getResult(sid);
            if (!result) {
                return res.status(404).json({ error: 'not found' });
            }
            if (req.query.short) {
                const { status, passed } = result;
                return res.json({ status, passed });
            }

            res.json(result);
        } catch (error) {
            res.status(500).json({ error: 'Failed to get result', message: error.message });
        }
    });


    // Get problem statement
    router.get('/problem/:pid/statement', async (req, res) => {
        try {
            const statement = await problemManager.getStatement(req.params.pid);
            res.type('text/plain').send(statement);
        } catch {
            res.status(404).send('statement not found');
        }
    });

    // Get list of all problems
    router.get('/problems', async (req, res) => {
        try {
            const includeStatement = req.query.statement === 'true';
            const problems = await problemManager.listProblems(includeStatement);
            res.json({ problems });
        } catch (error) {
            res.status(500).json({ 
                error: 'Failed to list problems', 
                message: error.message 
            });
        }
    });

    // Export submissions
    router.get('/submissions/export', async (req, res) => {
        try {
            // Set response headers
            res.setHeader('Content-Type', 'application/gzip');
            res.setHeader('Content-Disposition', `attachment; filename=submissions_${Date.now()}.tar.gz`);

            // Create tar stream directly and pipe to response
            tar.c(
                {
                    gzip: true,
                    cwd: submissionManager.submissionsRoot
                },
                ['.']
            ).pipe(res);
            
        } catch (error) {
            if (!res.headersSent) {
                res.status(500).json({ 
                    error: 'Failed to export submissions', 
                    message: error.message 
                });
            }
        }
    });

    // Reset submissions
    router.post('/submissions/reset', async (req, res) => {
        try {
            await emptyDir(submissionManager.submissionsRoot);
            await submissionManager.resetCounter();

            // Clear result cache in memory
            judgeEngine.clearResults();
            
            res.json({
                success: true,
                message: 'Submissions reset successfully'
            });
            
        } catch (error) {
            res.status(500).json({ 
                error: 'Failed to reset submissions', 
                message: error.message 
            });
        }
    });

    router.post('/problem/setup', uploadZip, async (req, res) => {
        const { pid } = req.body || {};
        const zipPath = req.file?.path || null;

        if (!pid) {
            return res.status(400).json({ error: 'pid is required' });
        }
        try {
            await problemManager.setupProblem(pid, zipPath);
            res.json({ message: 'Problem setup successfully', pid });
        } catch (error) {
            res.status(500).json({ error: 'Failed to setup problem', message: error.message });
        }
    });

    router.post('/problem/add-problem', uploadZip, async (req, res) => {
        try {
            const { pid } = req.body || {};
            if (!pid) {
                return res.status(400).json({ error: 'pid is required' });
            }

            const zipfilePath = req.file?.path || null;
            const result = await problemManager.addProblem(pid, zipfilePath, {});
            return res.json(result);
        } catch (error) {
            return res.status(500).json({ error: 'Failed to add problem', message: error.message });
        }
    });

    router.get('/package/:pid' , async (req, res) => {
        const pid = req.params.pid;
        if (!pid) {
            return res.status(400).json({ error: 'pid is required' });
        }

        try {
            const packagePath = await problemManager.getPackage(pid);
            if (packagePath) {
                res.download(packagePath, `${pid}.tar.gz`);
            } else {
                res.status(404).json({ error: 'Package not found' });
            }
        } catch (error) {
            res.status(500).json({ error: 'Failed to get package', message: error.message });
        }
    });

    // Health check
    router.get('/health', (_, res) => {
        res.json({ ok: true });
    });

    return router;
}