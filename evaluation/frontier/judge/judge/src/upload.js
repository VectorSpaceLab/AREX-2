import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import multer from 'multer';

const uploadRoot = path.join(os.tmpdir(), 'uploads');
await fs.mkdir(uploadRoot, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadRoot),
  filename: (req, file, cb) => {
    // Preserve original file extension
    const ext = path.extname(file.originalname) || '.zip';
    cb(null, `problem_${Date.now()}${ext}`);
  }
});

// Only accept zip files (common mime types), also allow extension-based fallback
function zipFileFilter(req, file, cb) {
  const okMime = ['application/zip', 'application/x-zip-compressed'];
  const okExt = /\.zip$/i.test(file.originalname);
  if (okMime.includes(file.mimetype) || okExt) cb(null, true);
  else cb(new Error('Only .zip is allowed'));
}

export const uploadZip = multer({
  storage,
  fileFilter: zipFileFilter,
  limits: { fileSize: 2 * 1024 * 1024 * 1024 } // 2GB limit, adjust as needed
}).single('zipfile');


export const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 5 * 1024 * 1024 } });
