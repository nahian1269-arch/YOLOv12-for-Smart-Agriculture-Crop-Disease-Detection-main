# NuroAgro

Smart agriculture platform for traditional, hydroponic, aquaponic, aeroponic, and hybrid vertical farms.

## Local Run

Use the npm wrapper:

```bash
npm start
```

On Windows PowerShell, use:

```powershell
npm.cmd start
```

Open `http://127.0.0.1:5000`.

For Flask debug mode:

```bash
npm run dev
```

## Railway Deployment

This repo includes `railway.json`, `nixpacks.toml`, and `.python-version` for Railway.

Railway uses:

- Build command: `npm install && npm run setup`
- Start command: `npm start`
- Healthcheck: `/healthz`
- Production server: Gunicorn on `0.0.0.0:$PORT`

### Required Railway Variables

Set these in Railway before deploying:

```env
NODE_ENV=production
FLASK_SECRET_KEY=replace-with-a-long-random-secret
NUROAGRO_ADMIN_USERNAME=admin
NUROAGRO_ADMIN_PASSWORD=replace-with-a-strong-password
NUROAGRO_ADMIN_ACCESS_KEY=replace-with-a-strong-access-key
```

Optional:

```env
SUPABASE_URL=your-supabase-url
SUPABASE_PUBLISHABLE_KEY=your-publishable-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
OPENAI_API_KEY=your-openai-key
OPENAI_PROJECT_MODEL=gpt-5.5
WEB_CONCURRENCY=1
GUNICORN_TIMEOUT=180
```

If you add a Railway volume, set:

```env
NUROAGRO_RUNTIME_DIR=/data
```

or rely on Railway's `RAILWAY_VOLUME_MOUNT_PATH` if configured.

## Notes

- The first Railway build installs large ML dependencies, including TensorFlow and Ultralytics.
- Keep `WEB_CONCURRENCY=1` unless your Railway plan has enough memory to load multiple YOLO/TensorFlow model copies.
- OpenCV uses the headless wheel for server deployment, so no GUI or `libGL` Nix package is required.
- Do not deploy with local defaults like `admin123`; production startup requires admin and secret variables.
