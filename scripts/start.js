import { existsSync } from "node:fs";
import { join } from "node:path";
import { spawn } from "node:child_process";
import {
  ensureVirtualEnvironment,
  installRequirements,
  projectRoot,
  requirementsAvailable,
  venvDir
} from "./python-runtime.js";

const requirementsMarker = join(venvDir, ".nuroagro-ready");
const development = process.argv.includes("--dev");
const production = !development && (process.env.RAILWAY_ENVIRONMENT || process.env.NODE_ENV === "production");

try {
  const python = ensureVirtualEnvironment();
  if (!existsSync(requirementsMarker)) {
    if (!requirementsAvailable(python)) {
      installRequirements(python);
    }
    const { writeFileSync } = await import("node:fs");
    writeFileSync(requirementsMarker, "ready\n", "utf8");
  }

  const args = production
    ? [
        "-m",
        "gunicorn",
        "app:app",
        "--bind",
        `0.0.0.0:${process.env.PORT || "5000"}`,
        "--workers",
        process.env.WEB_CONCURRENCY || "1",
        "--timeout",
        process.env.GUNICORN_TIMEOUT || "180",
        "--log-file",
        "-"
      ]
    : ["-u", "app.py"];

  const child = spawn(python, args, {
    cwd: projectRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      NODE_ENV: production ? "production" : (process.env.NODE_ENV || "development"),
      PYTHONUNBUFFERED: "1",
      NUROAGRO_DEBUG: development ? "1" : "0",
      NUROAGRO_HOST: production ? "0.0.0.0" : (process.env.NUROAGRO_HOST || "127.0.0.1"),
      MPLCONFIGDIR: join(projectRoot, ".matplotlib")
    }
  });

  const stop = (signal) => {
    if (!child.killed) child.kill(signal);
  };
  process.on("SIGINT", () => stop("SIGINT"));
  process.on("SIGTERM", () => stop("SIGTERM"));
  child.on("exit", (code) => process.exit(code ?? 0));
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
