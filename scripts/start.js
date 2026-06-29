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

try {
  const python = ensureVirtualEnvironment();
  if (!existsSync(requirementsMarker)) {
    if (!requirementsAvailable(python)) {
      installRequirements(python);
    }
    const { writeFileSync } = await import("node:fs");
    writeFileSync(requirementsMarker, "ready\n", "utf8");
  }

  const child = spawn(python, ["-u", "app.py"], {
    cwd: projectRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      NUROAGRO_DEBUG: development ? "1" : "0",
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
