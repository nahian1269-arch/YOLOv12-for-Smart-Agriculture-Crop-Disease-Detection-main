import { existsSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

export const projectRoot = process.cwd();
export const isWindows = process.platform === "win32";
export const venvDir = join(projectRoot, ".venv311");
export const venvPython = join(
  venvDir,
  isWindows ? "Scripts" : "bin",
  isWindows ? "python.exe" : "python"
);

function findSystemPython() {
  const candidates = isWindows
    ? [["py", ["-3.11"]], ["py", ["-3"]], ["python", []]]
    : [["python3.11", []], ["python3", []], ["python", []]];

  for (const [command, args] of candidates) {
    const result = spawnSync(command, [...args, "--version"], {
      cwd: projectRoot,
      stdio: "ignore"
    });
    if (result.status === 0) {
      return { command, args };
    }
  }
  return null;
}

export function ensureVirtualEnvironment() {
  if (existsSync(venvPython)) {
    return venvPython;
  }

  const systemPython = findSystemPython();
  if (!systemPython) {
    throw new Error("Python 3 was not found. Install Python 3.11 or newer, then run npm start again.");
  }

  const createResult = spawnSync(
    systemPython.command,
    [...systemPython.args, "-m", "venv", ".venv311"],
    { cwd: projectRoot, stdio: "inherit" }
  );
  if (createResult.status !== 0 || !existsSync(venvPython)) {
    throw new Error("Unable to create the Python virtual environment.");
  }
  return venvPython;
}

export function installRequirements(python) {
  const result = spawnSync(
    python,
    ["-m", "pip", "install", "-r", "requirements.txt"],
    { cwd: projectRoot, stdio: "inherit" }
  );
  if (result.status !== 0) {
    throw new Error("Python dependency installation failed.");
  }
}

export function requirementsAvailable(python) {
  const check = spawnSync(
    python,
    [
      "-c",
      "import cv2, flask, keras, numpy, requests, sklearn, supervision, tensorflow, ultralytics, werkzeug"
    ],
    { cwd: projectRoot, stdio: "ignore" }
  );
  return check.status === 0;
}
