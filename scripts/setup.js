import { ensureVirtualEnvironment, installRequirements } from "./python-runtime.js";

try {
  const python = ensureVirtualEnvironment();
  installRequirements(python);
  console.log("NuroAgro environment is ready.");
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
