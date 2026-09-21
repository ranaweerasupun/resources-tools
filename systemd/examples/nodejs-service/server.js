// © 2025 [Supun Akalanka Sriyananda] — MIT License
/**
 * server.js
 * ---------
 * A minimal Node.js HTTP server for learning systemd.
 * Responds to every request with the current server time so
 * you can verify it's running by visiting http://<pi-ip>:3000
 * or by running: curl http://localhost:3000
 *
 * The PORT and HOST values are read from environment variables
 * set in the service file, demonstrating how to pass configuration
 * to an application without hardcoding it.
 *
 * Deploy:
 *   sudo cp hello-node.service /etc/systemd/system/
 *   sudo systemctl daemon-reload
 *   sudo systemctl enable --now hello-node.service
 */

const http = require("http");

// Read configuration from environment variables injected by the
// service file. Provide sensible defaults for local development
// so the script also works when run manually without systemd.
const PORT = parseInt(process.env.PORT || "3000", 10);
const HOST = process.env.HOST || "0.0.0.0";

// Create the HTTP server. Every request gets a plain-text response
// with the current timestamp and the request path.
const server = http.createServer((req, res) => {
  const timestamp = new Date().toISOString();
  const message   = `Hello from systemd!\nServer time: ${timestamp}\nPath: ${req.url}\n`;

  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end(message);

  // Log to stdout — systemd captures this to the journal automatically.
  console.log(`[${timestamp}] ${req.method} ${req.url} — responded 200`);
});

server.listen(PORT, HOST, () => {
  console.log(`[${new Date().toISOString()}] Server listening on http://${HOST}:${PORT}`);
});

// -----------------------------------------------------------------------
// Graceful shutdown handler
// -----------------------------------------------------------------------
// When you run 'systemctl stop', systemd sends SIGTERM to the process
// before force-killing it. Without this handler, Node.js would terminate
// immediately, potentially dropping in-flight requests.
//
// This handler tells the HTTP server to stop accepting new connections
// and waits for existing connections to finish before exiting cleanly.
// The TimeoutStopSec directive in the service file (15 seconds) gives
// this handler a generous window to complete before SIGKILL is sent.
process.on("SIGTERM", () => {
  console.log(`[${new Date().toISOString()}] Received SIGTERM — shutting down gracefully`);

  server.close(() => {
    console.log(`[${new Date().toISOString()}] All connections closed. Exiting cleanly.`);
    // Exit with 0 (success) so systemd doesn't count this as a failure
    // and trigger an unnecessary restart.
    process.exit(0);
  });
});

// Also handle SIGINT (Ctrl+C when running manually) the same way,
// so the shutdown path is consistent whether running under systemd or not.
process.on("SIGINT", () => {
  console.log(`[${new Date().toISOString()}] Received SIGINT — shutting down`);
  server.close(() => process.exit(0));
});
