const os = require("node:os");

// The Vercel CLI adds the local hostname to a request header. Windows permits
// non-Latin hostnames, while the Fetch Headers implementation accepts only
// ByteString values. Use an ASCII-only hostname for CLI requests.
os.hostname = () => "ARCH-EPVC-PC";
