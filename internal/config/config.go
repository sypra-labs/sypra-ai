package config

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	ListenAddr, DataPath, ModelEndpoint, ModelAPIKey, CORSOrigin, WebSearchEndpoint string
	OpenAIAPIKey, XAIAPIKey, GeminiAPIKey                                           string
	AgentTerminalAllowlist, AgentMCPAllowlist, AgentWorkDir                         string
	DatabaseURL, LogFormat, LogFile                                                 string
	OTLPEndpoint, OTLPServiceName                                                   string
	LogMaxBytes, LogBackups, MaxConcurrentJobs, MaxActiveRunners, RunnerGPUBudgetMB int
	AgentTimeoutSeconds, AgentMaxOutputBytes                                        int
}

func Load() Config {
	loadDotEnv(".env")
	legacyKey := os.Getenv("SYPRA_MODEL_API_KEY")
	return Config{
		ListenAddr:             value("SYPRA_LISTEN_ADDR", ":8080"),
		DataPath:               value("SYPRA_DATA_PATH", "./data/sypra.json"),
		ModelEndpoint:          strings.TrimSpace(os.Getenv("SYPRA_MODEL_ENDPOINT")),
		ModelAPIKey:            legacyKey,
		CORSOrigin:             value("SYPRA_CORS_ORIGIN", "http://localhost:5173"),
		WebSearchEndpoint:      strings.TrimSpace(os.Getenv("SYPRA_WEB_SEARCH_ENDPOINT")),
		OpenAIAPIKey:           value("SYPRA_OPENAI_API_KEY", legacyKey),
		XAIAPIKey:              os.Getenv("SYPRA_XAI_API_KEY"),
		GeminiAPIKey:           os.Getenv("SYPRA_GEMINI_API_KEY"),
		AgentTerminalAllowlist: strings.TrimSpace(os.Getenv("SYPRA_AGENT_TERMINAL_ALLOWLIST")),
		AgentMCPAllowlist:      strings.TrimSpace(os.Getenv("SYPRA_AGENT_MCP_ALLOWLIST")),
		AgentWorkDir:           strings.TrimSpace(os.Getenv("SYPRA_AGENT_WORK_DIR")),
		DatabaseURL:            value("SYPRA_DATABASE_URL", "sqlite://./data/sypra.db"),
		LogFormat:              value("SYPRA_LOG_FORMAT", "json"),
		LogFile:                strings.TrimSpace(os.Getenv("SYPRA_LOG_FILE")),
		OTLPEndpoint:           strings.TrimSpace(os.Getenv("SYPRA_OTLP_ENDPOINT")),
		OTLPServiceName:        value("SYPRA_OTLP_SERVICE_NAME", "sypra-ai"),
		LogMaxBytes:            intValue("SYPRA_LOG_MAX_BYTES", 10*1024*1024),
		LogBackups:             intValue("SYPRA_LOG_BACKUPS", 5),
		MaxConcurrentJobs:      intValue("SYPRA_MAX_CONCURRENT_JOBS", 1),
		MaxActiveRunners:       intValue("SYPRA_MAX_ACTIVE_RUNNERS", 2),
		RunnerGPUBudgetMB:      intValue("SYPRA_RUNNER_GPU_BUDGET_MB", 0),
		AgentTimeoutSeconds:    intValue("SYPRA_AGENT_TIMEOUT_SECONDS", 30),
		AgentMaxOutputBytes:    intValue("SYPRA_AGENT_MAX_OUTPUT_BYTES", 64<<10),
	}
}
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		sep := strings.Index(line, "=")
		if sep < 0 {
			sep = strings.Index(line, ":")
		}
		if sep < 1 {
			continue
		}
		key := strings.TrimSpace(line[:sep])
		value := strings.Trim(strings.TrimSpace(line[sep+1:]), "\"'")
		if key != "" && os.Getenv(key) == "" {
			_ = os.Setenv(key, value)
		}
	}
}
func value(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}
func intValue(k string, fallback int) int {
	v, err := strconv.Atoi(strings.TrimSpace(os.Getenv(k)))
	if err != nil || v < 0 {
		return fallback
	}
	return v
}
