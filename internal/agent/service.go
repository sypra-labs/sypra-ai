// Package agent provides Sypra's local-first tool control plane. Models never
// receive ambient shell/MCP access: every execution is checked against an
// explicit model/project policy before a tool process or network search starts.
package agent

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/sypra-labs/sypra-ai/internal/domain"
	"github.com/sypra-labs/sypra-ai/internal/store"
	"github.com/sypra-labs/sypra-ai/internal/websearch"
)

type Service struct {
	Store             *store.Store
	Web               websearch.Service
	TerminalAllowlist map[string]bool
	MCPAllowlist      map[string]bool
	WorkRoot          string
	Timeout           time.Duration
	MaxOutputBytes    int
}

type Options struct {
	MCPAllowlist   []string
	WorkRoot       string
	Timeout        time.Duration
	MaxOutputBytes int
}

func New(st *store.Store, web websearch.Service, terminalAllowlist []string, options ...Options) *Service {
	option := Options{Timeout: 30 * time.Second, MaxOutputBytes: 64 << 10}
	if len(options) > 0 {
		option = options[0]
	}
	if option.Timeout < time.Second {
		option.Timeout = 30 * time.Second
	}
	if option.MaxOutputBytes < 1024 {
		option.MaxOutputBytes = 64 << 10
	}
	return &Service{Store: st, Web: web, TerminalAllowlist: trustedSet(terminalAllowlist), MCPAllowlist: trustedSet(option.MCPAllowlist), WorkRoot: strings.TrimSpace(option.WorkRoot), Timeout: option.Timeout, MaxOutputBytes: option.MaxOutputBytes}
}

func trustedSet(commands []string) map[string]bool {
	allowed := map[string]bool{}
	for _, command := range commands {
		if command = strings.TrimSpace(command); command != "" {
			allowed[strings.ToLower(filepath.Clean(command))] = true
		}
	}
	return allowed
}

func (s *Service) Catalog(ctx context.Context) ([]domain.ToolDefinition, error) {
	tools := []domain.ToolDefinition{
		{ID: "browser.search", Name: "Browser search", Description: "Search the web when explicitly enabled for an agent task.", Source: "builtin", InputSchema: objectSchema(map[string]any{"query": map[string]any{"type": "string", "minLength": 1}}, "query")},
		{ID: "terminal.run", Name: "Terminal command", Description: "Run a trusted local command in an isolated working directory with explicit arguments.", Source: "builtin", InputSchema: objectSchema(map[string]any{"command": map[string]any{"type": "string", "minLength": 1}, "args": map[string]any{"type": "array", "items": map[string]any{"type": "string"}}}, "command")},
	}
	connections, err := s.Store.ToolConnections(ctx)
	if err != nil {
		return nil, err
	}
	for _, connection := range connections {
		if !connection.Enabled {
			continue
		}
		switch connection.Kind {
		case "pipeline":
			tools = append(tools, domain.ToolDefinition{ID: "pipeline/" + connection.ID, Name: connection.Name, Description: "Configured pipeline: " + strings.Join(connection.ToolIDs, " → "), Source: "pipeline", InputSchema: objectSchema(map[string]any{})})
		case "mcp":
			for _, toolID := range connection.ToolIDs {
				schema := connection.ToolSchemas[toolID]
				if schema == nil {
					schema = objectSchema(map[string]any{})
				}
				tools = append(tools, domain.ToolDefinition{ID: "mcp/" + connection.ID + "/" + toolID, Name: toolID, Description: "Tool discovered from MCP connection " + connection.Name + ".", Source: "mcp", InputSchema: schema})
			}
		}
	}
	sort.SliceStable(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
	return tools, nil
}

// Discover validates an MCP stdio command once and returns its advertised
// tools. The caller persists the names, avoiding process launches during UI
// polling or ordinary catalog reads.
func (s *Service) Discover(ctx context.Context, connection domain.ToolConnection) ([]mcpTool, error) {
	tools, err := s.listMCP(ctx, connection)
	if err != nil {
		return nil, err
	}
	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
	return tools, nil
}

func objectSchema(properties map[string]any, required ...string) map[string]any {
	schema := map[string]any{"type": "object", "properties": properties, "additionalProperties": false}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

func (s *Service) Execute(ctx context.Context, modelID, projectID, toolID string, arguments map[string]any, approved bool) (domain.AgentRun, error) {
	requiresApproval, err := s.authorize(ctx, modelID, projectID, toolID)
	if err != nil {
		return domain.AgentRun{}, err
	}
	if err := s.validateArguments(ctx, toolID, arguments); err != nil {
		return domain.AgentRun{}, err
	}
	now := store.Now()
	run := domain.AgentRun{ID: newID(), ModelID: modelID, ProjectID: projectID, ToolID: toolID, Arguments: arguments, Status: "running", CreatedAt: now}
	if requiresApproval && !approved {
		run.Status = "awaiting_approval"
		if err := s.Store.AddAgentRun(ctx, run); err != nil {
			return domain.AgentRun{}, err
		}
		return run, nil
	}
	if err := s.Store.AddAgentRun(ctx, run); err != nil {
		return domain.AgentRun{}, err
	}
	return s.executeRecorded(ctx, run)
}

func (s *Service) executeRecorded(ctx context.Context, run domain.AgentRun) (domain.AgentRun, error) {
	runContext, cancel := context.WithTimeout(ctx, s.Timeout)
	defer cancel()
	output, err := s.invoke(runContext, run.ToolID, run.Arguments)
	finished := store.Now()
	run.FinishedAt = &finished
	if err != nil {
		run.Status, run.Error = "failed", err.Error()
	} else {
		run.Status, run.Output = "completed", output
	}
	if updateErr := s.Store.UpdateAgentRun(context.Background(), run); updateErr != nil && err == nil {
		err = updateErr
	}
	return run, err
}

// Approve resumes one previously validated request. It never reuses arguments
// supplied by the approval UI, preventing approval from being confused with a
// second, modified tool request.
func (s *Service) Approve(ctx context.Context, runID string) (domain.AgentRun, error) {
	runs, err := s.Store.AgentRuns(ctx)
	if err != nil {
		return domain.AgentRun{}, err
	}
	for _, run := range runs {
		if run.ID == runID {
			if run.Status != "awaiting_approval" {
				return domain.AgentRun{}, fmt.Errorf("agent run %s is not awaiting approval", runID)
			}
			run.Status = "running"
			if err := s.Store.UpdateAgentRun(ctx, run); err != nil {
				return domain.AgentRun{}, err
			}
			return s.executeRecorded(ctx, run)
		}
	}
	return domain.AgentRun{}, fmt.Errorf("agent run %s not found", runID)
}

// Callable returns only tools enabled by a matching policy. Models never see
// denied tools; tools requiring approval remain visible so their requested run
// can enter the auditable approval queue.
func (s *Service) Callable(ctx context.Context, modelID, projectID string) ([]domain.ToolDefinition, error) {
	tools, err := s.Catalog(ctx)
	if err != nil {
		return nil, err
	}
	result := make([]domain.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if _, err := s.authorize(ctx, modelID, projectID, tool.ID); err == nil {
			result = append(result, tool)
		}
	}
	return result, nil
}

func (s *Service) authorize(ctx context.Context, modelID, projectID, toolID string) (bool, error) {
	permissions, err := s.Store.ToolPermissions(ctx)
	if err != nil {
		return false, err
	}
	for _, permission := range permissions {
		if !permission.Enabled || (permission.ToolID != toolID && permission.ToolID != "*") {
			continue
		}
		if permission.ModelID != "" && permission.ModelID != modelID {
			continue
		}
		if permission.ProjectID != "" && permission.ProjectID != projectID {
			continue
		}
		return permission.RequireApproval, nil
	}
	return false, fmt.Errorf("tool %s is denied: add an enabled model/project permission first", toolID)
}

func (s *Service) invoke(ctx context.Context, toolID string, arguments map[string]any) (string, error) {
	switch toolID {
	case "browser.search":
		query, _ := arguments["query"].(string)
		results, err := s.Web.Search(ctx, query)
		if err != nil {
			return "", err
		}
		lines := make([]string, 0, len(results))
		for _, result := range results {
			lines = append(lines, result.Title+"\n"+result.URL+"\n"+result.Snippet)
		}
		return strings.Join(lines, "\n\n"), nil
	case "terminal.run":
		return s.runTerminal(ctx, arguments)
	}
	if strings.HasPrefix(toolID, "mcp/") {
		return s.callMCP(ctx, toolID, arguments)
	}
	if strings.HasPrefix(toolID, "pipeline/") {
		return s.runPipeline(ctx, strings.TrimPrefix(toolID, "pipeline/"), arguments)
	}
	return "", fmt.Errorf("unknown tool %s", toolID)
}

func (s *Service) runTerminal(ctx context.Context, arguments map[string]any) (string, error) {
	command, _ := arguments["command"].(string)
	command, err := s.trustedExecutable(command, s.TerminalAllowlist, "terminal")
	if err != nil {
		return "", err
	}
	args := []string{}
	if raw, ok := arguments["args"].([]any); ok {
		for _, value := range raw {
			text, ok := value.(string)
			if !ok {
				return "", fmt.Errorf("terminal args must contain only strings")
			}
			args = append(args, text)
		}
	}
	process := exec.CommandContext(ctx, command, args...)
	cleanup, err := s.sandboxCommand(process)
	if err != nil {
		return "", err
	}
	defer cleanup()
	output := &boundedBuffer{limit: s.MaxOutputBytes}
	process.Stdout, process.Stderr = output, output
	err = process.Run()
	if output.exceeded || errors.Is(err, errOutputLimit) || errors.Is(outputErr(err), errOutputLimit) {
		return "", fmt.Errorf("terminal output exceeded %d bytes", s.MaxOutputBytes)
	}
	if ctx.Err() != nil {
		return "", fmt.Errorf("terminal tool exceeded %s", s.Timeout)
	}
	if err != nil {
		return "", fmt.Errorf("terminal tool failed: %w: %s", err, strings.TrimSpace(output.String()))
	}
	return output.String(), nil
}

func outputErr(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, errOutputLimit) {
		return err
	}
	return nil
}

func (s *Service) connection(ctx context.Context, id string) (domain.ToolConnection, error) {
	connections, err := s.Store.ToolConnections(ctx)
	if err != nil {
		return domain.ToolConnection{}, err
	}
	for _, connection := range connections {
		if connection.ID == id {
			return connection, nil
		}
	}
	return domain.ToolConnection{}, fmt.Errorf("MCP connection %s not found", id)
}

func (s *Service) callMCP(ctx context.Context, toolID string, arguments map[string]any) (string, error) {
	parts := strings.SplitN(strings.TrimPrefix(toolID, "mcp/"), "/", 2)
	if len(parts) != 2 {
		return "", fmt.Errorf("invalid MCP tool ID")
	}
	connection, err := s.connection(ctx, parts[0])
	if err != nil {
		return "", err
	}
	result, err := s.callMCPServer(ctx, connection, "tools/call", map[string]any{"name": parts[1], "arguments": arguments})
	if err != nil {
		return "", err
	}
	encoded, _ := json.MarshalIndent(result, "", "  ")
	return string(encoded), nil
}

func (s *Service) runPipeline(ctx context.Context, id string, arguments map[string]any) (string, error) {
	connection, err := s.connection(ctx, id)
	if err != nil {
		return "", err
	}
	if connection.Kind != "pipeline" {
		return "", fmt.Errorf("connection %s is not a pipeline", id)
	}
	outputs := make([]string, 0, len(connection.ToolIDs))
	for _, toolID := range connection.ToolIDs {
		output, err := s.invoke(ctx, toolID, arguments)
		if err != nil {
			return "", fmt.Errorf("pipeline %s at %s: %w", connection.Name, toolID, err)
		}
		outputs = append(outputs, "## "+toolID+"\n"+output)
	}
	return strings.Join(outputs, "\n\n"), nil
}

type mcpTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

func (s *Service) listMCP(ctx context.Context, connection domain.ToolConnection) ([]mcpTool, error) {
	result, err := s.callMCPServer(ctx, connection, "tools/list", map[string]any{})
	if err != nil {
		return nil, err
	}
	bytes, _ := json.Marshal(result)
	var payload struct {
		Tools []mcpTool `json:"tools"`
	}
	if err := json.Unmarshal(bytes, &payload); err != nil {
		return nil, err
	}
	return payload.Tools, nil
}

func (s *Service) callMCPServer(ctx context.Context, connection domain.ToolConnection, method string, params any) (any, error) {
	if connection.Kind != "mcp" || !connection.Enabled || strings.TrimSpace(connection.Command) == "" {
		return nil, fmt.Errorf("MCP connection is not enabled/configured")
	}
	trusted, err := s.trustedExecutable(connection.Command, s.MCPAllowlist, "MCP")
	if err != nil {
		return nil, err
	}
	command := exec.CommandContext(ctx, trusted, connection.Args...)
	cleanup, err := s.sandboxCommand(command)
	if err != nil {
		return nil, err
	}
	defer cleanup()
	stdin, err := command.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := command.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := command.Start(); err != nil {
		return nil, fmt.Errorf("start MCP server: %w", err)
	}
	defer func() { _ = command.Process.Kill(); _ = command.Wait() }()
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 4096), s.MaxOutputBytes)
	outputBytes := 0
	request := func(id int, name string, values any) (any, error) {
		payload := map[string]any{"jsonrpc": "2.0", "id": id, "method": name, "params": values}
		encoded, _ := json.Marshal(payload)
		if _, err := stdin.Write(append(encoded, '\n')); err != nil {
			return nil, err
		}
		for scanner.Scan() {
			outputBytes += len(scanner.Bytes())
			if outputBytes > s.MaxOutputBytes {
				return nil, fmt.Errorf("MCP output exceeded %d bytes", s.MaxOutputBytes)
			}
			var response struct {
				ID     int `json:"id"`
				Result any `json:"result"`
				Error  *struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if json.Unmarshal(scanner.Bytes(), &response) != nil || response.ID != id {
				continue
			}
			if response.Error != nil {
				return nil, fmt.Errorf("MCP %s: %s", name, response.Error.Message)
			}
			return response.Result, nil
		}
		if err := scanner.Err(); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("MCP server closed before %s", name)
	}
	if _, err := request(1, "initialize", map[string]any{"protocolVersion": "2024-11-05", "capabilities": map[string]any{}, "clientInfo": map[string]string{"name": "sypra", "version": "1"}}); err != nil {
		return nil, err
	}
	initialized, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized", "params": map[string]any{}})
	if _, err := stdin.Write(append(initialized, '\n')); err != nil {
		return nil, err
	}
	return request(2, method, params)
}

func newID() string {
	bytes := make([]byte, 16)
	_, _ = rand.Read(bytes)
	return fmt.Sprintf("%x-%x-%x-%x-%x", bytes[0:4], bytes[4:6], bytes[6:8], bytes[8:10], bytes[10:])
}

func DefaultTimeout() time.Duration { return 60 * time.Second }
