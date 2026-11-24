package domain

import "time"

type Model struct {
	ID          string             `json:"id"`
	Name        string             `json:"name"`
	Path        string             `json:"path"`
	Format      string             `json:"format"`
	Endpoint    string             `json:"endpoint"`
	Provider    string             `json:"provider,omitempty"`
	RemoteID    string             `json:"remoteId,omitempty"`
	Family      string             `json:"family,omitempty"`
	Modalities  []string           `json:"modalities,omitempty"`
	Kind        string             `json:"kind"`
	Default     bool               `json:"default,omitempty"`
	ParentModel string             `json:"parentModel,omitempty"`
	Method      string             `json:"method,omitempty"`
	Version     string             `json:"version,omitempty"`
	Status      string             `json:"status,omitempty"`
	Metrics     map[string]float64 `json:"metrics,omitempty"`
	PromotedAt  *time.Time         `json:"promotedAt,omitempty"`
	CreatedAt   time.Time          `json:"createdAt"`
}
type Collection struct {
	ID             string `json:"id"`
	Name           string `json:"name"`
	EmbeddingModel string `json:"embeddingModel"`
	// ChunkUnit is "tokens" for collections indexed by the model-aware
	// chunker. Collections created by earlier Sypra versions omit it.
	ChunkUnit        string `json:"chunkUnit,omitempty"`
	EmbeddingVersion string `json:"embeddingVersion,omitempty"`
	ChunkSize        int    `json:"chunkSize"`
	ChunkOverlap     int    `json:"chunkOverlap"`
	DocumentCount    int    `json:"documentCount"`
	// RoutingTags are optional labels used by the RAG router to select this
	// collection for an otherwise unbound retrieval request.
	RoutingTags []string  `json:"routingTags,omitempty"`
	CreatedAt   time.Time `json:"createdAt"`
}

// RAGDocument is the durable source record behind an indexed document. Chunks
// deliberately retain a document ID rather than merely a source filename, so
// duplicate filenames, replacements and audit trails remain unambiguous.
type RAGDocument struct {
	ID                     string            `json:"id"`
	CollectionID           string            `json:"collectionId"`
	Name                   string            `json:"name"`
	Source                 string            `json:"source"`
	Path                   string            `json:"path,omitempty"`
	URL                    string            `json:"url,omitempty"`
	Text                   string            `json:"text,omitempty"`
	ContentType            string            `json:"contentType,omitempty"`
	Metadata               map[string]string `json:"metadata,omitempty"`
	ExtractionStatus       string            `json:"extractionStatus"`
	ExtractionError        string            `json:"extractionError,omitempty"`
	Extractor              string            `json:"extractor,omitempty"`
	ChunkCount             int               `json:"chunkCount"`
	RefreshIntervalMinutes int               `json:"refreshIntervalMinutes,omitempty"`
	LastRefreshedAt        *time.Time        `json:"lastRefreshedAt,omitempty"`
	NextRefreshAt          *time.Time        `json:"nextRefreshAt,omitempty"`
	CreatedAt              time.Time         `json:"createdAt"`
	UpdatedAt              time.Time         `json:"updatedAt"`
}
type Project struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"createdAt"`
}
type Chat struct {
	ID           string `json:"id"`
	Title        string `json:"title"`
	ProjectID    string `json:"projectId,omitempty"`
	ModelID      string `json:"modelId,omitempty"`
	CollectionID string `json:"collectionId,omitempty"`
	RAGMode      string `json:"ragMode,omitempty"`
	RAGTopK      int    `json:"ragTopK,omitempty"`
	// RAGEnabled is nil for legacy chats. A bound legacy collection defaults to
	// enabled, while a stored false explicitly disables retrieval for the next turn.
	RAGEnabled *bool `json:"ragEnabled,omitempty"`
	WebEnabled bool  `json:"webEnabled,omitempty"`
	Pinned     bool  `json:"pinned,omitempty"`
	// WorkspacePath is set only for terminal chats. It is immutable after the
	// first CLI turn and identifies the folder that owns the conversation.
	WorkspacePath    string    `json:"workspacePath,omitempty"`
	CLISessionActive bool      `json:"cliSessionActive,omitempty"`
	CreatedAt        time.Time `json:"createdAt"`
	UpdatedAt        time.Time `json:"updatedAt"`
}
type ChatMessage struct {
	ID        string        `json:"id"`
	ChatID    string        `json:"chatId"`
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	Parts     []ContentPart `json:"parts,omitempty"`
	Sources   []string      `json:"sources,omitempty"`
	Status    string        `json:"status,omitempty"`
	LatencyMS int64         `json:"latencyMs,omitempty"`
	CreatedAt time.Time     `json:"createdAt"`
}
type Chunk struct {
	ID           string            `json:"id"`
	CollectionID string            `json:"collectionId"`
	DocumentID   string            `json:"documentId,omitempty"`
	Source       string            `json:"source"`
	Content      string            `json:"content"`
	ContentType  string            `json:"contentType,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
	Embedding    []float32         `json:"embedding,omitempty"`
	Position     int               `json:"position"`
	CreatedAt    time.Time         `json:"createdAt"`
}

// ChunkMatch supplies inspectable retrieval evidence. Scores are retained
// separately so the UI can explain hybrid vector/lexical retrieval instead of
// presenting an opaque source citation.
type ChunkMatch struct {
	Chunk         Chunk   `json:"chunk"`
	Score         float64 `json:"score"`
	SemanticScore float64 `json:"semanticScore"`
	LexicalScore  float64 `json:"lexicalScore"`
}

type RAGEvaluationCase struct {
	ID                  string   `json:"id"`
	Query               string   `json:"query"`
	RelevantDocumentIDs []string `json:"relevantDocumentIds,omitempty"`
	RelevantChunkIDs    []string `json:"relevantChunkIds,omitempty"`
	ExpectedAnswer      string   `json:"expectedAnswer,omitempty"`
}

type RAGEvaluationSet struct {
	ID           string              `json:"id"`
	CollectionID string              `json:"collectionId"`
	Name         string              `json:"name"`
	Cases        []RAGEvaluationCase `json:"cases"`
	CreatedAt    time.Time           `json:"createdAt"`
	UpdatedAt    time.Time           `json:"updatedAt"`
}

type RAGEvaluationResult struct {
	CaseID     string   `json:"caseId"`
	Query      string   `json:"query"`
	Retrieved  []string `json:"retrievedChunkIds"`
	Hit        bool     `json:"hit"`
	Rank       int      `json:"rank,omitempty"`
	Recall     float64  `json:"recall"`
	Precision  float64  `json:"precision"`
	Reciprocal float64  `json:"reciprocalRank"`
}

type RAGEvaluationRun struct {
	ID           string                `json:"id"`
	EvaluationID string                `json:"evaluationId"`
	CollectionID string                `json:"collectionId"`
	Limit        int                   `json:"limit"`
	Metrics      map[string]float64    `json:"metrics"`
	Results      []RAGEvaluationResult `json:"results"`
	CreatedAt    time.Time             `json:"createdAt"`
}

// MediaJob is a local, observable workflow for OCR, transcription, image
// generation, and image editing. OutputURL is an API-served local artifact;
// no source media is sent off-machine unless the configured command does so.
type MediaJob struct {
	ID         string     `json:"id"`
	Kind       string     `json:"kind"`
	Prompt     string     `json:"prompt,omitempty"`
	InputPath  string     `json:"inputPath,omitempty"`
	InputName  string     `json:"inputName,omitempty"`
	Status     string     `json:"status"`
	Phase      string     `json:"phase,omitempty"`
	Progress   float64    `json:"progress"`
	OutputPath string     `json:"outputPath,omitempty"`
	OutputURL  string     `json:"outputUrl,omitempty"`
	OutputMIME string     `json:"outputMime,omitempty"`
	Text       string     `json:"text,omitempty"`
	Error      string     `json:"error,omitempty"`
	CreatedAt  time.Time  `json:"createdAt"`
	StartedAt  *time.Time `json:"startedAt,omitempty"`
	FinishedAt *time.Time `json:"finishedAt,omitempty"`
}

// ToolConnection describes a local extension endpoint. MCP connections use
// stdio JSON-RPC; pipeline connections compose registered tools into an
// explicit, auditable sequence. Commands are never granted to a model until a
// matching ToolPermission is enabled.
type ToolConnection struct {
	ID          string                    `json:"id"`
	Name        string                    `json:"name"`
	Kind        string                    `json:"kind"` // mcp or pipeline
	Command     string                    `json:"command,omitempty"`
	Args        []string                  `json:"args,omitempty"`
	ToolIDs     []string                  `json:"toolIds,omitempty"`
	ToolSchemas map[string]map[string]any `json:"toolSchemas,omitempty"`
	Enabled     bool                      `json:"enabled"`
	Status      string                    `json:"status,omitempty"`
	Error       string                    `json:"error,omitempty"`
	CreatedAt   time.Time                 `json:"createdAt"`
	UpdatedAt   time.Time                 `json:"updatedAt"`
}

// ToolPermission is a deny-by-default policy. An empty ModelID or ProjectID
// is a wildcard for that dimension. RequireApproval keeps the tool visible to
// an agent while requiring an explicit task-level approval flag to execute.
type ToolPermission struct {
	ID              string    `json:"id"`
	ToolID          string    `json:"toolId"`
	ModelID         string    `json:"modelId,omitempty"`
	ProjectID       string    `json:"projectId,omitempty"`
	Enabled         bool      `json:"enabled"`
	RequireApproval bool      `json:"requireApproval"`
	CreatedAt       time.Time `json:"createdAt"`
}

type ToolDefinition struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Source      string         `json:"source"`
	InputSchema map[string]any `json:"inputSchema"`
}

type AgentRun struct {
	ID         string         `json:"id"`
	ModelID    string         `json:"modelId,omitempty"`
	ProjectID  string         `json:"projectId,omitempty"`
	ToolID     string         `json:"toolId"`
	Arguments  map[string]any `json:"arguments"`
	Status     string         `json:"status"`
	Output     string         `json:"output,omitempty"`
	Error      string         `json:"error,omitempty"`
	CreatedAt  time.Time      `json:"createdAt"`
	FinishedAt *time.Time     `json:"finishedAt,omitempty"`
}
type TrainingJob struct {
	ID              string         `json:"id"`
	Name            string         `json:"name"`
	BaseModel       string         `json:"baseModel"`
	DatasetPath     string         `json:"datasetPath"`
	Method          string         `json:"method"`
	Parameters      map[string]any `json:"parameters"`
	Status          string         `json:"status"`
	Progress        float64        `json:"progress"`
	Phase           string         `json:"phase,omitempty"`
	Epoch           int            `json:"epoch,omitempty"`
	Epochs          int            `json:"epochs,omitempty"`
	Step            int            `json:"step,omitempty"`
	Steps           int            `json:"steps,omitempty"`
	Tokens          int            `json:"tokens,omitempty"`
	TokensPerSecond float64        `json:"tokensPerSecond,omitempty"`
	TrainLoss       *float64       `json:"trainLoss,omitempty"`
	EvalLoss        *float64       `json:"evalLoss,omitempty"`
	CreatedAt       time.Time      `json:"createdAt"`
	StartedAt       *time.Time     `json:"startedAt,omitempty"`
	FinishedAt      *time.Time     `json:"finishedAt,omitempty"`
	Error           string         `json:"error,omitempty"`
}
type UsageEvent struct {
	ID           string    `json:"id"`
	Kind         string    `json:"kind"`
	ModelID      string    `json:"modelId"`
	InputTokens  int       `json:"inputTokens"`
	OutputTokens int       `json:"outputTokens"`
	LatencyMS    int64     `json:"latencyMs"`
	CreatedAt    time.Time `json:"createdAt"`
}
type UsageSummary struct {
	Requests         int     `json:"requests"`
	InputTokens      int     `json:"inputTokens"`
	OutputTokens     int     `json:"outputTokens"`
	AverageLatencyMS float64 `json:"averageLatencyMs"`
	TrainingJobs     int     `json:"trainingJobs"`
}
