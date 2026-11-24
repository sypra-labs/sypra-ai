package observability

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"
)

// OTLPConfig enables standard OTLP/HTTP JSON exports to an OpenTelemetry
// Collector. Endpoint is a collector base URL such as http://localhost:4318.
// An empty endpoint leaves telemetry entirely local and creates no egress.
type OTLPConfig struct {
	Endpoint    string
	ServiceName string
	Logger      *slog.Logger
}

// Telemetry batches OpenTelemetry Protocol compatible traces and metrics. It
// is intentionally small and dependency-free so local-only deployments do not
// need a telemetry SDK or collector unless an endpoint is explicitly set.
type Telemetry struct {
	endpoint string
	service  string
	client   *http.Client
	logger   *slog.Logger
	events   chan telemetryEvent
	stop     chan struct{}
	done     chan struct{}
	once     sync.Once
}

type telemetryEvent struct {
	trace  otlpSpan
	metric otlpMetric
}
type otlpSpan struct {
	TraceID, SpanID, ParentSpanID, Name, Method, Path string
	Status                                            int
	Started, Ended                                    time.Time
}
type otlpMetric struct {
	Method, Path string
	Status       int
	Duration     time.Duration
	Count        int
}

// NewTelemetry returns nil when OTLP is not configured. This preserves
// Sypra's local-first default and makes collector use explicit.
func NewTelemetry(config OTLPConfig) (*Telemetry, error) {
	endpoint := strings.TrimRight(strings.TrimSpace(config.Endpoint), "/")
	if endpoint == "" {
		return nil, nil
	}
	if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
		return nil, fmt.Errorf("OTLP endpoint must start with http:// or https://")
	}
	service := strings.TrimSpace(config.ServiceName)
	if service == "" {
		service = "sypra-ai"
	}
	logger := config.Logger
	if logger == nil {
		logger = slog.Default()
	}
	telemetry := &Telemetry{endpoint: endpoint, service: service, client: &http.Client{Timeout: 5 * time.Second}, logger: logger, events: make(chan telemetryEvent, 1024), stop: make(chan struct{}), done: make(chan struct{})}
	go telemetry.run()
	return telemetry, nil
}

// HTTPRequest is one server span plus its correlated OTLP metric.
type HTTPRequest struct {
	telemetry *Telemetry
	span      otlpSpan
}

func (t *Telemetry) BeginHTTP(request *http.Request) *HTTPRequest {
	if t == nil {
		return nil
	}
	traceID, parentID := parseTraceparent(request.Header.Get("traceparent"))
	if traceID == "" {
		traceID = randomHex(16)
	}
	spanID := randomHex(8)
	return &HTTPRequest{telemetry: t, span: otlpSpan{TraceID: traceID, SpanID: spanID, ParentSpanID: parentID, Name: request.Method + " " + route(request.URL.Path), Method: request.Method, Path: route(request.URL.Path), Started: time.Now().UTC()}}
}
func (r *HTTPRequest) TraceID() string {
	if r == nil {
		return ""
	}
	return r.span.TraceID
}
func (r *HTTPRequest) Traceparent() string {
	if r == nil {
		return ""
	}
	return "00-" + r.span.TraceID + "-" + r.span.SpanID + "-01"
}
func (r *HTTPRequest) End(status int, duration time.Duration) {
	if r == nil || r.telemetry == nil {
		return
	}
	r.span.Status, r.span.Ended = status, time.Now().UTC()
	event := telemetryEvent{trace: r.span, metric: otlpMetric{Method: r.span.Method, Path: r.span.Path, Status: status, Duration: duration, Count: 1}}
	select {
	case r.telemetry.events <- event:
	default:
		r.telemetry.logger.Warn("OTLP telemetry queue is full; dropping event")
	}
}

func (t *Telemetry) run() {
	defer close(t.done)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	batch := make([]telemetryEvent, 0, 128)
	flush := func() {
		if len(batch) == 0 {
			return
		}
		t.export(batch)
		batch = batch[:0]
	}
	for {
		select {
		case event := <-t.events:
			batch = append(batch, event)
			if len(batch) >= 128 {
				flush()
			}
		case <-ticker.C:
			flush()
		case <-t.stop:
			// The stop signal can race with queued completed requests. Drain the
			// buffered channel before the final synchronous flush so shutdown
			// does not silently discard their traces or metrics.
			for {
				select {
				case event := <-t.events:
					batch = append(batch, event)
				default:
					flush()
					return
				}
			}
		}
	}
}

// Shutdown flushes queued telemetry with the caller's deadline.
func (t *Telemetry) Shutdown(ctx context.Context) error {
	if t == nil {
		return nil
	}
	t.once.Do(func() { close(t.stop) })
	select {
	case <-t.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (t *Telemetry) export(events []telemetryEvent) {
	spans := make([]map[string]any, 0, len(events))
	metrics := map[string]otlpMetric{}
	for _, event := range events {
		span := event.trace
		attributes := attributes(map[string]any{"http.request.method": span.Method, "url.path": span.Path, "http.response.status_code": span.Status})
		payload := map[string]any{"traceId": span.TraceID, "spanId": span.SpanID, "name": span.Name, "kind": 2, "startTimeUnixNano": strconvNano(span.Started), "endTimeUnixNano": strconvNano(span.Ended), "attributes": attributes}
		if span.ParentSpanID != "" {
			payload["parentSpanId"] = span.ParentSpanID
		}
		if span.Status >= 500 {
			payload["status"] = map[string]any{"code": 2, "message": "server error"}
		} else {
			payload["status"] = map[string]any{"code": 1}
		}
		spans = append(spans, payload)
		key := event.metric.Method + "\x00" + event.metric.Path + "\x00" + fmt.Sprint(event.metric.Status)
		current := metrics[key]
		current.Method, current.Path, current.Status = event.metric.Method, event.metric.Path, event.metric.Status
		current.Duration += event.metric.Duration
		current.Count += event.metric.Count
		metrics[key] = current
	}
	tracesPayload := map[string]any{
		"resourceSpans": []any{
			map[string]any{
				"resource": resource(t.service),
				"scopeSpans": []any{
					map[string]any{"scope": map[string]any{"name": "sypra"}, "spans": spans},
				},
			},
		},
	}
	if err := t.post("/v1/traces", tracesPayload); err != nil {
		t.logger.Warn("OTLP trace export failed", "error", err)
	}
	metricData := make([]any, 0, len(metrics)*2)
	now := time.Now().UTC()
	for _, value := range metrics {
		attributes := attributes(map[string]any{"http.request.method": value.Method, "url.path": value.Path, "http.response.status_code": value.Status})
		requestMetric := map[string]any{
			"name": "sypra.http.server.requests",
			"unit": "1",
			"sum": map[string]any{
				"aggregationTemporality": 2,
				"isMonotonic":            true,
				"dataPoints": []any{
					map[string]any{"attributes": attributes, "startTimeUnixNano": strconvNano(now), "timeUnixNano": strconvNano(now), "asInt": fmt.Sprint(value.Count)},
				},
			},
		}
		durationMetric := map[string]any{
			"name": "sypra.http.server.duration",
			"unit": "s",
			"gauge": map[string]any{
				"dataPoints": []any{
					map[string]any{"attributes": attributes, "timeUnixNano": strconvNano(now), "asDouble": value.Duration.Seconds()},
				},
			},
		}
		metricData = append(metricData, requestMetric, durationMetric)
	}
	metricsPayload := map[string]any{
		"resourceMetrics": []any{
			map[string]any{
				"resource": resource(t.service),
				"scopeMetrics": []any{
					map[string]any{"scope": map[string]any{"name": "sypra"}, "metrics": metricData},
				},
			},
		},
	}
	if err := t.post("/v1/metrics", metricsPayload); err != nil {
		t.logger.Warn("OTLP metric export failed", "error", err)
	}
}

func (t *Telemetry) post(path string, payload any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequest(http.MethodPost, t.endpoint+path, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := t.client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("collector returned %s", response.Status)
	}
	return nil
}

func resource(service string) map[string]any {
	return map[string]any{"attributes": attributes(map[string]any{"service.name": service})}
}
func attributes(values map[string]any) []any {
	result := make([]any, 0, len(values))
	for key, value := range values {
		switch typed := value.(type) {
		case int:
			result = append(result, map[string]any{"key": key, "value": map[string]any{"intValue": fmt.Sprint(typed)}})
		default:
			result = append(result, map[string]any{"key": key, "value": map[string]any{"stringValue": fmt.Sprint(value)}})
		}
	}
	return result
}
func strconvNano(value time.Time) string { return fmt.Sprint(value.UnixNano()) }
func randomHex(size int) string {
	buffer := make([]byte, size)
	if _, err := rand.Read(buffer); err != nil {
		return fmt.Sprintf("%0*x", size*2, time.Now().UnixNano())
	}
	return hex.EncodeToString(buffer)
}
func parseTraceparent(value string) (traceID, parentID string) {
	parts := strings.Split(value, "-")
	if len(parts) != 4 || len(parts[1]) != 32 || len(parts[2]) != 16 {
		return "", ""
	}
	if _, err := hex.DecodeString(parts[1]); err != nil {
		return "", ""
	}
	if _, err := hex.DecodeString(parts[2]); err != nil {
		return "", ""
	}
	return strings.ToLower(parts[1]), strings.ToLower(parts[2])
}
