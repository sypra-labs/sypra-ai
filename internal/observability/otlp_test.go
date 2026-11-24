package observability

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestTelemetryExportsOTLPTraceAndMetrics(t *testing.T) {
	var mu sync.Mutex
	payloads := map[string]string{}
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		payloads[r.URL.Path] = string(body)
		mu.Unlock()
		w.WriteHeader(http.StatusAccepted)
	}))
	defer collector.Close()
	telemetry, err := NewTelemetry(OTLPConfig{Endpoint: collector.URL, ServiceName: "sypra-test"})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/api/v1/chats/abc", nil)
	request.Header.Set("traceparent", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
	span := telemetry.BeginHTTP(request)
	if span.TraceID() != "0af7651916cd43dd8448eb211c80319c" || span.Traceparent() == "" {
		t.Fatalf("unexpected trace propagation: id=%q parent=%q", span.TraceID(), span.Traceparent())
	}
	span.End(http.StatusOK, 25*time.Millisecond)
	if err = telemetry.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if payloads["/v1/traces"] == "" || payloads["/v1/metrics"] == "" {
		t.Fatalf("missing OTLP payloads: %#v", payloads)
	}
	if !strings.Contains(payloads["/v1/traces"], `"service.name"`) || !strings.Contains(payloads["/v1/metrics"], `"sypra.http.server.requests"`) {
		t.Fatalf("invalid OTLP payloads: %#v", payloads)
	}
}
