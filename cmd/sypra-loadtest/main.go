// sypra-loadtest is a dependency-free, repeatable API load/soak runner. It is
// deliberately separate from the application binary so production deployments
// do not include test-only traffic generation code.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type result struct {
	duration time.Duration
	err      error
}

func main() {
	endpoint := flag.String("endpoint", "http://127.0.0.1:8080/api/v1/chat", "Sypra chat API endpoint")
	duration := flag.Duration("duration", 30*time.Second, "test duration")
	concurrency := flag.Int("concurrency", 4, "simultaneous requests")
	timeout := flag.Duration("timeout", 45*time.Second, "per-request timeout")
	maxErrorRate := flag.Float64("max-error-rate", .01, "maximum acceptable error ratio")
	maxP95 := flag.Duration("max-p95", 10*time.Second, "maximum acceptable p95 latency")
	flag.Parse()
	if *duration <= 0 || *concurrency < 1 || *timeout <= 0 {
		fatal("duration, concurrency, and timeout must be positive")
	}

	ctx, cancel := context.WithTimeout(context.Background(), *duration)
	defer cancel()
	client := &http.Client{Timeout: *timeout}
	results := make(chan result, *concurrency*4)
	var started sync.WaitGroup
	var requests uint64
	for worker := 0; worker < *concurrency; worker++ {
		started.Add(1)
		go func(worker int) {
			defer started.Done()
			for ctx.Err() == nil {
				sequence := atomic.AddUint64(&requests, 1)
				payload, _ := json.Marshal(map[string]any{"prompt": fmt.Sprintf("load validation worker=%d request=%d", worker, sequence)})
				request, err := http.NewRequestWithContext(ctx, http.MethodPost, *endpoint, bytes.NewReader(payload))
				if err != nil {
					results <- result{err: err}
					continue
				}
				request.Header.Set("Content-Type", "application/json")
				begin := time.Now()
				response, err := client.Do(request)
				elapsed := time.Since(begin)
				if response != nil {
					_ = response.Body.Close()
				}
				if ctx.Err() != nil {
					return
				}
				if err == nil && (response.StatusCode < 200 || response.StatusCode >= 300) {
					err = fmt.Errorf("HTTP %s", response.Status)
				}
				results <- result{duration: elapsed, err: err}
			}
		}(worker)
	}
	go func() { started.Wait(); close(results) }()

	latencies := make([]time.Duration, 0)
	errors := 0
	for outcome := range results {
		if outcome.err != nil {
			errors++
			continue
		}
		latencies = append(latencies, outcome.duration)
	}
	total := len(latencies) + errors
	if total == 0 {
		fatal("no requests completed")
	}
	if len(latencies) == 0 {
		fatal("all completed requests failed")
	}
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	p95 := latencies[(len(latencies)-1)*95/100]
	errorRate := float64(errors) / float64(total)
	throughput := float64(total) / duration.Seconds()
	fmt.Printf("requests=%d successes=%d errors=%d error_rate=%.4f throughput=%.2f/s p95=%s\n", total, len(latencies), errors, errorRate, throughput, p95.Round(time.Millisecond))
	if errorRate > *maxErrorRate || p95 > *maxP95 {
		os.Exit(1)
	}
}

func fatal(message string) {
	fmt.Fprintln(os.Stderr, "load test error:", message)
	os.Exit(2)
}
