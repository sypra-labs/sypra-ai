// Package websearch provides an intentionally small, opt-in web retrieval
// service. It is separate from local RAG: searched pages are not indexed or
// retained unless a caller explicitly saves them later.
package websearch

import (
	"context"
	"fmt"
	"html"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

const defaultSearchEndpoint = "https://html.duckduckgo.com/html/"
const fallbackSearchEndpoint = "https://www.bing.com/search"

type Result struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
	Content string `json:"content"`
}

type Service struct {
	SearchEndpoint string
	HTTP           *http.Client
	Logger         *slog.Logger
	MaxResults     int
	MaxPageBytes   int64
}

func New(endpoint ...string) Service {
	searchEndpoint := defaultSearchEndpoint
	if len(endpoint) > 0 && strings.TrimSpace(endpoint[0]) != "" {
		searchEndpoint = strings.TrimSpace(endpoint[0])
	}
	return Service{SearchEndpoint: searchEndpoint, HTTP: &http.Client{}, Logger: slog.Default()}
}

func (s Service) Search(ctx context.Context, query string) ([]Result, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, fmt.Errorf("web search query is required")
	}
	endpoint := s.SearchEndpoint
	if endpoint == "" {
		endpoint = defaultSearchEndpoint
	}
	client := s.HTTP
	if client == nil {
		client = New().HTTP
	}
	endpoints := []string{endpoint}
	if endpoint == defaultSearchEndpoint {
		endpoints = append(endpoints, fallbackSearchEndpoint)
	}
	var body []byte
	var results []Result
	var lastErr error
	for _, candidate := range endpoints {
		searchURL, err := url.Parse(candidate)
		if err != nil {
			lastErr = fmt.Errorf("web search endpoint %q: %w", candidate, err)
			continue
		}
		params := searchURL.Query()
		params.Set("q", query)
		searchURL.RawQuery = params.Encode()
		s.logger().Info("web search request", "endpoint", searchURL.String(), "query", query)
		body, _, err = fetch(ctx, client, searchURL.String(), 0)
		if err != nil {
			lastErr = err
			s.logger().Warn("web search provider failed", "endpoint", searchURL.String(), "error", err)
			continue
		}
		results = parseResults(string(body))
		if len(results) > 0 {
			break
		}
		lastErr = fmt.Errorf("search provider returned no parseable results")
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("search web: %w", lastErr)
	}
	if s.MaxResults > 0 && len(results) > s.MaxResults {
		results = results[:s.MaxResults]
	}
	maxPageBytes := s.MaxPageBytes
	s.logger().Info("web search results parsed", "resultCount", len(results), "responseBytes", len(body))
	for i := range results {
		page, contentType, pageErr := fetch(ctx, client, results[i].URL, maxPageBytes)
		if pageErr != nil {
			s.logger().Warn("web source fetch skipped", "url", results[i].URL, "error", pageErr)
			continue
		}
		if strings.Contains(strings.ToLower(contentType), "text/") || strings.Contains(strings.ToLower(contentType), "html") {
			results[i].Content = extractText(string(page))
		}
		if results[i].Content == "" {
			results[i].Content = results[i].Snippet
		}
		s.logger().Info("web source retrieved", "url", results[i].URL, "contentType", contentType, "contentChars", len([]rune(results[i].Content)))
	}
	s.logger().Info("web search completed", "resultCount", len(results))
	return results, nil
}
func (s Service) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}

func fetch(ctx context.Context, client *http.Client, rawURL string, maxBytes int64) ([]byte, string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return nil, "", fmt.Errorf("invalid public web URL")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", "Sypra/0.1 local web retrieval")
	response, err := client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		return nil, "", fmt.Errorf("%s", response.Status)
	}
	reader := io.Reader(response.Body)
	if maxBytes > 0 {
		reader = io.LimitReader(response.Body, maxBytes)
	}
	data, err := io.ReadAll(reader)
	return data, response.Header.Get("Content-Type"), err
}

var (
	resultLinkRE  = regexp.MustCompile(`(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)
	snippetRE     = regexp.MustCompile(`(?is)<[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</[^>]+>`)
	tagRE         = regexp.MustCompile(`(?is)<[^>]+>`)
	scriptRE      = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<noscript[^>]*>.*?</noscript>`)
	bingResultRE  = regexp.MustCompile(`(?is)<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?</h2>(.*?)</li>`)
	bingSnippetRE = regexp.MustCompile(`(?is)<p[^>]*>(.*?)</p>`)
)

func parseResults(page string) []Result {
	links, snippets := resultLinkRE.FindAllStringSubmatch(page, -1), snippetRE.FindAllStringSubmatch(page, -1)
	results := make([]Result, 0, len(links))
	for i, match := range links {
		rawURL := html.UnescapeString(match[1])
		resolved, err := url.Parse(rawURL)
		if err != nil {
			continue
		}
		if strings.Contains(resolved.Host, "duckduckgo.com") {
			if target := resolved.Query().Get("uddg"); target != "" {
				rawURL = target
			}
		}
		parsed, err := url.Parse(rawURL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
			continue
		}
		snippet := ""
		if i < len(snippets) {
			snippet = visibleText(snippets[i][1])
		}
		results = append(results, Result{Title: visibleText(match[2]), URL: rawURL, Snippet: snippet})
	}
	if len(results) > 0 {
		return results
	}
	for _, match := range bingResultRE.FindAllStringSubmatch(page, -1) {
		resultURL := html.UnescapeString(match[1])
		parsed, err := url.Parse(resultURL)
		if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
			continue
		}
		snippet := ""
		if paragraph := bingSnippetRE.FindStringSubmatch(match[3]); len(paragraph) > 1 {
			snippet = visibleText(paragraph[1])
		}
		results = append(results, Result{Title: visibleText(match[2]), URL: resultURL, Snippet: snippet})
	}
	return results
}

func extractText(page string) string { return visibleText(scriptRE.ReplaceAllString(page, " ")) }
func visibleText(value string) string {
	return strings.Join(strings.Fields(html.UnescapeString(tagRE.ReplaceAllString(value, " "))), " ")
}
