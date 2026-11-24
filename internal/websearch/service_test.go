package websearch

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSearchFetchesReadableResultContent(t *testing.T) {
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/search":
			_, _ = w.Write([]byte(`<a class="result__a" href="` + server.URL + `/article">Example source</a><div class="result__snippet">A short description.</div>`))
		case "/article":
			_, _ = w.Write([]byte(`<html><head><style>body{display:none}</style></head><body><h1>Release notes</h1><p>The local result is readable.</p><script>ignore()</script></body></html>`))
		}
	}))
	defer server.Close()

	service := New()
	service.SearchEndpoint = server.URL + "/search"
	service.MaxResults = 1
	results, err := service.Search(context.Background(), "release notes")
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Title != "Example source" || results[0].Content != "Release notes The local result is readable." {
		t.Fatalf("unexpected results: %#v", results)
	}
}

func TestParseResultsSupportsBingMarkup(t *testing.T) {
	page := `<li class="b_algo"><h2><a href="https://example.com/article">Example result</a></h2><div><p>Readable result summary.</p></div></li>`
	results := parseResults(page)
	if len(results) != 1 || results[0].Title != "Example result" || results[0].URL != "https://example.com/article" || results[0].Snippet != "Readable result summary." {
		t.Fatalf("unexpected Bing result parse: %#v", results)
	}
}
