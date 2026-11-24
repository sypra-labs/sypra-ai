// Package webui serves the prebuilt browser application distributed beside a
// packaged Sypra binary. Development continues to use Vite on port 5173.
package webui

import (
	"net/http"
	"os"
	"path/filepath"
)

// Handler locates a packaged web directory beside the executable, then falls
// back to the development build directory. It returns false when no built UI
// is available, allowing API-only contributor workflows to remain unchanged.
func Handler() (http.Handler, bool) {
	candidates := make([]string, 0, 2)
	if executable, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(executable), "web"))
	}
	if workingDirectory, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(workingDirectory, "web", "dist"))
	}
	for _, directory := range candidates {
		if info, err := os.Stat(filepath.Join(directory, "index.html")); err == nil && !info.IsDir() {
			return spa(directory), true
		}
	}
	return nil, false
}

func spa(directory string) http.Handler {
	files := http.FileServer(http.Dir(directory))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.NotFound(w, r)
			return
		}
		// The browser router owns non-file paths, so it must receive index.html.
		if extension := filepath.Ext(r.URL.Path); extension == "" {
			http.ServeFile(w, r, filepath.Join(directory, "index.html"))
			return
		}
		files.ServeHTTP(w, r)
	})
}
