// sypra-mock is an optional, standalone OpenAI-compatible test model.
// It intentionally has no dependency on the Sypra API, workspace storage, or
// model-runner manager. Operators may run and register it like any other local
// endpoint when they need a deterministic integration-test model.
package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"

	"github.com/sypra-labs/sypra-ai/pkg/mockrunner"
)

func main() {
	listen := flag.String("listen", "127.0.0.1:8082", "loopback listen address")
	flag.Parse()
	fmt.Printf("Sypra Mock listening at http://%s/v1\n", *listen)
	if err := http.ListenAndServe(*listen, mockrunner.Handler()); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
