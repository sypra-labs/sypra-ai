package finetune

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadDatasetAcceptsChatJSONLAndKeepsDuplicateGroupsTogether(t *testing.T) {
	path := filepath.Join(t.TempDir(), "training.jsonl")
	data := "{\"messages\":[{\"role\":\"system\",\"content\":\"Be concise\"},{\"role\":\"user\",\"content\":\"Where is my order?\"},{\"role\":\"assistant\",\"content\":\"It is in transit.\"}]}\n" +
		"{\"prompt\":\"Reset password\",\"completion\":\"Use the reset link.\"}\n" +
		"{\"instruction\":\"Cancel order\",\"response\":\"I can help cancel it.\"}\n"
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	examples, err := ReadDataset(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(examples) != 3 || examples[0].Response != "It is in transit." || examples[0].Instruction == "" {
		t.Fatalf("unexpected examples: %#v", examples)
	}
	train, validation, test := SplitDataset3(examples, 42)
	if len(train)+len(validation)+len(test) != len(examples) || len(train) == 0 || len(validation) == 0 || len(test) == 0 {
		t.Fatalf("invalid split: train=%d validation=%d test=%d", len(train), len(validation), len(test))
	}
}

func TestReadDatasetRequiresInstructionAndResponse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "training.csv")
	if err := os.WriteFile(path, []byte("instruction,category\nhello,support\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadDataset(path); err == nil {
		t.Fatal("expected missing response column to fail")
	}
}
