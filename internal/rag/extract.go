package rag

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"unicode"
)

// extractFile is local-first. Text and Office XML formats work without extra
// dependencies. OCR, PDF extraction, speech-to-text, and video transcription
// use locally installed tools when present or commands configured through
// SYPRA_RAG_*_COMMAND. Unhandled binary files are preserved and indexed with
// explicit pending-extraction metadata instead of being decoded as garbage.
func extractFile(ctx context.Context, path string) (text, contentType string, metadata map[string]string, err error) {
	ext := strings.ToLower(filepath.Ext(path))
	contentType = contentTypeForExtension(ext)
	metadata = map[string]string{"path": path, "contentType": contentType}
	switch ext {
	case ".docx":
		text, err = extractDOCX(path)
	case ".pptx", ".xlsx", ".odt", ".ods", ".odp":
		text, err = extractOfficeArchive(path)
	case ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xml", ".yaml", ".yml", ".log", ".rtf":
		var data []byte
		data, err = os.ReadFile(path)
		text = string(data)
	case ".pdf":
		text, err = runExtractor(ctx, "SYPRA_RAG_PDF_COMMAND", []string{"pdftotext", "{path}", "-"}, path)
		metadata["extractor"] = "pdf"
	case ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic":
		text, err = runExtractor(ctx, "SYPRA_RAG_OCR_COMMAND", []string{"tesseract", "{path}", "stdout"}, path)
		metadata["extractor"] = "ocr"
	case ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v":
		text, err = runExtractor(ctx, "SYPRA_RAG_TRANSCRIBE_COMMAND", nil, path)
		metadata["extractor"] = "transcription"
	default:
		text, err = runExtractor(ctx, "SYPRA_RAG_EXTRACT_COMMAND", nil, path)
		metadata["extractor"] = "custom"
	}
	if err != nil {
		metadata["extractionStatus"] = "pending"
		metadata["extractionError"] = err.Error()
		return fmt.Sprintf("File %s is stored locally as %s. Text extraction is pending: %s", filepath.Base(path), contentType, err), contentType, metadata, nil
	}
	metadata["extractionStatus"] = "complete"
	return strings.TrimSpace(text), contentType, metadata, nil
}

func runExtractor(ctx context.Context, env string, fallback []string, path string) (string, error) {
	arguments := fallback
	if configured := strings.TrimSpace(os.Getenv(env)); configured != "" {
		arguments = strings.Fields(configured)
	}
	if len(arguments) == 0 {
		return "", fmt.Errorf("no local extractor configured; set %s to a command that writes extracted text to stdout", env)
	}
	for index, argument := range arguments {
		arguments[index] = strings.ReplaceAll(argument, "{path}", path)
	}
	if _, err := exec.LookPath(arguments[0]); err != nil {
		return "", fmt.Errorf("extractor %q is unavailable; install it or set %s", arguments[0], env)
	}
	output, err := exec.CommandContext(ctx, arguments[0], arguments[1:]...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("extractor failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	if strings.TrimSpace(string(output)) == "" {
		return "", fmt.Errorf("extractor returned no text")
	}
	return string(output), nil
}

func extractOfficeArchive(path string) (string, error) {
	archive, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer archive.Close()
	var out strings.Builder
	for _, file := range archive.File {
		name := strings.ToLower(file.Name)
		if !strings.HasSuffix(name, ".xml") || !(strings.Contains(name, "document") || strings.Contains(name, "sharedstrings") || strings.Contains(name, "slide")) {
			continue
		}
		reader, err := file.Open()
		if err != nil {
			return "", err
		}
		data, readErr := io.ReadAll(reader)
		_ = reader.Close()
		if readErr != nil {
			continue
		}
		if text := stripXML(string(data)); text != "" {
			if out.Len() > 0 {
				out.WriteByte('\n')
			}
			out.WriteString(text)
		}
	}
	if out.Len() == 0 {
		return "", fmt.Errorf("Office archive has no readable text XML")
	}
	return out.String(), nil
}

func stripXML(value string) string {
	var out strings.Builder
	inTag := false
	for _, r := range value {
		switch r {
		case '<':
			inTag = true
		case '>':
			inTag = false
			if out.Len() > 0 {
				out.WriteByte(' ')
			}
		default:
			if !inTag && (unicode.IsPrint(r) || unicode.IsSpace(r)) {
				out.WriteRune(r)
			}
		}
	}
	return strings.Join(strings.Fields(out.String()), " ")
}

func contentTypeForExtension(ext string) string {
	switch ext {
	case ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic":
		return "image"
	case ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac":
		return "audio"
	case ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v":
		return "video"
	case ".pdf":
		return "pdf"
	case ".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp":
		return "office"
	case ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xml", ".yaml", ".yml", ".log", ".rtf":
		return "text"
	default:
		return "binary"
	}
}
