package rag

import (
	"archive/zip"
	"encoding/xml"
	"fmt"
	"io"
	"strings"
)

// extractDOCX extracts visible paragraph/table text from Word's document XML.
// It handles ordinary DOCX files without requiring Office, Python, or a native
// document conversion dependency.
func extractDOCX(path string) (string, error) {
	archive, err := zip.OpenReader(path)
	if err != nil {
		return "", err
	}
	defer archive.Close()
	for _, file := range archive.File {
		if file.Name != "word/document.xml" {
			continue
		}
		reader, err := file.Open()
		if err != nil {
			return "", err
		}
		defer reader.Close()
		decoder := xml.NewDecoder(reader)
		var out strings.Builder
		for {
			token, err := decoder.Token()
			if err != nil {
				if err == io.EOF {
					break
				}
				return "", err
			}
			switch value := token.(type) {
			case xml.CharData:
				out.Write([]byte(value))
			case xml.EndElement:
				switch value.Name.Local {
				case "p", "tr":
					out.WriteByte('\n')
				case "tc":
					out.WriteString("\t")
				}
			case xml.StartElement:
				if value.Name.Local == "tab" || value.Name.Local == "br" || value.Name.Local == "cr" {
					out.WriteByte('\n')
				}
			}
		}
		text := strings.TrimSpace(out.String())
		if text == "" {
			return "", fmt.Errorf("DOCX contains no extractable text")
		}
		return text, nil
	}
	return "", fmt.Errorf("invalid DOCX: word/document.xml is missing")
}
