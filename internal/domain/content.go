package domain

import (
	"fmt"
	"strings"
)

// ContentPart preserves a structured chat or document payload. Text remains
// available through ChatMessage.Content for backward compatibility, while
// Parts carries content that a multimodal model can consume or return.
// URL and Data intentionally support both remote references and local data
// URLs; callers decide which form their selected model endpoint accepts.
type ContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	URL      string `json:"url,omitempty"`
	Data     string `json:"data,omitempty"`
	Name     string `json:"name,omitempty"`
	MIMEType string `json:"mimeType,omitempty"`
}

func (p ContentPart) Validate() error {
	switch strings.ToLower(strings.TrimSpace(p.Type)) {
	case "text", "markdown", "diagram":
		if strings.TrimSpace(p.Text) == "" {
			return fmt.Errorf("text content cannot be empty")
		}
	case "image_url", "video_url", "audio", "video", "image", "artifact":
		if strings.TrimSpace(p.URL) == "" && strings.TrimSpace(p.Data) == "" {
			return fmt.Errorf("%s content requires a URL or data", p.Type)
		}
	case "input_audio", "file":
		if strings.TrimSpace(p.URL) == "" && strings.TrimSpace(p.Data) == "" {
			return fmt.Errorf("%s content requires a URL or data", p.Type)
		}
	default:
		return fmt.Errorf("unsupported content type %q", p.Type)
	}
	return nil
}

func ValidateContentParts(parts []ContentPart) error {
	for _, part := range parts {
		if err := part.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// ContentText creates a safe textual projection for titles, retrieval queries,
// legacy text-only models, and chat history previews. Binary payloads are not
// copied into that projection.
func ContentText(text string, parts []ContentPart) string {
	values := make([]string, 0, len(parts)+1)
	if trimmed := strings.TrimSpace(text); trimmed != "" {
		values = append(values, trimmed)
	}
	for _, part := range parts {
		switch strings.ToLower(part.Type) {
		case "text":
			if trimmed := strings.TrimSpace(part.Text); trimmed != "" {
				values = append(values, trimmed)
			}
		case "image_url", "image":
			values = append(values, "[image]")
		case "video_url", "video":
			values = append(values, "[video]")
		case "input_audio", "audio":
			values = append(values, "[audio]")
		case "file":
			if part.Name != "" {
				values = append(values, "[file: "+part.Name+"]")
			} else {
				values = append(values, "[file]")
			}
		case "artifact":
			values = append(values, "[artifact: "+part.Name+"]")
		case "diagram":
			values = append(values, "[diagram]")
		case "markdown":
			values = append(values, part.Text)
		}
	}
	return strings.Join(values, "\n")
}

func HasNonTextContent(parts []ContentPart) bool {
	for _, part := range parts {
		if strings.ToLower(part.Type) != "text" {
			return true
		}
	}
	return false
}
