package agent

import (
	"context"
	"fmt"
)

func (s *Service) validateArguments(ctx context.Context, toolID string, arguments map[string]any) error {
	tools, err := s.Catalog(ctx)
	if err != nil {
		return err
	}
	for _, tool := range tools {
		if tool.ID == toolID {
			return validateSchema(tool.InputSchema, arguments, "arguments")
		}
	}
	return fmt.Errorf("tool %s is not in the enabled catalog", toolID)
}

func validateSchema(schema map[string]any, value any, path string) error {
	typeName, _ := schema["type"].(string)
	switch typeName {
	case "object":
		object, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("%s must be an object", path)
		}
		properties, _ := schema["properties"].(map[string]any)
		for _, required := range stringList(schema["required"]) {
			if _, ok := object[required]; !ok {
				return fmt.Errorf("%s.%s is required", path, required)
			}
		}
		additional, hasAdditional := schema["additionalProperties"].(bool)
		for key, child := range object {
			raw, known := properties[key]
			if !known && hasAdditional && !additional {
				return fmt.Errorf("%s.%s is not permitted", path, key)
			}
			if childSchema, ok := raw.(map[string]any); ok {
				if err := validateSchema(childSchema, child, path+"."+key); err != nil {
					return err
				}
			}
		}
	case "string":
		text, ok := value.(string)
		if !ok {
			return fmt.Errorf("%s must be a string", path)
		}
		min := 0
		switch value := schema["minLength"].(type) {
		case int:
			min = value
		case float64: // JSON-decoded MCP schemas use float64 for numbers.
			min = int(value)
		}
		if min > 0 && len([]rune(text)) < min {
			return fmt.Errorf("%s is too short", path)
		}
	case "array":
		items, ok := value.([]any)
		if !ok {
			return fmt.Errorf("%s must be an array", path)
		}
		if itemSchema, ok := schema["items"].(map[string]any); ok {
			for index, item := range items {
				if err := validateSchema(itemSchema, item, fmt.Sprintf("%s[%d]", path, index)); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
func stringList(value any) []string {
	switch values := value.(type) {
	case []string:
		return values
	case []any:
		result := make([]string, 0, len(values))
		for _, value := range values {
			if text, ok := value.(string); ok {
				result = append(result, text)
			}
		}
		return result
	}
	return nil
}
