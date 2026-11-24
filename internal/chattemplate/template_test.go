package chattemplate

import "testing"

func TestQwenTrainingResponseMaskMarksOnlyFinalAssistant(t *testing.T) {
	rendered, err := Render("qwen3", []Message{{Role: "system", Content: "rules"}, {Role: "user", Content: "hello"}, {Role: "assistant", Content: "hi"}}, true)
	if err != nil {
		t.Fatal(err)
	}
	ids, mask, err := ResponseTokenMask(rendered.Segments, func(segment Segment) ([]int, error) {
		if segment.Control != "" {
			return []int{1}, nil
		}
		return make([]int, len([]rune(segment.Text))), nil
	})
	if err != nil || len(ids) != len(mask) {
		t.Fatalf("mask construction failed: %v", err)
	}
	marked := 0
	for _, value := range mask {
		if value {
			marked++
		}
	}
	if marked != len("hi")+2 {
		t.Fatalf("expected final response plus Qwen end/newline tokens, got %d", marked)
	}
}

func TestFamiliesRenderGenerationPrompts(t *testing.T) {
	for _, family := range []string{FamilyQwen, FamilyLlama, FamilyMistral, FamilyGemma, FamilyPhi, FamilyGeneric} {
		out, err := Render(family, []Message{{Role: "user", Content: "hello"}}, false)
		if err != nil || len(out.Segments) == 0 {
			t.Fatalf("%s: %v", family, err)
		}
	}
}
