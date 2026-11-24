package qwen

import "testing"

func TestMatMul(t *testing.T) {
	a, _ := From([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := From([]int{2, 1}, []float32{5, 6})
	got, e := MatMul(a, b)
	if e != nil {
		t.Fatal(e)
	}
	if got.Data[0] != 17 || got.Data[1] != 39 {
		t.Fatalf("unexpected product %v", got.Data)
	}
}
func TestLoRAForward(t *testing.T) {
	base, _ := From([]int{1, 2}, []float32{2, 3})
	layer, e := NewLoRA("q_proj", base, 1, 2)
	if e != nil {
		t.Fatal(e)
	}
	input, _ := From([]int{1, 2}, []float32{1, 1})
	out, e := layer.Forward(input)
	if e != nil {
		t.Fatal(e)
	}
	if len(out.Data) != 1 {
		t.Fatalf("unexpected output %v", out)
	}
}
