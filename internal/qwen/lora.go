package qwen

import (
	"fmt"
	"math"
)

// LoRA is the trainable low-rank delta W + scale*(B*A). Base is frozen.
type LoRA struct {
	Name       string
	Base, A, B Tensor
	Alpha      float32
}

func NewLoRA(name string, base Tensor, rank int, alpha float32) (*LoRA, error) {
	if len(base.Shape) != 2 || rank < 1 || rank > base.Cols() {
		return nil, fmt.Errorf("invalid LoRA dimensions")
	}
	a := New(rank, base.Cols())
	b := New(base.Rows(), rank)
	for i := range a.Data {
		a.Data[i] = float32(math.Sin(float64(i+1)) * .01)
	}
	return &LoRA{Name: name, Base: base, A: a, B: b, Alpha: alpha}, nil
}
func (l *LoRA) Forward(input Tensor) (Tensor, error) {
	baseT := transpose(l.Base)
	out, e := MatMul(input, baseT)
	if e != nil {
		return Tensor{}, e
	}
	delta, e := MatMul(l.B, l.A)
	if e != nil {
		return Tensor{}, e
	}
	deltaT := transpose(delta)
	adapt, e := MatMul(input, deltaT)
	if e != nil {
		return Tensor{}, e
	}
	scale := l.Alpha / float32(l.A.Rows())
	for i := range adapt.Data {
		adapt.Data[i] *= scale
	}
	return out, AddInPlace(out, adapt)
}
func transpose(t Tensor) Tensor {
	out := New(t.Cols(), t.Rows())
	for i := 0; i < t.Rows(); i++ {
		for j := 0; j < t.Cols(); j++ {
			out.Set(j, i, t.At(i, j))
		}
	}
	return out
}
