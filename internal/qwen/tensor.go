// Package qwen is Sypra's dependency-free CPU foundation for Qwen models.
// It intentionally uses only the Go standard library.
package qwen

import "fmt"

type Tensor struct {
	Shape []int
	Data  []float32
}

func New(shape ...int) Tensor {
	n := 1
	for _, d := range shape {
		if d < 1 {
			panic("invalid tensor dimension")
		}
		n *= d
	}
	return Tensor{Shape: append([]int(nil), shape...), Data: make([]float32, n)}
}
func From(shape []int, data []float32) (Tensor, error) {
	n := 1
	for _, d := range shape {
		if d < 1 {
			return Tensor{}, fmt.Errorf("invalid tensor shape")
		}
		n *= d
	}
	if len(data) != n {
		return Tensor{}, fmt.Errorf("tensor data length %d does not match shape %v", len(data), shape)
	}
	return Tensor{Shape: append([]int(nil), shape...), Data: data}, nil
}
func (t Tensor) Rows() int {
	if len(t.Shape) != 2 {
		panic("tensor is not a matrix")
	}
	return t.Shape[0]
}
func (t Tensor) Cols() int {
	if len(t.Shape) != 2 {
		panic("tensor is not a matrix")
	}
	return t.Shape[1]
}
func (t Tensor) At(row, col int) float32     { return t.Data[row*t.Cols()+col] }
func (t Tensor) Set(row, col int, v float32) { t.Data[row*t.Cols()+col] = v }
func MatMul(a, b Tensor) (Tensor, error) {
	if a.Cols() != b.Rows() {
		return Tensor{}, fmt.Errorf("matrix shape mismatch %v x %v", a.Shape, b.Shape)
	}
	out := New(a.Rows(), b.Cols())
	for i := 0; i < a.Rows(); i++ {
		for k := 0; k < a.Cols(); k++ {
			v := a.At(i, k)
			for j := 0; j < b.Cols(); j++ {
				out.Data[i*out.Cols()+j] += v * b.At(k, j)
			}
		}
	}
	return out, nil
}
func AddInPlace(dst, add Tensor) error {
	if len(dst.Data) != len(add.Data) {
		return fmt.Errorf("tensor shape mismatch")
	}
	for i := range dst.Data {
		dst.Data[i] += add.Data[i]
	}
	return nil
}
