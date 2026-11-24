package qwen

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestNF4QuantizedMatrixVector(t *testing.T) {
	values := []float32{1, -1, .5, -.5}
	data := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(data[i*2:], uint16(math.Float32bits(value)>>16))
	}
	quantized := quantizeNF4(weight{data: data, shape: []int{2, 2}}, true)
	result, err := quantized.matrixVec([]float32{2, 4})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(result[0]+2)) > .1 || math.Abs(float64(result[1]+1)) > .3 {
		t.Fatalf("NF4 output = %v, expected approximately [-2 -1]", result)
	}
}
