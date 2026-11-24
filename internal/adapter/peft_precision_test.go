package adapter

import (
	"math"
	"testing"
)

func TestDecodePEFTTensorNormalizesSupportedPrecisions(t *testing.T) {
	tests := []struct {
		name  string
		dtype string
		data  []byte
	}{
		{name: "float32", dtype: "F32", data: []byte{0x00, 0x00, 0xc0, 0x3f}},
		{name: "bfloat16", dtype: "BF16", data: []byte{0xc0, 0x3f}},
		{name: "float16", dtype: "F16", data: []byte{0x00, 0x3e}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			values, err := decodePEFTTensor(test.dtype, test.data)
			if err != nil {
				t.Fatal(err)
			}
			if len(values) != 1 || math.Abs(float64(values[0]-1.5)) > 1e-6 {
				t.Fatalf("decoded %#v, want [1.5]", values)
			}
		})
	}
}

func TestDecodePEFTTensorRejectsUnsupportedPrecision(t *testing.T) {
	if _, err := decodePEFTTensor("I8", []byte{0}); err == nil {
		t.Fatal("expected unsupported dtype error")
	}
}
