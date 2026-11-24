package qwen

import (
	"fmt"
	"math"
)

// NF4 uses the normal-float-4 codebook used by QLoRA-style 4-bit weight
// quantization. We quantize frozen matrix blocks; LoRA matrices stay float32.
var nf4Codebook = [...]float32{
	-1.0, -0.6961928, -0.52507305, -0.3949175,
	-0.28444138, -0.18477343, -0.09105004, 0,
	0.0795803, 0.1609302, 0.2461123, 0.33791524,
	0.44070983, 0.562617, 0.72295684, 1.0,
}

const nf4BlockSize = 64

type nf4Weight struct {
	shape       []int
	values      []byte
	scales      []float32
	scaleCodes  []uint8
	scaleAbsMax float32
	doubleQuant bool
}

func quantizeNF4(w weight, doubleQuant bool) *nf4Weight {
	count := len(w.data) / 2
	blocks := (count + nf4BlockSize - 1) / nf4BlockSize
	result := &nf4Weight{
		shape:       append([]int(nil), w.shape...),
		values:      make([]byte, (count+1)/2),
		scales:      make([]float32, blocks),
		doubleQuant: doubleQuant,
	}
	for block := 0; block < blocks; block++ {
		start, end := block*nf4BlockSize, minInt((block+1)*nf4BlockSize, count)
		var maxAbs float32
		for i := start; i < end; i++ {
			value := float32(math.Abs(float64(bfloat(w.data, i))))
			if value > maxAbs {
				maxAbs = value
			}
		}
		if maxAbs == 0 {
			maxAbs = 1
		}
		result.scales[block] = maxAbs
		for i := start; i < end; i++ {
			value := bfloat(w.data, i) / maxAbs
			code := nearestNF4(value)
			if i%2 == 0 {
				result.values[i/2] = code
			} else {
				result.values[i/2] |= code << 4
			}
		}
	}
	if doubleQuant {
		for _, scale := range result.scales {
			if scale > result.scaleAbsMax {
				result.scaleAbsMax = scale
			}
		}
		result.scaleCodes = make([]uint8, len(result.scales))
		if result.scaleAbsMax > 0 {
			for i, scale := range result.scales {
				result.scaleCodes[i] = uint8(math.Round(float64(scale / result.scaleAbsMax * 255)))
			}
		}
		result.scales = nil
	}
	return result
}

func nearestNF4(value float32) byte {
	best := 0
	distance := float32(math.MaxFloat32)
	for i, candidate := range nf4Codebook {
		current := float32(math.Abs(float64(value - candidate)))
		if current < distance {
			best, distance = i, current
		}
	}
	return byte(best)
}

func (w *nf4Weight) value(index int) float32 {
	code := w.values[index/2]
	if index%2 == 0 {
		code &= 0x0f
	} else {
		code >>= 4
	}
	block := index / nf4BlockSize
	scale := float32(0)
	if w.doubleQuant {
		scale = float32(w.scaleCodes[block]) / 255 * w.scaleAbsMax
	} else {
		scale = w.scales[block]
	}
	return nf4Codebook[code] * scale
}

func (w *nf4Weight) matrixVec(input []float32) ([]float32, error) {
	if len(w.shape) != 2 || w.shape[1] != len(input) {
		return nil, fmt.Errorf("quantized matrix shape %v cannot multiply %d values", w.shape, len(input))
	}
	result := make([]float32, w.shape[0])
	parallelRows(w.shape[0], func(row int) {
		start := row * w.shape[1]
		for column, value := range input {
			result[row] += w.value(start+column) * value
		}
	})
	return result, nil
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}
