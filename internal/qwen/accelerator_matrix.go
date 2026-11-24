package qwen

import "fmt"

// acceleratorMatrixValues materializes an immutable projection as float32 for
// accelerator backends. QLoRA's NF4 values are dequantized only for the cached
// device copy; its source weights remain quantized in the model.
func acceleratorMatrixValues(model *Model, name string) (rows, cols int, values []float32, err error) {
	if model.useNF4 {
		quantized, quantizeErr := model.nf4(name)
		if quantizeErr != nil {
			return 0, 0, nil, quantizeErr
		}
		if len(quantized.shape) != 2 {
			return 0, 0, nil, fmt.Errorf("accelerator target %s is not a matrix", name)
		}
		rows, cols = quantized.shape[0], quantized.shape[1]
		values = make([]float32, rows*cols)
		for index := range values {
			values[index] = quantized.value(index)
		}
		return rows, cols, values, nil
	}
	weight, tensorErr := model.tensor(name)
	if tensorErr != nil {
		return 0, 0, nil, tensorErr
	}
	if len(weight.shape) != 2 {
		return 0, 0, nil, fmt.Errorf("accelerator target %s is not a matrix", name)
	}
	rows, cols = weight.shape[0], weight.shape[1]
	values = make([]float32, rows*cols)
	for index := range values {
		values[index] = bfloat(weight.data, index)
	}
	return rows, cols, values, nil
}
