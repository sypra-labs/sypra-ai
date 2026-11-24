package qwen

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type safeEntry struct {
	DType   string  `json:"dtype"`
	Shape   []int   `json:"shape"`
	Offsets []int64 `json:"data_offsets"`
}

// SafeTensorFile is a read-only, lazy view over a local SafeTensors file.
// Checkpoint payloads are materialized only when a model operation needs them.
type SafeTensorFile struct {
	file      *os.File
	dataStart int64
	entries   map[string]safeEntry
}

func OpenSafeTensorFile(path string) (*SafeTensorFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var headerSize uint64
	if err = binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		_ = f.Close()
		return nil, err
	}
	if headerSize > 64<<20 {
		_ = f.Close()
		return nil, fmt.Errorf("invalid safetensors header")
	}
	header := make([]byte, headerSize)
	if _, err = io.ReadFull(f, header); err != nil {
		_ = f.Close()
		return nil, err
	}
	var entries map[string]safeEntry
	if err = json.Unmarshal(header, &entries); err != nil {
		_ = f.Close()
		return nil, err
	}
	return &SafeTensorFile{file: f, dataStart: int64(8 + headerSize), entries: entries}, nil
}

func (s *SafeTensorFile) Close() error { return s.file.Close() }

func (s *SafeTensorFile) Entry(name string) (dtype string, shape []int, bytes int64, err error) {
	entry, ok := s.entries[name]
	if !ok || len(entry.Offsets) != 2 {
		return "", nil, 0, fmt.Errorf("tensor %q not found", name)
	}
	return entry.DType, append([]int(nil), entry.Shape...), entry.Offsets[1] - entry.Offsets[0], nil
}

func (s *SafeTensorFile) ReadBytes(name string) ([]byte, []int, error) {
	entry, ok := s.entries[name]
	if !ok || len(entry.Offsets) != 2 || entry.Offsets[1] < entry.Offsets[0] {
		return nil, nil, fmt.Errorf("tensor %q not found", name)
	}
	data := make([]byte, entry.Offsets[1]-entry.Offsets[0])
	if _, err := s.file.ReadAt(data, s.dataStart+entry.Offsets[0]); err != nil {
		return nil, nil, err
	}
	return data, append([]int(nil), entry.Shape...), nil
}

// ReadSafeTensorF32 reads an F32 tensor. Qwen checkpoints use BF16 and are
// handled directly by the local runner's BF16 matrix operations.
func ReadSafeTensorF32(path, name string) (Tensor, error) {
	s, err := OpenSafeTensorFile(path)
	if err != nil {
		return Tensor{}, err
	}
	defer s.Close()
	dtype, _, _, err := s.Entry(name)
	if err != nil {
		return Tensor{}, err
	}
	if dtype != "F32" {
		return Tensor{}, fmt.Errorf("only F32 safetensors are supported for %q", name)
	}
	raw, shape, err := s.ReadBytes(name)
	if err != nil {
		return Tensor{}, err
	}
	data := make([]float32, len(raw)/4)
	if err = binary.Read(bytes.NewReader(raw), binary.LittleEndian, &data); err != nil {
		return Tensor{}, err
	}
	return From(shape, data)
}
