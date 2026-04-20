import { useRef, useState } from 'react';

interface Props {
  mode: 'parlay' | 'results';
  onFile: (file: File) => void;
  disabled: boolean;
}

export function FileUploadZone({ mode, onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  function handle(file: File) {
    setFileName(file.name);
    onFile(file);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handle(file);
  }

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handle(file);
  }

  const label = mode === 'parlay' ? 'Upload Parlay (TXT / PDF / Image)' : 'Upload Result Screenshot';

  return (
    <div
      className={`pe-upload-zone ${dragging ? 'pe-upload-zone--dragging' : ''} ${disabled ? 'pe-upload-zone--disabled' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.pdf,.txt"
        style={{ display: 'none' }}
        onChange={onChange}
        disabled={disabled}
      />
      <span className="pe-upload-zone__icon">📂</span>
      <span className="pe-upload-zone__label">{label}</span>
      {fileName
        ? <span className="pe-upload-zone__file">{fileName}</span>
        : <span className="pe-upload-zone__hint">Drag & drop or click to browse</span>}
      {disabled && <span className="pe-upload-zone__spinner">Processing…</span>}
    </div>
  );
}
