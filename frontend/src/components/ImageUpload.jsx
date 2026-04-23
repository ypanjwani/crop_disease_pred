import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, X, Leaf } from 'lucide-react'

const ACCEPTED = { 'image/jpeg': [], 'image/png': [], 'image/webp': [] }
const MAX_SIZE = 10 * 1024 * 1024

export default function ImageUpload({ onImageSelected, disabled }) {
  const [preview, setPreview] = useState(null)
  const [fileName, setFileName] = useState(null)
  const [fileSize, setFileSize] = useState(null)

  const onDrop = useCallback((accepted, rejected) => {
    if (rejected.length > 0) return
    const file = accepted[0]
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreview(url)
    setFileName(file.name)
    setFileSize((file.size / 1024).toFixed(1) + ' KB')
    onImageSelected(file)
  }, [onImageSelected])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    maxSize: MAX_SIZE,
    disabled,
  })

  const clear = (e) => {
    e.stopPropagation()
    setPreview(null)
    setFileName(null)
    setFileSize(null)
    onImageSelected(null)
  }

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={[
          'relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden',
          isDragActive && !isDragReject
            ? 'border-forest-400 bg-forest-950/40 scale-[1.01]'
            : isDragReject
            ? 'border-red-500 bg-red-950/20'
            : preview
            ? 'border-forest-700 bg-[var(--c-surface)]'
            : 'border-[var(--c-border)] bg-[var(--c-surface)] hover:border-forest-600',
          disabled && 'opacity-50 pointer-events-none',
        ].join(' ')}
        style={{ minHeight: preview ? '320px' : '240px' }}
      >
        <input {...getInputProps()} />

        {preview ? (
          /* ── Preview state ──────────────────────────────────────── */
          <div className="relative w-full h-full" style={{ minHeight: '320px' }}>
            <img
              src={preview}
              alt="Selected leaf"
              className="w-full h-full object-contain"
              style={{ maxHeight: '400px' }}
            />
            {/* Overlay gradient */}
            <div className="absolute inset-0 bg-gradient-to-t from-[var(--c-bg)]/80 via-transparent to-transparent pointer-events-none" />

            {/* File info strip */}
            <div className="absolute bottom-0 left-0 right-0 p-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ImageIcon size={14} className="text-forest-400" />
                <span className="font-mono text-xs text-forest-300 truncate max-w-[200px]">{fileName}</span>
                <span className="tag bg-forest-950 text-forest-400 border border-forest-800">{fileSize}</span>
              </div>
              <button
                onClick={clear}
                className="p-1.5 rounded-lg bg-[var(--c-bg)]/80 border border-[var(--c-border)] 
                           text-[var(--c-muted)] hover:text-red-400 hover:border-red-800 transition-colors"
                title="Remove image"
              >
                <X size={14} />
              </button>
            </div>

            {/* Scan animation overlay */}
            {isDragActive && (
              <div className="absolute inset-0 pointer-events-none overflow-hidden">
                <div className="absolute left-0 right-0 h-0.5 bg-forest-400/60 animate-scan" />
              </div>
            )}
          </div>
        ) : (
          /* ── Empty state ────────────────────────────────────────── */
          <div className="flex flex-col items-center justify-center gap-5 py-14 px-6 text-center">
            {/* Icon cluster */}
            <div className="relative">
              <div className="w-20 h-20 rounded-2xl bg-forest-950 border border-forest-800 
                              flex items-center justify-center">
                <Leaf
                  size={36}
                  className={[
                    'transition-colors duration-300',
                    isDragActive ? 'text-forest-400' : 'text-forest-600',
                  ].join(' ')}
                />
              </div>
              <div className="absolute -top-2 -right-2 w-8 h-8 rounded-full 
                              bg-[var(--c-bg)] border border-[var(--c-border)]
                              flex items-center justify-center">
                <Upload size={14} className="text-[var(--c-muted)]" />
              </div>
            </div>

            <div>
              <p className="font-display font-semibold text-base text-[var(--c-text)] mb-1">
                {isDragActive ? 'Drop leaf image here' : 'Upload a plant leaf image'}
              </p>
              <p className="text-sm text-[var(--c-muted)]">
                Drag & drop or{' '}
                <span className="text-forest-400 underline underline-offset-2 cursor-pointer">
                  browse files
                </span>
              </p>
            </div>

            <div className="flex items-center gap-3 flex-wrap justify-center">
              {['JPEG', 'PNG', 'WebP'].map((fmt) => (
                <span key={fmt} className="tag bg-[var(--c-bg)] text-[var(--c-muted)] border border-[var(--c-border)]">
                  {fmt}
                </span>
              ))}
              <span className="tag bg-[var(--c-bg)] text-[var(--c-muted)] border border-[var(--c-border)]">
                max 10 MB
              </span>
            </div>
          </div>
        )}
      </div>

      {isDragReject && (
        <p className="mt-2 text-xs text-red-400 text-center font-mono">
          ✗ Invalid file type — only JPEG, PNG, WebP accepted
        </p>
      )}
    </div>
  )
}
