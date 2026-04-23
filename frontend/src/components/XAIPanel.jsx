import { useState } from 'react'
import { Eye, HelpCircle } from 'lucide-react'

const XAI_META = [
  {
    key:   'grad_cam',
    label: 'Grad-CAM',
    color: 'text-orange-400',
    border: 'border-orange-900/50',
    bg:    'bg-orange-950/20',
    desc:  'Gradient-weighted Class Activation Map. Highlights the spatial regions of the image that most influence the model\'s decision by computing gradients of the target class score w.r.t. the final convolutional layer.',
  },
  {
    key:   'lime',
    label: 'LIME',
    color: 'text-emerald-400',
    border: 'border-emerald-900/50',
    bg:    'bg-emerald-950/20',
    desc:  'Local Interpretable Model-agnostic Explanations. Perturbs superpixels and fits a local linear surrogate model to identify which image segments contribute positively to the predicted class.',
  },
  {
    key:   'ig',
    label: 'Integrated Gradients',
    color: 'text-violet-400',
    border: 'border-violet-900/50',
    bg:    'bg-violet-950/20',
    desc:  'Axiomatic attribution method that integrates gradients along a straight-line path from a black baseline to the input. Satisfies completeness and sensitivity axioms.',
  },
]

function Tooltip({ text, children }) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative inline-block"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-2.5
                        rounded-lg bg-[#0d1a10] border border-[var(--c-border)]
                        text-[10px] text-[var(--c-muted)] leading-relaxed shadow-xl">
          {text}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-[var(--c-border)]" />
        </div>
      )}
    </div>
  )
}

function XAICard({ meta, src, loading }) {
  const [zoomed, setZoomed] = useState(false)

  return (
    <>
      <div className={`rounded-xl border ${meta.border} ${meta.bg} overflow-hidden flex flex-col`}>
        {/* Header */}
        <div className="px-3 py-2 flex items-center justify-between border-b border-[var(--c-border)]">
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${meta.bg} border ${meta.border}`}
              style={{ backgroundColor: 'currentColor', color: meta.color.replace('text-', '') }}
            />
            <span className={`font-mono text-xs font-medium ${meta.color}`}>{meta.label}</span>
          </div>
          <div className="flex items-center gap-1">
            <Tooltip text={meta.desc}>
              <HelpCircle size={12} className="text-[var(--c-muted)] cursor-help" />
            </Tooltip>
            {src && (
              <button onClick={() => setZoomed(true)}
                className="p-1 rounded hover:bg-[var(--c-border)] transition-colors"
                title="Zoom">
                <Eye size={12} className="text-[var(--c-muted)]" />
              </button>
            )}
          </div>
        </div>

        {/* Image */}
        <div className="flex-1 flex items-center justify-center p-2 min-h-[160px]">
          {loading ? (
            <div className="w-full h-36 rounded-lg bg-[var(--c-bg)] border border-[var(--c-border)] animate-pulse" />
          ) : src ? (
            <img
              src={src}
              alt={meta.label}
              className="w-full rounded-lg object-contain cursor-zoom-in max-h-[200px]"
              onClick={() => setZoomed(true)}
            />
          ) : (
            <div className="text-center text-[var(--c-muted)]">
              <p className="font-mono text-xs">No data</p>
            </div>
          )}
        </div>
      </div>

      {/* Zoom lightbox */}
      {zoomed && (
        <div className="fixed inset-0 z-[100] bg-black/90 flex items-center justify-center p-6"
          onClick={() => setZoomed(false)}>
          <div className="relative max-w-3xl w-full" onClick={(e) => e.stopPropagation()}>
            <div className={`rounded-2xl border-2 ${meta.border} overflow-hidden`}>
              <div className={`px-4 py-3 ${meta.bg} flex items-center justify-between`}>
                <span className={`font-display font-semibold ${meta.color}`}>{meta.label}</span>
                <button onClick={() => setZoomed(false)}
                  className="text-[var(--c-muted)] hover:text-[var(--c-text)] transition-colors font-mono text-sm">
                  ✕ close
                </button>
              </div>
              <img src={src} alt={meta.label} className="w-full object-contain" />
            </div>
            <p className="mt-3 text-xs text-[var(--c-muted)] font-mono text-center px-4">{meta.desc}</p>
          </div>
        </div>
      )}
    </>
  )
}

export default function XAIPanel({ xai, loading }) {
  return (
    <div className="space-y-2">
      <span className="section-label">XAI Explanations</span>
      <div className="grid grid-cols-3 gap-2">
        {XAI_META.map((meta) => (
          <XAICard
            key={meta.key}
            meta={meta}
            src={xai?.[meta.key]}
            loading={loading}
          />
        ))}
      </div>
    </div>
  )
}
