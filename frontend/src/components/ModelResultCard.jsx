import { useState } from 'react'
import { Star, Clock, ChevronDown, ChevronUp } from 'lucide-react'
import ConfidenceBar from './ConfidenceBar'
import XAIPanel from './XAIPanel'

function formatClass(raw) {
  return raw?.replace(/___/g, ' — ').replace(/_/g, ' ') ?? '—'
}

function formatMs(ms) {
  if (!ms) return '—'
  return ms > 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export default function ModelResultCard({ result, rank }) {
  const [expanded, setExpanded] = useState(true)

  const isRecommended = result.reliable

  return (
    <div className={[
      'rounded-2xl border overflow-hidden transition-all duration-300',
      isRecommended
        ? 'border-forest-700 shadow-[0_0_24px_rgba(34,197,94,0.08)]'
        : 'border-[var(--c-border)]',
    ].join(' ')}>
      {/* ── Card header ─────────────────────────────────────────────────── */}
      <div
        className={[
          'px-4 py-3 flex items-center justify-between cursor-pointer',
          'transition-colors duration-200',
          isRecommended
            ? 'bg-forest-950/60 hover:bg-forest-950/80'
            : 'bg-[var(--c-surface)] hover:bg-[var(--c-border)]/20',
        ].join(' ')}
        onClick={() => setExpanded((e) => !e)}
      >
        <div className="flex items-center gap-3">
          {/* Rank badge */}
          <div className={[
            'w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold',
            isRecommended
              ? 'bg-forest-900 text-forest-300 border border-forest-700'
              : 'bg-[var(--c-bg)] text-[var(--c-muted)] border border-[var(--c-border)]',
          ].join(' ')}>
            {rank}
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="font-display font-semibold text-sm text-[var(--c-text)]">
                {result.model_name}
              </span>
              {isRecommended && (
                <span className="tag bg-forest-900/60 text-forest-300 border border-forest-700">
                  <Star size={9} className="fill-current" /> Best XAI
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="font-mono text-xs text-[var(--c-muted)] truncate max-w-[200px]">
                {formatClass(result.prediction)}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Confidence pill */}
          <div className={[
            'font-mono text-sm font-bold px-2.5 py-1 rounded-lg',
            result.confidence >= 0.85
              ? 'bg-forest-950 text-forest-300 border border-forest-800'
              : result.confidence >= 0.6
              ? 'bg-yellow-950 text-yellow-300 border border-yellow-800'
              : 'bg-red-950 text-red-300 border border-red-800',
          ].join(' ')}>
            {(result.confidence * 100).toFixed(1)}%
          </div>

          {/* Timing */}
          <div className="hidden sm:flex items-center gap-1 text-[var(--c-muted)]">
            <Clock size={11} />
            <span className="font-mono text-xs">{formatMs(result.inference_ms)}</span>
          </div>

          {expanded ? (
            <ChevronUp size={16} className="text-[var(--c-muted)]" />
          ) : (
            <ChevronDown size={16} className="text-[var(--c-muted)]" />
          )}
        </div>
      </div>

      {/* ── Expanded body ────────────────────────────────────────────────── */}
      {expanded && (
        <div className="p-4 space-y-5 border-t border-[var(--c-border)]">
          {/* Confidence bar */}
          <ConfidenceBar value={result.confidence} size="lg" showPercent={false} />

          {/* Top-5 predictions */}
          {result.top5?.length > 0 && (
            <div className="space-y-2">
              <span className="section-label">Top-5 Predictions</span>
              <div className="space-y-1.5">
                {result.top5.map((item, i) => (
                  <ConfidenceBar
                    key={i}
                    value={item.confidence}
                    label={formatClass(item.class)}
                    size="sm"
                  />
                ))}
              </div>
            </div>
          )}

          {/* XAI Panel */}
          <XAIPanel xai={result.xai} loading={false} />

          {/* Reliability note for recommended model */}
          {isRecommended && (
            <div className="rounded-xl border border-forest-800 bg-forest-950/30 p-3">
              <p className="text-xs text-forest-300 leading-relaxed">
                <span className="font-semibold font-mono text-forest-400">★ Most Reliable</span>
                {' '}— EfficientNet-B0 achieves positive AOPC scores across all three XAI methods
                (IG: +0.136, Grad-CAM: +0.054, LIME: +0.087), confirming it focuses on
                actual disease lesions rather than background noise.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
