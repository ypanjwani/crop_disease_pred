import { useState } from 'react'
import { RefreshCw, Clock, CheckCircle2, LayoutGrid, LayoutList } from 'lucide-react'
import ModelResultCard from './ModelResultCard'
import AOPCChart from './AOPCChart'

function formatClass(raw) {
  return raw?.replace(/___/g, ' — ').replace(/_/g, ' ') ?? '—'
}

export default function ResultsDashboard({ result, onReset, duration }) {
  const [layout, setLayout] = useState('stack')   // 'stack' | 'grid'

  if (!result) return null

  const recommended = result.model_results.find((r) => r.reliable)
  const primaryPrediction = recommended?.prediction ?? result.model_results[0]?.prediction

  return (
    <div className="space-y-6 animate-fade-up">
      {/* ── Summary banner ─────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-forest-700 bg-forest-950/40 p-4">
        <div className="flex items-start gap-4">
          {/* Uploaded image */}
          <img
            src={result.original_image}
            alt="Uploaded leaf"
            className="w-20 h-20 rounded-xl object-cover border border-forest-800 shrink-0"
          />

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <CheckCircle2 size={16} className="text-forest-400" />
              <span className="section-label text-forest-400">Analysis Complete</span>
              {duration && (
                <span className="tag bg-[var(--c-bg)] text-[var(--c-muted)] border border-[var(--c-border)]">
                  <Clock size={9} /> {(duration / 1000).toFixed(1)}s
                </span>
              )}
            </div>

            <p className="font-display font-bold text-lg text-[var(--c-text)] leading-tight truncate">
              {formatClass(primaryPrediction)}
            </p>
            <p className="text-xs text-[var(--c-muted)] mt-1">
              {result.model_results.length} model{result.model_results.length > 1 ? 's' : ''} analysed
              {recommended && (
                <> · EfficientNet-B0 at{' '}
                  <span className="text-forest-400 font-semibold">
                    {(recommended.confidence * 100).toFixed(1)}%
                  </span>{' '}confidence</>
              )}
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Layout toggle */}
            <button
              onClick={() => setLayout((l) => l === 'stack' ? 'grid' : 'stack')}
              className="btn-ghost p-2"
              title={layout === 'stack' ? 'Grid view' : 'Stack view'}
            >
              {layout === 'stack' ? <LayoutGrid size={15} /> : <LayoutList size={15} />}
            </button>

            <button onClick={onReset} className="btn-ghost flex items-center gap-1.5">
              <RefreshCw size={13} /> New
            </button>
          </div>
        </div>
      </div>

      {/* ── Consensus check ─────────────────────────────────────────────── */}
      {result.model_results.length > 1 && (() => {
        const preds = result.model_results.map((r) => r.prediction)
        const allAgree = preds.every((p) => p === preds[0])
        return (
          <div className={[
            'rounded-xl border px-4 py-2.5 flex items-center gap-2.5 text-xs',
            allAgree
              ? 'border-forest-800 bg-forest-950/30 text-forest-300'
              : 'border-yellow-900/50 bg-yellow-950/20 text-yellow-300',
          ].join(' ')}>
            <span className="text-base">{allAgree ? '✓' : '⚠'}</span>
            {allAgree
              ? 'All models agree on the diagnosis — high confidence prediction.'
              : 'Models disagree — consult the EfficientNet-B0 result (highest XAI fidelity).'}
          </div>
        )
      })()}

      {/* ── Model result cards ───────────────────────────────────────────── */}
      <div className={layout === 'grid' ? 'grid grid-cols-1 xl:grid-cols-2 gap-4' : 'space-y-4'}>
        {result.model_results.map((r, i) => (
          <ModelResultCard key={r.model_key} result={r} rank={i + 1} />
        ))}
      </div>

      {/* ── AOPC Reference Chart ─────────────────────────────────────────── */}
      <AOPCChart highlightModel={recommended?.model_key} />
    </div>
  )
}
