const AOPC_DATA = {
  resnet18:        { ig: -0.107, gradcam: -0.157, lime: -0.013 },
  efficientnet_b0: { ig:  0.136, gradcam:  0.054, lime:  0.087 },
  densenet121:     { ig:  0.067, gradcam: -0.197, lime: -0.128 },
}

const XAI_COLORS = {
  ig:      { bar: '#7c3aed', label: 'Integrated Gradients', bg: 'bg-violet-900/30', text: 'text-violet-400' },
  gradcam: { bar: '#ea580c', label: 'Grad-CAM',             bg: 'bg-orange-900/30', text: 'text-orange-400' },
  lime:    { bar: '#059669', label: 'LIME',                  bg: 'bg-emerald-900/30', text: 'text-emerald-400' },
}

const MODEL_LABELS = {
  resnet18:        'ResNet-18',
  efficientnet_b0: 'EfficientNet-B0',
  densenet121:     'DenseNet-121',
}

const MIN_VAL = -0.22
const MAX_VAL =  0.16
const RANGE   = MAX_VAL - MIN_VAL

function pct(value) {
  return ((value - MIN_VAL) / RANGE) * 100
}

const ZERO_PCT = ((0 - MIN_VAL) / RANGE) * 100

function BarRow({ model, xai, value }) {
  const meta   = XAI_COLORS[xai]
  const isPos  = value >= 0
  const barPct = Math.abs((value / RANGE) * 100)
  const zeroPx = `${ZERO_PCT}%`

  return (
    <div className="flex items-center gap-3 h-5">
      <div className="relative w-full h-3 rounded-sm overflow-visible">
        {/* Zero line */}
        <div
          className="absolute top-0 bottom-0 w-px bg-[var(--c-muted)]/30 z-10"
          style={{ left: zeroPx }}
        />
        {/* Bar */}
        <div
          className="absolute top-0 h-full rounded-sm transition-all duration-700"
          style={{
            background: meta.bar,
            opacity: 0.85,
            ...(isPos
              ? { left: zeroPx, width: `${barPct}%` }
              : { right: `${100 - ZERO_PCT}%`, width: `${barPct}%` }),
          }}
        />
      </div>
      <span className={`font-mono text-xs w-14 text-right shrink-0 ${isPos ? 'text-forest-400' : 'text-red-400'}`}>
        {isPos ? '+' : ''}{value.toFixed(3)}
      </span>
    </div>
  )
}

export default function AOPCChart({ highlightModel }) {
  return (
    <div className="rounded-2xl border border-[var(--c-border)] bg-[var(--c-surface)] p-4 space-y-4">
      <div className="flex items-center justify-between">
        <span className="section-label">AOPC Comparison</span>
        <div className="flex items-center gap-3 flex-wrap">
          {Object.entries(XAI_COLORS).map(([k, v]) => (
            <div key={k} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full" style={{ background: v.bar }} />
              <span className="font-mono text-[10px] text-[var(--c-muted)]">{v.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-5">
        {Object.keys(AOPC_DATA).map((modelKey) => {
          const isHighlighted = modelKey === (highlightModel || 'efficientnet_b0')
          return (
            <div key={modelKey}
              className={[
                'rounded-xl p-3 border transition-all',
                isHighlighted
                  ? 'border-forest-700 bg-forest-950/30'
                  : 'border-transparent',
              ].join(' ')}
            >
              <div className="flex items-center gap-2 mb-3">
                <span className={[
                  'font-display font-semibold text-xs',
                  isHighlighted ? 'text-forest-300' : 'text-[var(--c-muted)]',
                ].join(' ')}>
                  {MODEL_LABELS[modelKey]}
                </span>
                {isHighlighted && (
                  <span className="tag bg-forest-900/60 text-forest-400 border border-forest-800 text-[9px]">
                    ★ Best
                  </span>
                )}
              </div>
              <div className="space-y-1.5">
                {Object.keys(XAI_COLORS).map((xai) => (
                  <BarRow key={xai} model={modelKey} xai={xai} value={AOPC_DATA[modelKey][xai]} />
                ))}
              </div>
            </div>
          )
        })}
      </div>

      <p className="text-[10px] font-mono text-[var(--c-muted)] leading-relaxed pt-1 border-t border-[var(--c-border)]">
        Positive AOPC indicates that removing the highlighted pixels decreases model confidence —
        proof the model looks at disease symptoms, not background clutter.
      </p>
    </div>
  )
}
