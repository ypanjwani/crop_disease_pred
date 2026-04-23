import { Star, Zap, Layers, GitBranch } from 'lucide-react'

const MODEL_META = {
  resnet18: {
    key:   'resnet18',
    label: 'ResNet-18',
    desc:  'Residual skip connections. Fastest inference.',
    icon:  GitBranch,
    aopc:  { ig: -0.107, gradcam: -0.157, lime: -0.013 },
    badge: null,
  },
  efficientnet_b0: {
    key:   'efficientnet_b0',
    label: 'EfficientNet-B0',
    desc:  'Compound scaling. Most faithful XAI.',
    icon:  Star,
    aopc:  { ig: 0.136, gradcam: 0.054, lime: 0.087 },
    badge: 'Recommended',
  },
  densenet121: {
    key:   'densenet121',
    label: 'DenseNet-121',
    desc:  'Dense feature reuse. Strong representation.',
    icon:  Layers,
    aopc:  { ig: 0.067, gradcam: -0.197, lime: -0.128 },
    badge: null,
  },
}

function AopcBadge({ value }) {
  const positive = value > 0
  return (
    <span
      className={[
        'font-mono text-[10px] px-1.5 py-0.5 rounded',
        positive
          ? 'bg-forest-950 text-forest-400 border border-forest-800'
          : 'bg-red-950/50 text-red-400 border border-red-900/50',
      ].join(' ')}
    >
      {positive ? '+' : ''}{value.toFixed(3)}
    </span>
  )
}

export default function ModelSelector({ selected, onChange, disabled }) {
  const toggle = (key) => {
    if (disabled) return
    if (selected.includes(key)) {
      if (selected.length > 1) onChange(selected.filter((k) => k !== key))
    } else {
      onChange([...selected, key])
    }
  }

  const toggleAll = () => {
    if (disabled) return
    if (selected.length === 3) onChange(['efficientnet_b0'])
    else onChange(Object.keys(MODEL_META))
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="section-label">Select Models</span>
        <button
          onClick={toggleAll}
          disabled={disabled}
          className="text-[10px] font-mono text-[var(--c-muted)] hover:text-forest-400 
                     transition-colors disabled:opacity-40"
        >
          {selected.length === 3 ? '— deselect all' : '+ select all'}
        </button>
      </div>

      <div className="grid grid-cols-1 gap-2">
        {Object.values(MODEL_META).map((m) => {
          const Icon = m.icon
          const active = selected.includes(m.key)

          return (
            <button
              key={m.key}
              onClick={() => toggle(m.key)}
              disabled={disabled}
              className={[
                'relative w-full text-left p-3 rounded-xl border transition-all duration-200',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                active
                  ? 'border-forest-700 bg-forest-950/50 shadow-[0_0_12px_rgba(34,197,94,0.06)]'
                  : 'border-[var(--c-border)] bg-[var(--c-surface)] hover:border-forest-800',
              ].join(' ')}
            >
              {/* Selection dot */}
              <div className={[
                'absolute top-3 right-3 w-4 h-4 rounded-full border-2 transition-all',
                active
                  ? 'border-forest-500 bg-forest-500'
                  : 'border-[var(--c-border)]',
              ].join(' ')}>
                {active && (
                  <svg viewBox="0 0 16 16" fill="white" className="w-full h-full p-0.5">
                    <path d="M13 4L6.5 11.5 3 8" stroke="white" strokeWidth="2" fill="none" strokeLinecap="round"/>
                  </svg>
                )}
              </div>

              <div className="flex items-start gap-3 pr-6">
                <div className={[
                  'mt-0.5 p-1.5 rounded-lg border',
                  active
                    ? 'bg-forest-950 border-forest-700 text-forest-400'
                    : 'bg-[var(--c-bg)] border-[var(--c-border)] text-[var(--c-muted)]',
                ].join(' ')}>
                  <Icon size={14} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={[
                      'font-display font-semibold text-sm',
                      active ? 'text-[var(--c-text)]' : 'text-[var(--c-muted)]',
                    ].join(' ')}>
                      {m.label}
                    </span>
                    {m.badge && (
                      <span className="tag bg-forest-900/60 text-forest-300 border border-forest-700">
                        ★ {m.badge}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-[var(--c-muted)] mt-0.5">{m.desc}</p>

                  {/* AOPC scores */}
                  <div className="flex items-center gap-2 mt-2 flex-wrap">
                    <span className="text-[10px] text-[var(--c-muted)] font-mono">AOPC:</span>
                    <AopcBadge value={m.aopc.ig} />
                    <span className="text-[10px] text-[var(--c-muted)]/50 font-mono">IG</span>
                    <AopcBadge value={m.aopc.gradcam} />
                    <span className="text-[10px] text-[var(--c-muted)]/50 font-mono">GC</span>
                    <AopcBadge value={m.aopc.lime} />
                    <span className="text-[10px] text-[var(--c-muted)]/50 font-mono">LM</span>
                  </div>
                </div>
              </div>
            </button>
          )
        })}
      </div>

      <p className="text-[10px] text-[var(--c-muted)] font-mono leading-relaxed">
        AOPC = Area Over Perturbation Curve. Positive values → faithful explanations.
        EfficientNet-B0 scores positive across all XAI methods.
      </p>
    </div>
  )
}
