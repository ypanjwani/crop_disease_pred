import { Loader2 } from 'lucide-react'

export default function ProgressIndicator({ progress, stage }) {
  return (
    <div className="w-full space-y-4 animate-fade-in">
      {/* Animated scanner */}
      <div className="relative rounded-xl border border-forest-800 bg-forest-950/30 overflow-hidden h-32 flex items-center justify-center">
        {/* Scan line */}
        <div className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-forest-400 to-transparent animate-scan" />
        {/* Grid overlay */}
        <div className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `linear-gradient(var(--c-border) 1px, transparent 1px),
                              linear-gradient(90deg, var(--c-border) 1px, transparent 1px)`,
            backgroundSize: '24px 24px',
          }}
        />
        <div className="relative z-10 flex flex-col items-center gap-2">
          <Loader2 size={28} className="text-forest-400 animate-spin" />
          <p className="font-mono text-xs text-forest-300">{stage?.label || 'Initialising…'}</p>
        </div>
      </div>

      {/* Progress bar */}
      <div>
        <div className="flex justify-between items-center mb-1.5">
          <span className="section-label">Analysis Progress</span>
          <span className="font-mono text-xs text-forest-400">{progress}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-[var(--c-border)] overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-forest-700 to-forest-400 transition-all duration-700 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Stage steps */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { id: 'models',  label: 'CNN Inference' },
          { id: 'xai',     label: 'XAI Generation' },
          { id: 'done',    label: 'Finalising' },
        ].map((s, i) => {
          const pctMap = [30, 88, 100]
          const reached = progress >= pctMap[i]
          return (
            <div key={s.id}
              className={[
                'px-3 py-2 rounded-lg border text-center transition-all duration-500',
                reached
                  ? 'border-forest-700 bg-forest-950/40 text-forest-300'
                  : 'border-[var(--c-border)] text-[var(--c-muted)]',
              ].join(' ')}
            >
              <div className={['w-2 h-2 rounded-full mx-auto mb-1 transition-all', reached ? 'bg-forest-400' : 'bg-[var(--c-border)]'].join(' ')} />
              <p className="text-[10px] font-mono">{s.label}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
