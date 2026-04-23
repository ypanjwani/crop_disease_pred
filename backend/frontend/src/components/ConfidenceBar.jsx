export default function ConfidenceBar({ value, label, size = 'md', showPercent = true }) {
  const pct = Math.round(value * 100)

  const color =
    pct >= 85 ? 'from-forest-700 to-forest-400' :
    pct >= 60 ? 'from-yellow-700 to-yellow-400' :
                'from-red-800 to-red-500'

  const textColor =
    pct >= 85 ? 'text-forest-400' :
    pct >= 60 ? 'text-yellow-400' :
                'text-red-400'

  const heights = { sm: 'h-1', md: 'h-1.5', lg: 'h-2' }

  return (
    <div className="w-full">
      {(label || showPercent) && (
        <div className="flex justify-between items-center mb-1">
          {label && <span className="text-xs text-[var(--c-muted)] truncate">{label}</span>}
          {showPercent && (
            <span className={`font-mono text-xs font-semibold ${textColor} ml-auto`}>
              {pct}%
            </span>
          )}
        </div>
      )}
      <div className={`w-full ${heights[size]} rounded-full bg-[var(--c-border)] overflow-hidden`}>
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color} transition-all duration-1000 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
