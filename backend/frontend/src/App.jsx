import { useState, useEffect } from 'react'
import { Leaf, Github, BookOpen, Wifi, WifiOff, AlertCircle } from 'lucide-react'
import ImageUpload from './components/ImageUpload'
import ModelSelector from './components/ModelSelector'
import ProgressIndicator from './components/ProgressIndicator'
import ResultsDashboard from './components/ResultsDashboard'
import { usePrediction } from './hooks/usePrediction'
import { checkHealth } from './services/api'

function StatusDot({ online }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className={[
        'w-1.5 h-1.5 rounded-full',
        online === null  ? 'bg-yellow-400 animate-pulse' :
        online           ? 'bg-forest-400' :
                           'bg-red-500',
      ].join(' ')} />
      <span className={[
        'font-mono text-[10px]',
        online === null ? 'text-yellow-400' :
        online          ? 'text-forest-400' :
                          'text-red-400',
      ].join(' ')}>
        {online === null ? 'connecting' : online ? 'API online' : 'API offline'}
      </span>
    </div>
  )
}

export default function App() {
  const [imageFile,     setImageFile]     = useState(null)
  const [selectedModels, setSelectedModels] = useState(['resnet18', 'efficientnet_b0', 'densenet121'])
  const [apiOnline,     setApiOnline]     = useState(null)

  const { status, progress, stage, result, error, duration, predict, reset } = usePrediction()

  // Health-check poll
  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        await checkHealth()
        if (!cancelled) setApiOnline(true)
      } catch {
        if (!cancelled) setApiOnline(false)
      }
    }
    check()
    const id = setInterval(check, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const handleReset = () => {
    reset()
    setImageFile(null)
  }

  const canAnalyse =
    imageFile &&
    selectedModels.length > 0 &&
    status !== 'uploading' &&
    status !== 'processing'

  const isProcessing = status === 'uploading' || status === 'processing'

  return (
    <div className="min-h-screen mesh-bg">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 glass border-b border-[var(--c-border)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-forest-900 border border-forest-700 flex items-center justify-center">
              <Leaf size={16} className="text-forest-400" />
            </div>
            <div>
              <span className="font-display font-bold text-sm text-[var(--c-text)]">CropXAI</span>
              <span className="hidden sm:inline font-mono text-xs text-[var(--c-muted)] ml-2">
                Explainable Plant Disease Detection
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <StatusDot online={apiOnline} />
            <a
              href="https://arxiv.org"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-1.5 btn-ghost py-1.5"
            >
              <BookOpen size={13} /> Paper
            </a>
          </div>
        </div>
      </header>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">

        {status === 'done' ? (
          /* ── Results view ─────────────────────────────────────────────── */
          <ResultsDashboard result={result} onReset={handleReset} duration={duration} />
        ) : (
          /* ── Upload + config view ─────────────────────────────────────── */
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">

            {/* Left column: Upload + Progress */}
            <div className="lg:col-span-2 space-y-5">
              {/* Hero text */}
              <div className="animate-fade-up" style={{ animationDelay: '0ms' }}>
                <p className="section-label mb-2">Research-grade XAI pipeline</p>
                <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-[var(--c-text)] leading-tight">
                  Diagnose plant disease
                  <br />
                  <span className="text-forest-400">and explain why.</span>
                </h1>
                <p className="mt-3 text-sm text-[var(--c-muted)] max-w-lg leading-relaxed">
                  Upload a leaf image. Three CNN architectures run simultaneously,
                  generating Grad-CAM, LIME, and Integrated Gradient explanations —
                  quantified with AOPC for trustworthy results.
                </p>
              </div>

              {/* Upload zone */}
              <div className="animate-fade-up" style={{ animationDelay: '80ms' }}>
                <ImageUpload
                  onImageSelected={setImageFile}
                  disabled={isProcessing}
                />
              </div>

              {/* Progress or error */}
              {isProcessing && (
                <ProgressIndicator progress={progress} stage={stage} />
              )}

              {status === 'error' && (
                <div className="rounded-xl border border-red-800 bg-red-950/30 p-4 flex gap-3">
                  <AlertCircle size={18} className="text-red-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-sm text-red-300">Analysis failed</p>
                    <p className="text-xs text-red-400/80 mt-1">{error}</p>
                    <button onClick={reset} className="mt-2 text-xs font-mono text-red-400 underline underline-offset-2">
                      Try again
                    </button>
                  </div>
                </div>
              )}

              {/* Analyse button */}
              {!isProcessing && (
                <button
                  onClick={() => predict(imageFile, selectedModels)}
                  disabled={!canAnalyse}
                  className="btn-primary w-full py-3 text-base animate-fade-up"
                  style={{ animationDelay: '120ms' }}
                >
                  {imageFile ? 'Analyse Leaf →' : 'Upload an image to begin'}
                </button>
              )}
            </div>

            {/* Right column: Config panel */}
            <div className="space-y-5 animate-fade-up" style={{ animationDelay: '160ms' }}>
              {/* Model selector */}
              <div className="glass rounded-2xl p-4">
                <ModelSelector
                  selected={selectedModels}
                  onChange={setSelectedModels}
                  disabled={isProcessing}
                />
              </div>

              {/* Pipeline info card */}
              <div className="glass rounded-2xl p-4 space-y-3">
                <span className="section-label">Pipeline Overview</span>
                {[
                  { step: '01', label: 'CNN Inference',         desc: 'Top-5 class predictions with softmax confidence' },
                  { step: '02', label: 'Grad-CAM',              desc: 'Gradient-weighted spatial activation heatmaps' },
                  { step: '03', label: 'LIME',                  desc: 'Superpixel-based local linear surrogate model' },
                  { step: '04', label: 'Integrated Gradients',  desc: 'Axiomatic attribution along interpolation path' },
                  { step: '05', label: 'AOPC Validation',       desc: 'Quantitative faithfulness via perturbation curve' },
                ].map((item) => (
                  <div key={item.step} className="flex gap-3">
                    <span className="font-mono text-xs text-[var(--c-muted)] shrink-0 pt-0.5">{item.step}</span>
                    <div>
                      <p className="text-xs font-semibold text-[var(--c-text)]">{item.label}</p>
                      <p className="text-[10px] text-[var(--c-muted)] leading-relaxed">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="mt-16 border-t border-[var(--c-border)] py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="font-mono text-xs text-[var(--c-muted)]">
            CropXAI · ResNet-18 · EfficientNet-B0 · DenseNet-121
          </p>
          <p className="font-mono text-xs text-[var(--c-muted)]">
            Grad-CAM · LIME · Integrated Gradients · AOPC
          </p>
        </div>
      </footer>
    </div>
  )
}
