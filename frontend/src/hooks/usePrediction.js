import { useState, useCallback } from 'react'
import { predictDisease } from '../services/api'

const STAGES = [
  { id: 'upload',   label: 'Uploading image …',             pct: 15 },
  { id: 'resnet',   label: 'Running ResNet-18 …',           pct: 30 },
  { id: 'effnet',   label: 'Running EfficientNet-B0 …',     pct: 55 },
  { id: 'densenet', label: 'Running DenseNet-121 …',        pct: 72 },
  { id: 'xai',      label: 'Generating XAI explanations …', pct: 88 },
  { id: 'done',     label: 'Finalising results …',          pct: 97 },
]

export function usePrediction() {
  const [status,   setStatus]   = useState('idle')   // idle | uploading | processing | done | error
  const [progress, setProgress] = useState(0)
  const [stage,    setStage]    = useState(null)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)
  const [duration, setDuration] = useState(null)

  const reset = useCallback(() => {
    setStatus('idle')
    setProgress(0)
    setStage(null)
    setResult(null)
    setError(null)
    setDuration(null)
  }, [])

  const predict = useCallback(async (imageFile, selectedModels) => {
    setStatus('uploading')
    setError(null)
    setResult(null)
    setProgress(0)

    // Simulate stage progression during backend processing
    let stageIdx = 0
    const advanceStage = () => {
      if (stageIdx < STAGES.length) {
        setStage(STAGES[stageIdx])
        setProgress(STAGES[stageIdx].pct)
        stageIdx++
      }
    }

    advanceStage()
    const ticker = setInterval(advanceStage, 1800)

    try {
      const { data, durationMs } = await predictDisease(
        imageFile,
        selectedModels,
        (pct) => {
          if (pct < 15) setProgress(pct)
        }
      )
      clearInterval(ticker)
      setProgress(100)
      setStage({ id: 'done', label: 'Analysis complete!', pct: 100 })
      setResult(data)
      setDuration(durationMs)
      setStatus('done')
    } catch (err) {
      clearInterval(ticker)
      setError(err.message || 'Prediction failed.')
      setStatus('error')
      setProgress(0)
    }
  }, [])

  return { status, progress, stage, result, error, duration, predict, reset }
}
