import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300_000, // 5 min — XAI generation is compute-heavy on CPU
})

// ── Request interceptor ───────────────────────────────────────────────────────
api.interceptors.request.use((config) => {
  config.metadata = { startTime: Date.now() }
  return config
})

// ── Response interceptor ──────────────────────────────────────────────────────
api.interceptors.response.use(
  (response) => {
    response.durationMs = Date.now() - response.config.metadata.startTime
    return response
  },
  (error) => {
    const msg =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'Unknown error'
    return Promise.reject(new Error(msg))
  }
)

// ── API calls ─────────────────────────────────────────────────────────────────

/**
 * POST /api/v1/predict
 * @param {File}     imageFile      - plant leaf image
 * @param {string[]} selectedModels - e.g. ['efficientnet_b0'] or null for all
 * @param {Function} onUploadProgress
 */
export async function predictDisease(imageFile, selectedModels = null, onUploadProgress) {
  const formData = new FormData()
  formData.append('file', imageFile)
  if (selectedModels?.length) {
    formData.append('models', selectedModels.join(','))
  }

  const response = await api.post('/api/v1/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onUploadProgress
      ? (evt) => onUploadProgress(Math.round((evt.loaded * 100) / evt.total))
      : undefined,
  })

  return { data: response.data, durationMs: response.durationMs }
}

/** GET /api/v1/models */
export async function fetchModels() {
  const { data } = await api.get('/api/v1/models')
  return data.models
}

/** GET /api/v1/classes */
export async function fetchClasses() {
  const { data } = await api.get('/api/v1/classes')
  return data.classes
}

/** GET /health */
export async function checkHealth() {
  const { data } = await api.get('/health')
  return data
}
