'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { fetchEmails, type Email } from '../../lib/api'
import { Zap, Mail, ChevronRight, AlertCircle } from 'lucide-react'

const DIFFICULTY_STYLES: Record<string, string> = {
  easy:   'bg-green-900/40 text-green-400 border border-green-800/50',
  medium: 'bg-yellow-900/40 text-yellow-400 border border-yellow-800/50',
  hard:   'bg-red-900/40 text-red-400 border border-red-800/50',
}

const TYPE_STYLES: Record<string, string> = {
  quote_request:      'text-blue-400',
  availability_check: 'text-purple-400',
  multi_item:         'text-amber-400',
  ambiguous:          'text-orange-400',
  hard_ambiguous:     'text-orange-400',
  adversarial:        'text-red-400',
}

const TYPE_LABELS: Record<string, string> = {
  quote_request:      'Quote',
  availability_check: 'Availability',
  multi_item:         'Multi-item',
  ambiguous:          'Ambiguous',
  hard_ambiguous:     'Ambiguous',
  adversarial:        'Adversarial',
}

export default function InboxPage() {
  const [emails, setEmails] = useState<Email[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState<string | null>(null)
  const router = useRouter()

  useEffect(() => {
    fetchEmails()
      .then(setEmails)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-primary)' }}>

      {/* Header */}
      <header style={{ borderBottom: '1px solid var(--bg-border)', background: 'var(--bg-secondary)' }}>
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded flex items-center justify-center"
                 style={{ background: 'var(--amber)' }}>
              <Zap size={16} className="text-black" fill="black" />
            </div>
            <div>
              <span className="text-sm font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
                PartsPilot
              </span>
              <span className="text-xs ml-2" style={{ color: 'var(--text-muted)' }}>
                AI Sales Agent
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Agent online</span>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">

        {/* Page title */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-1">
            <Mail size={16} style={{ color: 'var(--text-muted)' }} />
            <h1 className="text-xs font-medium tracking-widest uppercase"
                style={{ color: 'var(--text-muted)' }}>
              Incoming Requests
            </h1>
          </div>
          <p className="text-2xl font-semibold" style={{ color: 'var(--text-primary)' }}>
            Customer Inbox
          </p>
          {!loading && (
            <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
              {emails.length} emails — click any to run the agent
            </p>
          )}
        </div>

        {/* Hosting disclaimer since on free Render tier */}
        <div className="flex items-center gap-2 mb-6 px-3 py-2 rounded-lg"
          style={{ background: 'var(--bg-elevated)', border: '1px solid var(--bg-border)' }}>
          <AlertCircle size={13} style={{ color: 'var(--text-muted)' }} className="shrink-0" />
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Hosted on Render's free tier — if a request fails, wait a moment and try again.
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-3 p-4 rounded-lg mb-6"
               style={{ background: '#2D1515', border: '1px solid #7F1D1D' }}>
            <AlertCircle size={16} className="text-red-400 shrink-0" />
            <div>
              <p className="text-sm font-medium text-red-400">Backend not reachable</p>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                Make sure the FastAPI server is running: <code className="mono text-amber-400">uvicorn backend.app.main:app --reload --port 8000</code>
              </p>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="space-y-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-20 rounded-lg animate-pulse"
                   style={{ background: 'var(--bg-elevated)', animationDelay: `${i * 60}ms` }} />
            ))}
          </div>
        )}

        {/* Email list */}
        {!loading && !error && (
          <div className="space-y-2">
            {emails.map((email, i) => (
              <button
                key={email.email_id}
                onClick={() => router.push(`/quote?emailId=${email.email_id}`)}
                className="w-full text-left rounded-lg p-4 transition-all duration-150 group animate-fade-in"
                style={{
                  background:    'var(--bg-elevated)',
                  border:        '1px solid var(--bg-border)',
                  animationDelay: `${i * 30}ms`,
                }}
                onMouseEnter={e => {
                  const el = e.currentTarget
                  el.style.borderColor = 'var(--amber)'
                  el.style.background  = 'var(--bg-secondary)'
                }}
                onMouseLeave={e => {
                  const el = e.currentTarget
                  el.style.borderColor = 'var(--bg-border)'
                  el.style.background  = 'var(--bg-elevated)'
                }}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">

                    {/* Company + subject row */}
                    <div className="flex items-center gap-2 mb-1">
                      {email.company_name && (
                        <span className="text-xs font-medium px-2 py-0.5 rounded"
                              style={{ background: 'var(--bg-border)', color: 'var(--text-secondary)' }}>
                          {email.company_name}
                        </span>
                      )}
                      <span className={`text-xs font-medium ${TYPE_STYLES[email.email_type] || 'text-gray-400'}`}>
                        {TYPE_LABELS[email.email_type] || email.email_type}
                      </span>
                    </div>

                    {/* Subject */}
                    <p className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                      {email.subject || '(no subject)'}
                    </p>

                    {/* Body preview */}
                    <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-muted)' }}>
                      {email.body.slice(0, 100).replace(/\n/g, ' ')}
                    </p>

                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    {/* Difficulty badge */}
                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${DIFFICULTY_STYLES[email.difficulty] || ''}`}>
                      {email.difficulty}
                    </span>
                    <ChevronRight size={14} style={{ color: 'var(--text-muted)' }}
                                  className="group-hover:translate-x-0.5 transition-transform" />
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
