'use client'

import { useEffect, useState, useCallback } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { runAgent, fetchEmail, type Email, type AgentResult, type ToolCall, type LineItem } from '../../lib/api'
import {
  Zap, ArrowLeft, Search, Package, GitBranch,
  History, FileText, CheckCircle, XCircle,
  Clock, AlertTriangle, Loader2, ChevronDown, ChevronUp
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Tool call display config
// ---------------------------------------------------------------------------

const TOOL_CONFIG: Record<string, { label: string; icon: React.ReactNode; color: string }> = {
  get_customer_history: { label: 'Customer history',  icon: <History  size={12} />, color: '#8B5CF6' },
  search_catalog:       { label: 'Search catalog',    icon: <Search   size={12} />, color: '#3B82F6' },
  check_inventory:      { label: 'Check inventory',   icon: <Package  size={12} />, color: '#06B6D4' },
  find_substitution:    { label: 'Find substitution', icon: <GitBranch size={12} />, color: '#F59E0B' },
  draft_quote:          { label: 'Draft quote',       icon: <FileText  size={12} />, color: '#22C55E' },
}

// ---------------------------------------------------------------------------
// Status badge for line items
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: string }) {
  const configs: Record<string, { label: string; bg: string; text: string }> = {
    in_stock:    { label: 'In Stock',    bg: '#14532D', text: '#86EFAC' },
    low_stock:   { label: 'Low Stock',   bg: '#713F12', text: '#FDE68A' },
    out_of_stock:{ label: 'Out of Stock',bg: '#7F1D1D', text: '#FCA5A5' },
    substituted: { label: 'Substituted', bg: '#1E3A5F', text: '#93C5FD' },
    lead_time:   { label: 'Lead Time',   bg: '#3B1F6E', text: '#C4B5FD' },
  }
  const c = configs[status] || { label: status, bg: '#1F1F2E', text: '#A1A1AA' }
  return (
    <span className="text-xs font-medium px-2 py-0.5 rounded"
          style={{ background: c.bg, color: c.text }}>
      {c.label}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Single tool call row in trajectory panel
// ---------------------------------------------------------------------------

function ToolCallRow({ tc, index }: { tc: ToolCall; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const config = TOOL_CONFIG[tc.tool] || { label: tc.tool, icon: <Zap size={12} />, color: '#A1A1AA' }

  const getPreview = () => {
    if (tc.tool === 'check_inventory')   return (tc.input as any).mpn
    if (tc.tool === 'find_substitution') return (tc.input as any).mpn
    if (tc.tool === 'search_catalog')    return (tc.input as any).query
    if (tc.tool === 'get_customer_history') return (tc.input as any).customer_id
    if (tc.tool === 'draft_quote') {
      const items = (tc.output as any).line_items || []
      return `${items.length} line item${items.length !== 1 ? 's' : ''}`
    }
    return ''
  }

  const getResultPreview = () => {
    const o = tc.output as any
    if (tc.tool === 'check_inventory') {
      if (!o.found) return '✗ Not in catalog'
      return `${o.stock_qty} units · $${o.unit_price_usd} · ${o.status}`
    }
    if (tc.tool === 'find_substitution') {
      if (!o.found) return '✗ No substitutes found'
      return `${o.substitutes?.length} substitute${o.substitutes?.length !== 1 ? 's' : ''} found`
    }
    if (tc.tool === 'get_customer_history') {
      if (!o.found) return '✗ Not found'
      return `${o.company_name} · ${o.price_tier} · ${o.order_count} orders`
    }
    if (tc.tool === 'search_catalog') {
      return `${o.results_count} results`
    }
    if (tc.tool === 'draft_quote') return '✓ Quote ready'
    return ''
  }

  return (
    <div className="animate-slide-in" style={{ animationDelay: `${index * 80}ms` }}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-3 rounded-lg transition-colors"
        style={{ background: expanded ? 'var(--bg-secondary)' : 'transparent',
                 border: '1px solid transparent' }}
        onMouseEnter={e => { if (!expanded) (e.currentTarget as HTMLElement).style.background = '#0F0F16' }}
        onMouseLeave={e => { if (!expanded) (e.currentTarget as HTMLElement).style.background = 'transparent' }}
      >
        <div className="flex items-center gap-3">
          {/* Step number */}
          <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-xs font-bold"
               style={{ background: `${config.color}22`, color: config.color, border: `1px solid ${config.color}44` }}>
            {index + 1}
          </div>

          {/* Tool icon + name */}
          <div className="flex items-center gap-1.5" style={{ color: config.color }}>
            {config.icon}
            <span className="text-xs font-medium">{config.label}</span>
          </div>

          {/* Input preview */}
          <span className="mono text-xs truncate flex-1" style={{ color: 'var(--text-muted)' }}>
            {getPreview()}
          </span>

          {/* Result preview */}
          <span className="text-xs shrink-0" style={{ color: 'var(--text-secondary)' }}>
            {getResultPreview()}
          </span>

          {/* Latency */}
          <span className="text-xs shrink-0" style={{ color: 'var(--text-muted)' }}>
            {tc.latency_ms}ms
          </span>

          {expanded ? <ChevronUp size={12} style={{ color: 'var(--text-muted)' }} />
                    : <ChevronDown size={12} style={{ color: 'var(--text-muted)' }} />}
        </div>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="mx-3 mb-2 p-3 rounded-lg text-xs font-mono overflow-auto"
             style={{ background: '#0A0A0F', border: '1px solid var(--bg-border)',
                      color: 'var(--text-muted)', maxHeight: '200px' }}>
          <div className="mb-2" style={{ color: 'var(--text-secondary)' }}>Input:</div>
          <pre>{JSON.stringify(tc.input, null, 2)}</pre>
          <div className="mt-3 mb-2" style={{ color: 'var(--text-secondary)' }}>Output:</div>
          <pre>{JSON.stringify(tc.output, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Quote table
// ---------------------------------------------------------------------------

function QuoteTable({ result }: { result: AgentResult }) {
  const q = result.quote
  if (!q) return null

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--bg-border)' }}>

      {/* Quote header */}
      <div className="px-5 py-4 flex items-center justify-between"
           style={{ background: 'var(--bg-elevated)', borderBottom: '1px solid var(--bg-border)' }}>
        <div className="flex items-center gap-2">
          <CheckCircle size={14} className="text-green-400" />
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Quote Ready
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
          <span>Valid {q.valid_days} days</span>
          <span className="font-semibold text-base" style={{ color: 'var(--amber)' }}>
            ${q.subtotal_usd.toFixed(2)} {q.currency}
          </span>
        </div>
      </div>

      {/* Line items table */}
      <div style={{ background: 'var(--bg-secondary)' }}>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: '1px solid var(--bg-border)' }}>
              {['Part Number', 'Description', 'Qty', 'Unit Price', 'Total', 'Status'].map(h => (
                <th key={h} className="text-left px-4 py-2.5 font-medium"
                    style={{ color: 'var(--text-muted)' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {q.line_items.map((item, i) => (
              <tr key={i} style={{ borderBottom: i < q.line_items.length - 1 ? '1px solid var(--bg-border)' : 'none' }}>
                <td className="px-4 py-3">
                  <div className="mono font-medium" style={{ color: 'var(--amber-bright)' }}>
                    {item.mpn}
                  </div>
                  {item.substitute_for && (
                    <div className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                      sub for {item.substitute_for}
                    </div>
                  )}
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {item.manufacturer}
                  </div>
                </td>
                <td className="px-4 py-3 max-w-xs">
                  <span style={{ color: 'var(--text-secondary)' }}>{item.description}</span>
                </td>
                <td className="px-4 py-3" style={{ color: 'var(--text-primary)' }}>
                  {item.requested_qty?.toLocaleString()}
                </td>
                <td className="px-4 py-3 mono" style={{ color: 'var(--text-primary)' }}>
                  ${item.unit_price_usd?.toFixed(4)}
                </td>
                <td className="px-4 py-3 mono font-medium" style={{ color: 'var(--text-primary)' }}>
                  ${item.total_price_usd?.toFixed(2)}
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-col gap-1">
                    <StatusBadge status={item.status} />
                    {item.lead_time_days > 0 && (
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {item.lead_time_days}d lead
                      </span>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Notes + clarification */}
      {(q.notes || q.clarification_needed) && (
        <div className="px-5 py-4" style={{ borderTop: '1px solid var(--bg-border)', background: 'var(--bg-elevated)' }}>
          {q.clarification_needed && (
            <div className="flex items-start gap-2 mb-3">
              <AlertTriangle size={13} className="text-yellow-400 shrink-0 mt-0.5" />
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                {q.clarification_needed}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main quote page
// ---------------------------------------------------------------------------

export default function QuotePage() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const emailId = searchParams.get('emailId')

  const [email,   setEmail]   = useState<Email | null>(null)
  const [result,  setResult]  = useState<AgentResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)

  // Fetch email details
  useEffect(() => {
    if (!emailId) return
    fetchEmail(emailId).then(setEmail).catch(e => setError(e.message))
  }, [emailId])

  // Elapsed timer while agent runs
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => setElapsed(e => e + 100), 100)
    return () => clearInterval(t)
  }, [running])

  const handleRun = useCallback(async () => {
    if (!email) return
    setRunning(true)
    setResult(null)
    setError(null)
    setElapsed(0)
    try {
      const r = await runAgent(email.body, email.email_id, email.customer_id)
      setResult(r)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }, [email])

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-primary)' }}>

      {/* Header */}
      <header style={{ borderBottom: '1px solid var(--bg-border)', background: 'var(--bg-secondary)' }}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button onClick={() => router.push('/inbox')}
                    className="flex items-center gap-1.5 text-xs transition-colors"
                    style={{ color: 'var(--text-muted)' }}
                    onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = 'var(--text-primary)'}
                    onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = 'var(--text-muted)'}>
              <ArrowLeft size={13} />
              Inbox
            </button>
            <div style={{ width: '1px', height: '14px', background: 'var(--bg-border)' }} />
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded flex items-center justify-center"
                   style={{ background: 'var(--amber)' }}>
                <Zap size={12} className="text-black" fill="black" />
              </div>
              <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                PartsPilot
              </span>
            </div>
          </div>

          {email && !result && !running && (
            <button onClick={handleRun}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
                    style={{ background: 'var(--amber)', color: '#000' }}
                    onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'var(--amber-bright)'}
                    onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'var(--amber)'}>
              <Zap size={13} fill="black" />
              Run Agent
            </button>
          )}

          {running && (
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm"
                 style={{ background: 'var(--bg-elevated)', border: '1px solid var(--bg-border)', color: 'var(--text-secondary)' }}>
              <Loader2 size={13} className="animate-spin" style={{ color: 'var(--amber)' }} />
              <span>{(elapsed / 1000).toFixed(1)}s</span>
            </div>
          )}

          {result && !running && (
            <button onClick={handleRun}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
                    style={{ background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                             border: '1px solid var(--bg-border)' }}>
              Re-run
            </button>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {!emailId && (
          <div className="text-center py-20" style={{ color: 'var(--text-muted)' }}>
            No email selected. <button onClick={() => router.push('/inbox')}
                                       className="underline" style={{ color: 'var(--amber)' }}>
              Go to inbox
            </button>
          </div>
        )}

        {email && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Left — Email + response */}
            <div className="space-y-4">

              {/* Email card */}
              <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--bg-border)' }}>
                <div className="px-5 py-4" style={{ background: 'var(--bg-elevated)', borderBottom: '1px solid var(--bg-border)' }}>
                  <div className="flex items-center gap-2 mb-2">
                    {email.company_name && (
                      <span className="text-xs px-2 py-0.5 rounded font-medium"
                            style={{ background: 'var(--bg-border)', color: 'var(--text-secondary)' }}>
                        {email.company_name}
                      </span>
                    )}
                    {email.contact_name && (
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {email.contact_name}
                      </span>
                    )}
                  </div>
                  <p className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                    {email.subject || '(no subject)'}
                  </p>
                </div>
                <div className="px-5 py-4" style={{ background: 'var(--bg-secondary)' }}>
                  <pre className="text-sm whitespace-pre-wrap font-sans leading-relaxed"
                       style={{ color: 'var(--text-secondary)' }}>
                    {email.body}
                  </pre>
                </div>
              </div>

              {/* Agent response */}
              {result?.final_message && (
                <div className="rounded-xl overflow-hidden animate-fade-in"
                     style={{ border: '1px solid #22C55E44' }}>
                  <div className="px-5 py-3 flex items-center gap-2"
                       style={{ background: '#052E16', borderBottom: '1px solid #22C55E44' }}>
                    <CheckCircle size={13} className="text-green-400" />
                    <span className="text-xs font-medium text-green-400">Draft Response</span>
                    <span className="text-xs ml-auto" style={{ color: 'var(--text-muted)' }}>
                      {result.total_latency_ms}ms · {result.tool_calls.length} tool calls
                    </span>
                  </div>
                  <div className="px-5 py-4" style={{ background: 'var(--bg-secondary)' }}>
                    <pre className="text-sm whitespace-pre-wrap font-sans leading-relaxed"
                         style={{ color: 'var(--text-primary)' }}>
                      {result.final_message}
                    </pre>
                  </div>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="p-4 rounded-xl flex items-start gap-3"
                     style={{ background: '#2D1515', border: '1px solid #7F1D1D' }}>
                  <XCircle size={14} className="text-red-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-red-400">{error}</p>
                </div>
              )}
            </div>

            {/* Right — Trajectory + Quote */}
            <div className="space-y-4">

              {/* Empty state */}
              {!running && !result && (
                <div className="rounded-xl p-8 text-center"
                     style={{ background: 'var(--bg-elevated)', border: '1px solid var(--bg-border)' }}>
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-3"
                       style={{ background: '#1C1208', border: '1px solid #3D2800' }}>
                    <Zap size={18} style={{ color: 'var(--amber)' }} />
                  </div>
                  <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>
                    Agent ready
                  </p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Click "Run Agent" to process this email
                  </p>
                </div>
              )}

              {/* Running state */}
              {running && (
                <div className="rounded-xl p-6"
                     style={{ background: 'var(--bg-elevated)', border: '1px solid var(--amber)44' }}>
                  <div className="flex items-center gap-3 mb-4">
                    <Loader2 size={14} className="animate-spin" style={{ color: 'var(--amber)' }} />
                    <span className="text-sm font-medium" style={{ color: 'var(--amber)' }}>
                      Agent running
                    </span>
                    <span className="text-xs ml-auto mono" style={{ color: 'var(--text-muted)' }}>
                      {(elapsed / 1000).toFixed(1)}s
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    {[0,1,2].map(i => (
                      <div key={i} className="w-2 h-2 rounded-full"
                           style={{ background: 'var(--amber)', animation: 'pulseDot 1.4s ease-in-out infinite',
                                    animationDelay: `${i * 0.16}s` }} />
                    ))}
                    <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>
                      Calling tools...
                    </span>
                  </div>
                </div>
              )}

              {/* Trajectory panel */}
              {result && (
                <div className="rounded-xl overflow-hidden animate-fade-in"
                     style={{ border: '1px solid var(--bg-border)' }}>
                  <div className="px-5 py-3 flex items-center justify-between"
                       style={{ background: 'var(--bg-elevated)', borderBottom: '1px solid var(--bg-border)' }}>
                    <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
                      Agent Trajectory
                    </span>
                    <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                      <span>{result.tool_calls.length} tool calls</span>
                      <span>{result.total_latency_ms}ms total</span>
                    </div>
                  </div>
                  <div className="p-2" style={{ background: 'var(--bg-secondary)' }}>
                    {result.tool_calls.map((tc, i) => (
                      <ToolCallRow key={i} tc={tc} index={i} />
                    ))}
                  </div>
                </div>
              )}

              {/* Quote table */}
              {result?.quote && <QuoteTable result={result} />}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
