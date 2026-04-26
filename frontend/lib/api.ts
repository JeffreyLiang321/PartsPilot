const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface Email {
  email_id:     string
  subject:      string
  body:         string
  email_type:   string
  difficulty:   string
  customer_id:  string | null
  company_name: string | null
  contact_name: string | null
}

export interface ToolCall {
  tool:        string
  input:       Record<string, unknown>
  output:      Record<string, unknown>
  latency_ms:  number
  iteration:   number
}

export interface LineItem {
  mpn:             string
  manufacturer:    string
  description:     string
  requested_qty:   number
  available_qty:   number
  unit_price_usd:  number
  total_price_usd: number
  status:          'in_stock' | 'low_stock' | 'out_of_stock' | 'substituted' | 'lead_time'
  substitute_for:  string | null
  lead_time_days:  number
  warehouse:       string
  notes:           string
}

export interface Quote {
  status:               string
  line_items:           LineItem[]
  subtotal_usd:         number
  currency:             string
  valid_days:           number
  notes:                string
  clarification_needed: string | null
}

export interface AgentResult {
  trace_id:         string
  quote:            Quote | null
  final_message:    string
  tool_calls:       ToolCall[]
  total_latency_ms: number
  iterations:       number
  error:            string | null
}

export async function fetchEmails(): Promise<Email[]> {
  const res = await fetch(`${API_BASE}/emails`, { cache: 'no-store' })
  if (!res.ok) throw new Error('Failed to fetch emails')
  const data = await res.json()
  return data.emails
}

export async function fetchEmail(emailId: string): Promise<Email> {
  const res = await fetch(`${API_BASE}/emails/${emailId}`, { cache: 'no-store' })
  if (!res.ok) throw new Error('Failed to fetch email')
  return res.json()
}

export async function runAgent(
  emailBody: string,
  emailId: string | null,
  customerId: string | null,
): Promise<AgentResult> {
  const res = await fetch(`${API_BASE}/agent/run`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      email_body:  emailBody,
      email_id:    emailId,
      customer_id: customerId,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Agent run failed')
  }
  return res.json()
}
