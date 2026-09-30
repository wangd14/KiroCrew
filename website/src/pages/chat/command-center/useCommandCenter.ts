import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../../api/client'
import { fmtList } from '../../../i18n/format'
import { i18nT } from '../../../i18n/t'
import { useAppSelector } from '../../../store'
import type { Artifact, SubagentActivity } from '../../../types'
import { buildCommandCenter, effectiveApprovalMode, scopedSlots, slotKey, type PendingQuestion, type WorkItem } from './model'

export const TASK_DASHBOARD_TAG = 'task-dashboard'
/** Names of the optional sources that can fail. Literal keys, so the catalog
 * tooling sees every one. */
const MISSING_SOURCE_KEYS = {
  runs: 'commandCenter.source_runs',
  work: 'commandCenter.source_work',
  views: 'commandCenter.source_views',
} as const

/** One notice naming every optional source that failed, so the reassurance is
 * said once however many are missing; null when none is. */
export function missingSourcesNotice(missing: readonly (keyof typeof MISSING_SOURCE_KEYS)[]): string | null {
  if (!missing.length) return null
  return i18nT('commandCenter.partial_sources', { sources: fmtList(missing.map(name => i18nT(MISSING_SOURCE_KEYS[name]))) })
}
const EMPTY_AGENTS: Record<string, SubagentActivity> = {}
const settledLatches = new Map<string, boolean>()

export function __resetSettledLatchesForTests() {
  settledLatches.clear()
}

/** Shared query keys let the dock and panel observe one read, not one per worker.
 * Nothing here polls: every source is refreshed by the frame that announces its
 * change (`approval*`, `question_card*`, `artifact_update`, the crew log's
 * `slot_projection` for the work board, workflow events into the store) and all
 * of them again on reconnect, so an open chat tab costs no periodic requests. */
export function useCommandCenter(root: string | null, enabled = true, scope: 'task' | 'fleet' = 'task', { dock = false }: { dock?: boolean } = {}) {
  const slots = useAppSelector(s => s.dashboard.slots)
  const approvalMode = useAppSelector(s => s.dashboard.approvalMode)
  const activeSlot = useAppSelector(s => s.chat.activeSlot)
  const liveAgents = useAppSelector(s => s.chat.subagents)
  const background = useAppSelector(s => s.chat.slotActivity)
  const liveWorkflows = useAppSelector(s => s.chat.workflowRuns)
  const connected = useAppSelector(s => s.dashboard.connected)
  const fleet = scope === 'fleet'
  const scoped = useMemo(() => fleet ? slots : root ? scopedSlots(slots, root) : [], [slots, root, fleet])
  const canRead = enabled && (fleet || !!root && scoped.length > 0)
  const scopedKeySet = useMemo(() => new Set(scoped.map(s => s.key)), [scoped])
  // The app-wide policy is never-stale; a finite staleTime here made every
  // window focus re-read all of these. Frames and reconnect own freshness; a
  // failed source also re-reads on focus, since its frame may never come.
  const sourceOptions = {
    enabled: canRead, staleTime: 3_000,
    refetchOnWindowFocus: (query: { state: { status: string } }) => query.state.status === 'error',
  }
  const questions = useQuery({ queryKey: ['command-center', 'questions'], queryFn: api.pendingQuestions, ...sourceOptions })
  // Only the mounted owner's actively drafted STATELESS cards survive retirement.
  // This is presentation continuity, never a cache of live approval/ask authority.
  const draftScope = JSON.stringify([scope, root])
  const [drafts, setDrafts] = useState<{ scope: string; cards: Record<string, PendingQuestion> }>({ scope: draftScope, cards: {} })
  if (drafts.scope !== draftScope) setDrafts({ scope: draftScope, cards: {} })
  const onQuestionDraftChange = (question: PendingQuestion, active: boolean) => {
    if (question.ask_id || !question.card_id) return
    const id = JSON.stringify([slotKey(question.slot), question.card_id])
    setDrafts(previous => {
      // A departing card's cleanup must not clear a new scope's draft.
      if (previous.scope !== draftScope) return previous
      if (active) return previous.cards[id] === question ? previous : { ...previous, cards: { ...previous.cards, [id]: question } }
      if (!previous.cards[id]) return previous
      const cards = { ...previous.cards }
      delete cards[id]
      return { ...previous, cards }
    })
  }
  const visibleQuestions = useMemo(() => {
    const live = questions.data || []
    const ids = new Set(live.map(q => JSON.stringify([slotKey(q.slot), q.card_id])))
    return [...live, ...Object.values(drafts.scope === draftScope ? drafts.cards : {}).filter(q => !ids.has(JSON.stringify([slotKey(q.slot), q.card_id])))]
  }, [questions.data, drafts, draftScope])
  // The same inventory the app shell already keeps: one cache, one request per frame.
  const approvals = useQuery({ queryKey: ['global-approvals'], queryFn: () => api.approvals(), ...sourceOptions })
  const workflows = useQuery({ queryKey: ['command-center', 'workflows'], queryFn: api.workflowRuns, ...sourceOptions })
  const work = useQuery({
    queryKey: ['command-center', root, 'work'],
    queryFn: () => api.sessionWorkProjection(root!) as Promise<{ value?: { items: WorkItem[]; omitted?: number } }>,
    // The dock is mounted in every chat and the board is a whole-log fold, so it
    // reads the board only for a team (workers are what feed it); the panel,
    // opened on purpose, always does.
    ...sourceOptions, enabled: canRead && !fleet && (!dock || scoped.length > 1),
  })
  const artifacts = useQuery({
    queryKey: ['command-center', 'artifacts'],
    queryFn: () => api.artifacts({ tag: TASK_DASHBOARD_TAG }) as Promise<{ artifacts?: Artifact[] }>,
    ...sourceOptions,
  })
  const model = useMemo(() => {
    const subagents = Object.fromEntries(scoped.map(s => [s.key,
      s.key === activeSlot ? liveAgents : background?.[s.key]?.subagents || EMPTY_AGENTS,
    ]))
    // REST restores completed runs after a reload; live events win until the next
    // authoritative snapshot read. Neither an unavailable endpoint nor an idle slot is success.
    const runs = new Map((workflows.data?.runs || []).map(r => [r.run_id, r]))
    for (const r of Object.values(liveWorkflows || {})) {
      runs.set(r.run_id, { ...runs.get(r.run_id), run_id: r.run_id, name: r.name, status: r.status,
        session_key: r.sessionKey || runs.get(r.run_id)?.session_key || '', error: r.error })
    }
    return buildCommandCenter({ root: fleet ? null : root, slots: scoped, subagents, approvalMode, workflows: [...runs.values()],
      questions: visibleQuestions, approvals: approvals.data || [], work: work.isEnabled ? work.data?.value : undefined })
  }, [root, scoped, activeSlot, liveAgents, background, liveWorkflows, workflows.data, visibleQuestions, approvals.data, work.data, work.isEnabled, approvalMode, fleet])
  const dashboards = (artifacts.data?.artifacts || []).filter(a =>
    (a.kind === 'html' || a.kind === 'widget') && a.tags.includes(TASK_DASHBOARD_TAG)
    && !!a.session_key && scopedKeySet.has(slotKey(a.session_key)),
  )
  const sources = [questions, approvals, workflows, ...(work.isEnabled ? [work] : []), artifacts]
  // Questions and approvals are what a person must act on; the rest decorate.
  // An optional source failing (workflows answer 503 while their service starts)
  // must not hide fresh decisions behind a stale notice.
  const required = [questions, approvals]
  const loading = canRead && sources.some(q => q.isPending)
  const stale = canRead && (!connected || required.some(q => q.isError))
  // Which optional source failed while decisions are fresh, so the notice can
  // name what is missing rather than hand the person a vague uncertainty. Only
  // once both decision reads have answered, since the notice vouches for them.
  const missing = canRead && connected && required.every(q => q.isSuccess)
    ? ([['runs', workflows], ['work', work], ['views', artifacts]] as const)
      .filter(([, q]) => q.isEnabled && q.isError).map(([name]) => name)
    : []
  // Only a complete, current read OF A READABLE SCOPE may declare the task
  // over: a half-loaded or disconnected inventory looks settled because it is
  // empty, and so does a scope with no slots yet — before the first slot list
  // lands nothing is loading, nothing is stale and an empty model is vacuously
  // settled, so an ungated verdict would arm on the first render, before any
  // read. A read with an optional source missing is not complete either: the
  // runs it could not see may be the ones still working. The verdict is then
  // LATCHED for this root: a websocket drop, a transient source error or a
  // remount's loading window must not bring the dock back for a task that is
  // over. Only evidence of new work releases it — a complete read that shows
  // something running, blocked or asking, or a live slot state that already
  // says someone is waiting on the user (`attention` reads the slot flags, so
  // it needs no completed read to be current). The release half is gated the
  // same way: with no readable scope the model is built from an empty slot
  // list plus whatever work-board data the query cache still holds, and that
  // must not count as a read that shows new work.
  const complete = canRead && !loading && !stale && missing.length === 0
  const [, rerenderSettledLatch] = useState(0)
  const latched = settledLatches.get(draftScope) || false
  const unsettledNow = (complete && !model.settled) || model.attention.length > 0
  const finished = latched ? !unsettledNow : complete && model.settled
  if (finished !== latched) {
    if (finished) settledLatches.set(draftScope, true)
    else settledLatches.delete(draftScope)
    rerenderSettledLatch(version => version + 1)
  }
  return {
    ...model, dashboards, connected, onQuestionDraftChange,
    approvalMode: effectiveApprovalMode(approvalMode, slots.find(s => s.key === root)),
    loading, stale, missing,
    // Real clock from completed reads. A websocket connection alone doesn't
    // establish that a server-side question/approval inventory is up to date.
    updatedAt: Math.min(...required.map(q => q.dataUpdatedAt)),
    approvalCount: model.attention.filter(a => a.kind === 'approval').length,
    // Anything waiting on the user makes the dock relevant, whatever its kind: the
    // Needs you tile is the dock's reason to exist. A lone session's TODO list is
    // deliberately NOT enough: TaskProgressBar already shows that plan above the
    // composer, and a second readout of the same numbers would only repeat it.
    relevant: scoped.length > 1 || model.nodes.some(n => n.kind !== 'session') || model.workItems.length > 0 || dashboards.length > 0 || model.attention.length > 0,
    finished,
  }
}

export type CommandCenterData = ReturnType<typeof useCommandCenter>
