import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, fireEvent, waitFor, within } from '@testing-library/react'
import { renderWithProviders } from '../../test/helpers'

/* The Crewmates page's own "Delete crewmate": the trash button in the thread
 * header opens a confirmation, the confirmation issues the crew
 * manager's DELETE and refetches the roster, Cancel issues nothing, the default
 * crew is never offered the control (the server refuses that delete with 409),
 * and a refused delete shows the server's own words inside the dialog. */

vi.mock('../../api/client', () => ({
  api: {
    members: vi.fn(),
    teams: { list: vi.fn(() => Promise.resolve({ teams: [] })) },
    memberThread: vi.fn((slug: string) =>
      Promise.resolve({ slot_key: 'member-' + slug, slug, member: slug, created: true }),
    ),
    memberActivity: vi.fn(() => Promise.resolve({ slug: '', member: '', capped: false, entries: [] })),
    memberProjections: vi.fn(() => Promise.resolve({ asOfSeq: 0, values: {} })),
    memberBriefing: vi.fn(() => Promise.resolve({ slug: '', member: '', supported: true, text: '', updated_ts: null, redacted: false, truncated: false })),
    sessionCrewLogProjections: vi.fn(() => Promise.resolve({ folds: {}, resolved: true, writesDrained: true })),
    crons: vi.fn(() => Promise.resolve({ jobs: [] })),
    webhooks: vi.fn(() => Promise.resolve({ tokens: [] })),
    // The page reads the default crew through the shared ['default-agent']
    // query: it decides who gets the delete control and who stays listed.
    defaultAgent: vi.fn(() => Promise.resolve({ default_agent: 'kirocrew' })),
    deleteKirocrewAgent: vi.fn(() => Promise.resolve({ ok: true })),
    autonudgeList: vi.fn(() => Promise.resolve({ enabled: true, loops: [] })),
    memberPanel: vi.fn(() => Promise.resolve({ panel: null, html: null })),
    kirocrewConfig: vi.fn(() => Promise.resolve({})),
  },
}))

vi.mock('../chat/ActivityViewer', () => ({ default: () => null }))
vi.mock('../chat/FilesHomePanel', () => ({ default: () => null }))
vi.mock('../chat/FolderPanel', () => ({ default: () => null }))
vi.mock('../../components/DiffPanel', () => ({ default: () => null }))
vi.mock('../../components/MarkdownPanel', () => ({ default: () => null }))
vi.mock('../../components/ArtifactPanel', () => ({ default: () => null }))
vi.mock('../../components/WebPreviewPanel', () => ({ default: () => null }))
vi.mock('../../components/McpAppFrame', () => ({ default: () => null }))
vi.mock('../../components/CliPanel', () => ({
  default: () => null,
  disposeTerminalSession: vi.fn(),
  useDeleteTerminalSession: () => ({ mutate: vi.fn() }),
}))
vi.mock('../../utils/terminalRegistry', () => ({
  useTerminalEnabled: () => true,
  useTerminalTitle: () => 'Terminal',
}))
vi.mock('../../hooks/useDevMode', () => ({ useDevMode: () => false }))
vi.mock('../../components/ChatPane', () => ({
  default: ({ slotKey }: { slotKey: string }) => <div data-testid="chat-pane-stub">{slotKey}</div>,
}))

import { api } from '../../api/client'
import MembersPage from './MembersPage'

function row(name: string, overrides: Record<string, unknown> = {}) {
  return {
    name,
    slug: name,
    slot_key: '',
    running: false,
    kiro_agent: name,
    workspace: 'default',
    memory_store: 'default',
    model: '',
    source: 'kirocrew',
    starred: false,
    dashboard_created: true,
    has_dm_message: false,
    ...overrides,
  }
}

/** Two created crewmates and the default crew, which is not "created" (its
 *  memory is the global store) but is listed because it is the default. */
const ROSTER = [
  row('oncall', { display_name: 'On-call' }),
  row('radar'),
  row('kirocrew', { source: 'builtin', dashboard_created: false }),
]

const PANE_READY = { timeout: 5000 }

async function openMember(name: string, members = ROSTER) {
  ;(api.members as ReturnType<typeof vi.fn>).mockResolvedValue({ members })
  const utils = renderWithProviders(<MembersPage />, { route: `/members?member=${name}` })
  await waitFor(() => expect(api.members).toHaveBeenCalled())
  await screen.findByTestId('chat-pane-stub', undefined, PANE_READY)
  await screen.findByTestId('member-thread-header', undefined, PANE_READY)
  return utils
}

const deleteButton = () => screen.queryByRole('button', { name: 'Delete crewmate' })

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  Object.defineProperty(window, 'innerWidth', { value: 1440, configurable: true, writable: true })
})

describe('MembersPage delete crewmate', () => {
  it('offers the control in the thread header and opens a confirmation naming the crewmate', async () => {
    await openMember('oncall')
    const btn = deleteButton()
    expect(btn).toBeInTheDocument()
    // A side control of the thread header, not part of the identity pill.
    expect(within(screen.getByTestId('member-thread-header')).getByTestId('member-delete-button')).toBe(btn)
    expect(within(screen.getByTestId('member-identity-pill')).queryByTestId('member-delete-button')).toBeNull()
    fireEvent.click(btn!)
    const dialog = await screen.findByRole('dialog')
    // Quoted display label, and the two things the copy promises: what goes
    // (roster, team, work log, picture, template copy) and what stays (chat
    // history, memory, workspace files).
    expect(within(dialog).getByTestId('delete-crewmate-body')).toHaveTextContent('Deleting “On-call” removes it from the roster')
    expect(within(dialog).getByTestId('delete-crewmate-body')).toHaveTextContent('chat history, memory and workspace files are kept')
    expect(within(dialog).getByRole('button', { name: 'Delete “On-call”' })).toBeInTheDocument()
    expect(api.deleteKirocrewAgent).not.toHaveBeenCalled()
  })

  it('confirm -> DELETE by crew NAME (not label), roster refetched, dialog closed', async () => {
    await openMember('oncall')
    const before = (api.members as ReturnType<typeof vi.fn>).mock.calls.length
    // The re-read after the delete answers without the row.
    ;(api.members as ReturnType<typeof vi.fn>).mockResolvedValue({ members: ROSTER.filter((m) => m.name !== 'oncall') })
    fireEvent.click(deleteButton()!)
    fireEvent.click(await screen.findByTestId('delete-crewmate-confirm'))
    await waitFor(() => expect(api.deleteKirocrewAgent).toHaveBeenCalledWith('oncall'))
    await waitFor(() => expect((api.members as ReturnType<typeof vi.fn>).mock.calls.length).toBeGreaterThan(before))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    // The page's own gone-crewmate handling takes over: the thread gives way
    // to another crewmate and the swap is announced, as for any deletion.
    await waitFor(() => expect(screen.getByTestId('member-gone-notice')).toHaveTextContent('“oncall” is no longer on the roster'), PANE_READY)
    expect(within(screen.getByTestId('member-roster')).queryByText('On-call')).toBeNull()
  })

  it('cancel -> nothing is deleted and the thread stays', async () => {
    await openMember('oncall')
    fireEvent.click(deleteButton()!)
    await screen.findByRole('dialog')
    fireEvent.click(screen.getByTestId('delete-crewmate-cancel'))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(api.deleteKirocrewAgent).not.toHaveBeenCalled()
    expect(screen.getByTestId('chat-pane-stub')).toHaveTextContent('member-oncall')
  })

  it('the default crew gets no delete control', async () => {
    await openMember('kirocrew')
    expect(screen.getByTestId('member-identity-pill')).toBeInTheDocument()
    expect(deleteButton()).toBeNull()
    expect(screen.queryByTestId('member-delete-button')).toBeNull()
  })

  it('a hidden row reached by the search can be opened and deleted', async () => {
    // pkg-tool: sync-generated, never chatted -> hidden from the default roster.
    const roster = [...ROSTER, row('pkg-tool', { source: 'package', dashboard_created: false, has_dm_message: false })]
    await openMember('radar', roster)
    const list = screen.getByTestId('member-roster')
    expect(within(list).queryByText('pkg-tool')).toBeNull()
    fireEvent.change(screen.getByTestId('member-search'), { target: { value: 'pkg' } })
    fireEvent.click(await within(list).findByText('pkg-tool'))
    await waitFor(() => expect(screen.getByTestId('chat-pane-stub')).toHaveTextContent('member-pkg-tool'), PANE_READY)
    fireEvent.click(deleteButton()!)
    fireEvent.click(await screen.findByTestId('delete-crewmate-confirm'))
    await waitFor(() => expect(api.deleteKirocrewAgent).toHaveBeenCalledWith('pkg-tool'))
  })

  it('a refused delete shows the server text inside the dialog and keeps it open', async () => {
    ;(api.deleteKirocrewAgent as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("Cannot delete default agent 'radar'. Change default_agent first."),
    )
    await openMember('radar')
    fireEvent.click(deleteButton()!)
    fireEvent.click(await screen.findByTestId('delete-crewmate-confirm'))
    const notice = await screen.findByTestId('delete-crewmate-error')
    expect(notice).toHaveTextContent("Cannot delete default agent 'radar'")
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByTestId('chat-pane-stub')).toHaveTextContent('member-radar')
  })
})
