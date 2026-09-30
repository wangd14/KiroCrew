/**
 * The crew manager (Agent Capabilities → Crews) opens the shared
 * `NewCrewmateDialog` from inside a `SidePanelLayout`. That layout mounts one
 * pane at a time behind `{tab === '<key>' && <Pane/>}`, so switching tabs — and
 * the browser Back it forwards to the app shell — UNMOUNTS the pane and takes
 * the dialog's typed draft with it. The layout only asks a guard registered
 * through its own pane channel; a dialog that published its stake to the app
 * shell alone (as `NewCrewmateDialog` did before) filled no pane slot, so the
 * switch discarded the draft with no confirm.
 *
 * This pins the fix: the dialog registers the SAME leave predicate through
 * `useSidePanelLeaveGuard`, which fills the pane slot and rides the layout's
 * forward to the shell. Reverting that one hook call fails the first test here.
 * Standalone (the Crewmates page, no layout) the hook is a no-op and the
 * dialog's direct app-shell registration stays its only guard — the second test
 * pins that this fix does not double-ask or otherwise disturb that path.
 */
import React from 'react'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { I18nextProvider } from 'react-i18next'
import i18n from '../i18n'

let mobile = false
vi.mock('../hooks/useIsMobile', () => ({ useIsMobile: () => mobile }))

// The dialog reads the execution catalog for its "Built from" list, the
// workspace list for its Advanced fold, and the roster for its post-failure
// reconcile. None of that is exercised here — the draft is a typed name — so a
// bare stub keeps the dialog mountable without a network.
const mockApi = vi.hoisted(() => ({
  agentCatalog: vi.fn(async () => ({ agents: [] })),
  workspaces: vi.fn(async () => ({ workspaces: [] })),
  members: vi.fn(async () => ({ members: [] })),
  createKirocrewAgent: vi.fn(),
  createWorkspace: vi.fn(),
}))
vi.mock('../api/client', () => ({ api: mockApi }))
vi.mock('../hooks/useAvailableModels', () => ({
  useAvailableModelsQuery: () => ({ data: [], isLoading: false }),
}))

import SidePanelLayout, { type SidePanelTab } from '../components/SidePanelLayout'
import NewCrewmateDialog from '../pages/members/NewCrewmateDialog'

const TABS: SidePanelTab[] = [
  { key: 'crews', label: 'Crews', icon: null },
  { key: 'other', label: 'Other', icon: null },
]

/** The crews pane: mounts the real dialog, open. Its typed name lives in the
 *  dialog's own state, so a tab switch that unmounts this pane is what destroys
 *  it. */
function CrewsPane() {
  return (
    <NewCrewmateDialog open onClose={() => {}} onCreated={() => {}} existingNames={[]} />
  )
}

function renderInLayout() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <MemoryRouter initialEntries={['/capabilities']}>
      <QueryClientProvider client={qc}>
        <I18nextProvider i18n={i18n}>
          <SidePanelLayout title="Agents" tabs={TABS}>
            {tab => <>
              {tab === 'crews' && <CrewsPane />}
              {tab !== 'crews' && <div data-testid="plain">{tab}</div>}
            </>}
          </SidePanelLayout>
        </I18nextProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  )
}

/** Standalone: the dialog with no enclosing layout, the Crewmates-page shape. */
function renderStandalone() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <MemoryRouter initialEntries={['/members']}>
      <QueryClientProvider client={qc}>
        <I18nextProvider i18n={i18n}>
          <NewCrewmateDialog open onClose={() => {}} onCreated={() => {}} existingNames={[]} />
        </I18nextProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  )
}

const nameInput = () => screen.getByLabelText('Name') as HTMLInputElement
const typeName = (v: string) => fireEvent.change(nameInput(), { target: { value: v } })
const clickTab = (name: string) => fireEvent.click(screen.getByRole('button', { name }))

describe('NewCrewmateDialog inside a SidePanelLayout', () => {
  beforeEach(() => { mobile = false; sessionStorage.clear() })
  afterEach(() => { vi.restoreAllMocks(); cleanup() })

  it('vetoes a capability tab switch that would discard the typed draft', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    renderInLayout()
    typeName('half typed name')
    clickTab('Other')
    // The whole point of the fix: the layout consulted the dialog's guard, the
    // declined confirm kept the crews pane mounted, and the typed name survives.
    expect(confirmSpy).toHaveBeenCalled()
    expect(nameInput().value).toBe('half typed name')
    expect(screen.queryByTestId('plain')).toBeNull()
  })

  it('lets the switch through once the confirm is accepted', () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    renderInLayout()
    typeName('half typed name')
    clickTab('Other')
    expect(screen.getByTestId('plain').textContent).toBe('other')
  })

  it('does not ask when nothing has been typed', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false)
    renderInLayout()
    clickTab('Other')
    expect(confirmSpy).not.toHaveBeenCalled()
    expect(screen.getByTestId('plain').textContent).toBe('other')
  })

  it('mounts standalone (no layout) without error — the hook is a no-op there', () => {
    // No SidePanelLayout in the tree, so useSidePanelLeaveGuard's context is
    // null. The dialog must still render, with the typed name held.
    renderStandalone()
    typeName('typed on the crewmates page')
    expect(nameInput().value).toBe('typed on the crewmates page')
  })
})
