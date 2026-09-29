/**
 * A thread's pane must DELEGATE to the chat surface, never re-implement it.
 *
 * Version 1 threads had a pane of their own, and every capability the chat
 * composer and transcript already had was missing from it: the composer did not
 * auto-grow (a second line hid the first), a turn in a thread could not call a
 * tool, and the composer offered no steer/queue chooser while its own turn ran.
 * Each was the same defect wearing a different hat -- a bespoke surface beside
 * the real one.
 *
 * Round 2's answer is structural: `ThreadPanel` renders `ChatPane`, which renders
 * the real `ChatInput`, so a thread inherits the composer and the transcript
 * whole. That makes these behaviours untestable by feature: there is no
 * thread-specific composer to assert auto-grow on, because it is the chat's
 * composer. What IS assertable, and what these tests pin, is the delegation
 * itself plus the presence of each capability in the component the thread
 * inherits -- so a future change that gives the thread its own composer, or drops
 * a capability from the shared one, fails here instead of in a person's hands.
 *
 * Source-level for the reason the `ctx.threads` contract is: a render test would
 * have to mount the whole pane with a live query client and a store to assert the
 * absence of a control, and `ThreadPanel.test.tsx` already mocks `ChatPane` away
 * (correctly -- its own contract is the rows above the pane).
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

const SRC = join(__dirname, '..')
const read = (rel: string) => readFileSync(join(SRC, rel), 'utf8')

const THREAD_PANEL = 'pages/members/ThreadPanel.tsx'
const CHAT_PANE = 'components/ChatPane.tsx'
const CHAT_INPUT = 'components/ChatInput.tsx'

describe('a thread inherits the chat surface', () => {
  it('renders ChatPane rather than a composer of its own', () => {
    const panel = read(THREAD_PANEL)
    expect(panel).toMatch(/import ChatPane from/)
    expect(panel).toMatch(/<ChatPane/)
    // No bespoke composer in the pane: a `<textarea>` here would be the version 1
    // shape returning, and it is what carried every missing capability.
    expect(panel).not.toMatch(/<textarea/)
  })

  it('reaches the real ChatInput through ChatPane', () => {
    const pane = read(CHAT_PANE)
    expect(pane).toMatch(/import ChatInput(,| )/)
    expect(pane).toMatch(/<ChatInput/)
  })

  it('inherits a composer that auto-grows with its content', () => {
    // `applyHeight` measures on an off-screen twin and writes one final height,
    // so a second line grows the box instead of hiding the first.
    expect(read(CHAT_INPUT)).toMatch(/from '\.\/chat-input\/sizing'/)
    const sizing = read('components/chat-input/sizing.ts')
    expect(sizing).toMatch(/function applyHeight\(/)
    expect(sizing).toMatch(/scrollHeight/)
  })

  it('inherits the steer/queue chooser shown while a turn runs', () => {
    expect(read(CHAT_INPUT)).toMatch(/from '\.\/chat-input\/busySend'/)
    const busy = read('components/chat-input/busySend.tsx')
    expect(busy).toMatch(/import BusySendButton/)
    expect(busy).toMatch(/<BusySendButton/)
  })

  it('inherits the transcript row set that draws tool calls', () => {
    // A turn inside a thread runs the ordinary loop, so its tool rows are drawn
    // by the shared renderers rather than by anything thread-specific.
    const pane = read(CHAT_PANE)
    expect(pane).toMatch(/createTranscriptRenderers/)
  })

  it('pops out by switching the slot, so the full page IS the thread', () => {
    // Pushing `/chat?sid=<thread>` from inside this page reaches nothing: the
    // session controller's sid effect honours only a POP, and the activeSlot ->
    // URL effect then rewrites the URL back to the slot on screen -- the PARENT.
    // Measured on a pod: the pop-out landed on `?sid=<parent slot>`.
    const page = read('pages/ChatPage.tsx')
    const handler = page.slice(page.indexOf('onOpenFull={(threadSlot)'))
    const body = handler.slice(0, handler.indexOf('}}'))
    // `announceOnMissing`: a clicked listed session, so a 404 belongs on screen.
    expect(body).toMatch(/dispatch\(switchSlot\(\{ key: threadSlot, announceOnMissing: true \}\)\)/)
    expect(body).not.toMatch(/navigate\(/)
  })
})
